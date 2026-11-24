"""Filesystem store for analyses. One directory per analysis:

    <storage_dir>/<analysis_id>/
        before.<ext>  after.<ext>   uploaded images (names fixed by the server)
        predictions.json            sanitised Phase 1 output (the single source of truth)
        damage_map.png  comparison.jpg
        questions.json              answered questions (official records + internal trace)
        report.html                 cached report
        meta.json                   written last: its presence means "complete"

Analysis ids are random uuid4 hex, directories are created with exist_ok=False, so
an analysis can never overwrite another. Ids from URLs must match a strict pattern
and resolve inside the storage root, which blocks path traversal.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import threading
import time
import uuid
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from backend.errors import ApiError

ID_RE = re.compile(r"^[0-9a-f]{32}$")
QUESTION_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json_atomic(path: Path, data) -> None:
    """Write to a temp file, then rename over `path`. On Windows the rename fails with PermissionError while another
    thread has the target open (e.g. a status poll), so it is retried briefly instead of failing the caller."""
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    for attempt in range(60):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 59:
                tmp.unlink(missing_ok=True)
                raise
            time.sleep(0.02)


class AnalysisStore:
    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._locks = defaultdict(threading.Lock)
        self._locks_guard = threading.Lock()

    # ---- ids / paths ------------------------------------------------------
    def new_dir(self) -> tuple[str, Path]:
        analysis_id = uuid.uuid4().hex
        d = self.root / analysis_id
        d.mkdir(exist_ok=False)            # raises FileExistsError rather than ever overwriting
        return analysis_id, d

    def path(self, analysis_id: str) -> Path:
        if not isinstance(analysis_id, str) or not ID_RE.fullmatch(analysis_id):
            raise ApiError(404, "analysis_not_found", "No analysis with this id.")
        d = (self.root / analysis_id).resolve()
        if d.parent != self.root or not (d / "meta.json").is_file():   # traversal guard + completeness
            raise ApiError(404, "analysis_not_found", "No analysis with this id.")
        return d

    def discard(self, d: Path) -> None:
        if d.resolve().parent == self.root:
            shutil.rmtree(d, ignore_errors=True)

    # ---- reads / writes ---------------------------------------------------
    def meta(self, analysis_id: str) -> dict:
        return json.loads((self.path(analysis_id) / "meta.json").read_text(encoding="utf-8"))

    def predictions(self, analysis_id: str) -> dict:
        return json.loads((self.path(analysis_id) / "predictions.json").read_text(encoding="utf-8"))

    def image_path(self, analysis_id: str, role: str) -> Path:
        d = self.path(analysis_id)
        for ext in (".jpg", ".png"):
            p = d / f"{role}{ext}"
            if p.is_file():
                return p
        raise ApiError(404, "image_not_found", f"The {role} image of this analysis is missing.")

    def questions(self, analysis_id: str) -> dict:
        p = self.path(analysis_id) / "questions.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}

    @contextmanager
    def locked(self, analysis_id: str):
        """Serialise read-modify-write of one analysis' mutable files."""
        with self._locks_guard:
            lock = self._locks[analysis_id]
        with lock:
            yield

    def save_questions(self, analysis_id: str, records: dict) -> None:
        write_json_atomic(self.path(analysis_id) / "questions.json", records)
