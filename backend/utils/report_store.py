"""Filesystem store for generated reports: <reports_dir>/<report_id>/{meta.json, report.pdf}.

Report ids are random uuid4 hex; each report gets its own directory (created with exist_ok=False), so a report can never
overwrite another, including an earlier report of the same analysis. Ids from URLs must match a strict pattern and resolve
inside the root (no path traversal). The PDF is written to a temp name and renamed when complete.
"""
from __future__ import annotations

import json
import re
import shutil
import threading
import uuid
from pathlib import Path

from backend.errors import ApiError
from backend.utils.storage import utcnow, write_json_atomic

REPORT_ID_RE = re.compile(r"^[0-9a-f]{32}$")


class ReportStore:
    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def new(self, analysis_id: str) -> tuple[str, Path]:
        rid = uuid.uuid4().hex
        d = self.root / rid
        d.mkdir(exist_ok=False)
        meta = {"report_id": rid, "analysis_id": analysis_id, "status": "queued", "stage": "Queued", "progress": 0.0,
                "created_utc": utcnow(), "started_utc": None, "finished_utc": None, "generation_seconds": None,
                "bytes": None, "sha256": None, "pages": None, "explanation_source": None, "fallback_reason": None, "error": None}
        write_json_atomic(d / "meta.json", meta)
        return rid, d

    def path(self, report_id: str) -> Path:
        if not isinstance(report_id, str) or not REPORT_ID_RE.fullmatch(report_id):
            raise ApiError(404, "report_not_found", "No report with this id.")
        d = (self.root / report_id).resolve()
        if d.parent != self.root or not (d / "meta.json").is_file():
            raise ApiError(404, "report_not_found", "No report with this id.")
        return d

    def meta(self, report_id: str) -> dict:
        d = self.path(report_id)
        with self._lock:                      # readers wait for an in-progress write (see storage.write_json_atomic)
            return json.loads((d / "meta.json").read_text(encoding="utf-8"))

    def update(self, report_id: str, **fields) -> dict:
        with self._lock:
            d = self.path(report_id)
            meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
            meta.update(fields)
            write_json_atomic(d / "meta.json", meta)
            return meta

    def pdf_path(self, report_id: str) -> Path:
        return self.path(report_id) / "report.pdf"

    def list_for_analysis(self, analysis_id: str) -> list[dict]:
        out = []
        for d in self.root.iterdir():
            f = d / "meta.json"
            if d.is_dir() and REPORT_ID_RE.fullmatch(d.name) and f.is_file():
                try:
                    with self._lock:
                        m = json.loads(f.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    continue
                if m.get("analysis_id") == analysis_id:
                    out.append(m)
        return sorted(out, key=lambda m: m["created_utc"], reverse=True)

    def recover_interrupted(self) -> int:
        """Reports left 'queued'/'generating' by a previous process can never finish: mark them failed."""
        n = 0
        for d in self.root.iterdir():
            f = d / "meta.json"
            if d.is_dir() and REPORT_ID_RE.fullmatch(d.name) and f.is_file():
                try:
                    m = json.loads(f.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    continue
                if m.get("status") in ("queued", "generating"):
                    for p in d.glob("*.tmp"):
                        p.unlink(missing_ok=True)
                    self.update(m["report_id"], status="failed", stage="Interrupted", finished_utc=utcnow(),
                                error="Generation was interrupted by a server restart. Generate the report again.")
                    n += 1
        return n

    def discard(self, report_id: str) -> None:
        d = (self.root / report_id).resolve()
        if d.parent == self.root:
            shutil.rmtree(d, ignore_errors=True)
