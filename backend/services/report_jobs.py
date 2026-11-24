"""Report generation jobs: gather real analysis data, render the PDF in a background worker, track status.

One worker thread renders PDFs, so report generation never competes with inference for the GPU lock and several
requests queue rather than run in parallel. A second request for an analysis whose report is already being generated
returns that report instead of starting a duplicate.
"""
from __future__ import annotations

import hashlib
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

import cv2
import numpy as np

from backend.config import Settings
from backend.services import explain, pdf_report, report_content, spatial_service
from backend.utils.report_store import ReportStore
from backend.utils.storage import AnalysisStore, utcnow
from inference import alignment as align
from reasoning.damage_map import render_damage_map
from reasoning.engine import answer_query
from reasoning.official import to_official
from reasoning.questions import parse_question
from reasoning.report_facts import build_report_facts

log = logging.getLogger("eagle.reports")

STANDARD_QUERIES = [   # (id, question, draw an evidence crop?)
    ("std_no_damage", "How many undamaged buildings are there in the image?", False),
    ("std_major", "How many buildings with major damage are there in the image?", False),
    ("std_destroyed", "Count the destroyed buildings in the image.", True),
    ("std_damaged", "Count the damaged buildings (major or destroyed) in the image.", True),
    ("std_quadrant", "Which quadrant has the most damaged buildings?", True),
    ("std_left_right", "Is the left half of the image more damaged than the right half?", True),
    ("std_top_bottom", "Is the top half of the image more damaged than the bottom half?", True),
]
MAX_USER_CROPS = 8


def _read(path) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("stored image could not be decoded")
    return img


def evidence_crop(scene, post: np.ndarray, region, ids) -> np.ndarray | None:
    if region is None:
        return None
    m = render_damage_map(post, scene, evidence_region=region, highlight_ids=tuple(ids), legend=False, draw_ids=False, draw_split_lines=False)
    x1, y1, x2, y2 = region
    pad = 30
    h, w = m.shape[:2]
    c = m[max(0, y1 - pad):min(h, y2 + pad), max(0, x1 - pad):min(w, x2 + pad)]
    return c if c.size else None


def gather(settings: Settings, store: AnalysisStore, analysis_id: str, report_id: str,
           progress: Callable[[str, float], None]) -> pdf_report.ReportData:
    progress("Loading analysis", 0.05)
    meta, preds = store.meta(analysis_id), store.predictions(analysis_id)
    scene = spatial_service.load_scene(store, analysis_id)
    facts = build_report_facts(scene, preds["alignment"], preds["warnings"])

    progress("Reading images", 0.12)
    pre, post = _read(store.image_path(analysis_id, "before")), _read(store.image_path(analysis_id, "after"))
    al = preds["alignment"]
    pre_used = align.register_pre_to_post(pre, al["dx"], al["dy"]) if al.get("registered") else pre

    progress("Rendering three-class damage map", 0.2)
    map_full = render_damage_map(post, scene)
    map_plain = render_damage_map(post, scene, legend=False)

    progress("Computing findings and recommendations", 0.28)
    seconds = float(meta["parameters"]["inference_seconds"])
    findings, recs = report_content.build_findings(facts), report_content.build_recommendations(facts)

    progress("Writing explanation", 0.32)
    # If the LLM was requested but cannot be used (no key, network down, unverifiable output) the template is used
    # and the reason is recorded on the report; it is never silently ignored.
    explanation = explain.explain(facts, seconds, enabled=settings.llm_enabled, model=settings.llm_model, timeout=settings.llm_timeout_s)

    progress("Answering standard questions and drawing evidence", 0.4)
    by_id = {b.building_id: b for b in scene.buildings}

    def item(qid, text, ans, official, crop=True):
        return {"id": qid, "question": text, "answer": official["answer"], "region": official["region"], "qtype": ans.qtype,
                "supporting_ids": list(ans.supporting_ids),
                "crop": evidence_crop(scene, post, ans.region, ans.supporting_ids) if crop else None}

    qa_standard = []
    for qid, text, crop in STANDARD_QUERIES:
        ans = answer_query(scene, parse_question(text))
        qa_standard.append(item(qid, text, ans, to_official(qid, ans), crop))

    qa_user = []
    for n, (qid, rec) in enumerate(store.questions(analysis_id).items()):
        ev = rec["internal"]["evidence"]
        ids = [i for i in ev["supporting_building_ids"] if i in by_id]
        region = ev["region"]
        crop = evidence_crop(scene, post, region, ids) if n < MAX_USER_CROPS else None
        qa_user.append({"id": qid, "question": rec.get("question_text") or "", "answer": rec["official"]["answer"],
                        "region": rec["official"]["region"], "qtype": rec["internal"]["question_type"],
                        "supporting_ids": ids, "crop": crop})

    return pdf_report.ReportData(
        report_id=report_id, analysis_id=analysis_id, generated_utc=utcnow(), meta=meta, preds=preds, scene=scene, facts=facts,
        pre=pre_used, post=post, map_full=map_full, map_plain=map_plain, findings=findings, recommendations=recs,
        explanation=explanation, qa_user=qa_user, qa_standard=qa_standard, seconds=seconds)


class ReportService:
    def __init__(self, settings: Settings, analyses: AnalysisStore, reports: ReportStore):
        self.settings, self.analyses, self.reports = settings, analyses, reports
        self.generator = pdf_report.build_pdf            # replaceable (tests inject failures)
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="report")
        self._inflight: dict[str, str] = {}
        self._lock = threading.Lock()

    def recover(self) -> int:
        return self.reports.recover_interrupted()

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)

    def submit(self, analysis_id: str) -> tuple[dict, bool]:
        """-> (meta, created). Returns the in-flight report for this analysis instead of starting a duplicate."""
        self.analyses.path(analysis_id)                    # 404 for unknown analysis
        with self._lock:
            rid = self._inflight.get(analysis_id)
            if rid:
                return self.reports.meta(rid), False
            rid, _ = self.reports.new(analysis_id)
            self._inflight[analysis_id] = rid
        self._pool.submit(self._run, rid, analysis_id)
        return self.reports.meta(rid), True

    def _progress(self, rid: str) -> Callable[[str, float], None]:
        def note(stage: str, frac: float) -> None:
            try:                                           # progress is advisory: a status-file hiccup must never fail the report
                self.reports.update(rid, stage=stage, progress=round(min(frac, 0.99), 2))
            except OSError:
                log.warning("could not record progress for report %s", rid)
        return note

    def _run(self, rid: str, analysis_id: str) -> None:
        d = self.reports.path(rid)
        tmp, final = d / "report.pdf.tmp", d / "report.pdf"
        t0 = time.perf_counter()
        try:
            self.reports.update(rid, status="generating", stage="Starting", started_utc=utcnow(), progress=0.02)
            progress = self._progress(rid)
            data = gather(self.settings, self.analyses, analysis_id, rid, progress)
            self.reports.update(rid, explanation_source=data.explanation.source, fallback_reason=data.explanation.fallback_reason)
            pages = self.generator(tmp, data, progress)
            os.replace(tmp, final)
            sha = hashlib.sha256(final.read_bytes()).hexdigest()
            self.reports.update(rid, status="complete", stage="Complete", progress=1.0, finished_utc=utcnow(), pages=pages,
                                bytes=final.stat().st_size, sha256=sha, generation_seconds=round(time.perf_counter() - t0, 2))
        except Exception as exc:                            # never leave a half-written report behind
            log.exception("Report %s for analysis %s failed", rid, analysis_id)
            tmp.unlink(missing_ok=True)
            final.unlink(missing_ok=True)
            self.reports.update(rid, status="failed", stage="Failed", finished_utc=utcnow(),
                                generation_seconds=round(time.perf_counter() - t0, 2),
                                error=f"PDF generation failed ({type(exc).__name__}). The analysis is unaffected; try again. See server logs.")
        finally:
            with self._lock:
                self._inflight.pop(analysis_id, None)
