from __future__ import annotations

import re
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Query, Response
from fastapi.responses import FileResponse

from backend.deps import get_report_service, get_report_store, get_store
from backend.errors import ApiError
from backend.schemas.analysis import ErrorResponse
from backend.schemas.reports import ReportMeta, ReportResponse
from backend.services import report_service
from backend.services.report_jobs import ReportService
from backend.utils.report_store import ReportStore
from backend.utils.storage import AnalysisStore

router = APIRouter(tags=["reports"])
NOT_FOUND = {404: {"model": ErrorResponse, "description": "Unknown analysis or report id"}}


def with_url(meta: dict) -> dict:
    out = dict(meta)
    out["download_url"] = f"/api/reports/{meta['report_id']}/download" if meta["status"] == "complete" else None
    return out


# ---------------------------------------------------------------- PDF reports (Novelty 3)

@router.post("/analysis/{analysis_id}/reports", response_model=ReportMeta, status_code=202, responses=NOT_FOUND,
             summary="Generate the Disaster Damage Assessment Report (PDF)")
def create_report(analysis_id: str, response: Response, svc: ReportService = Depends(get_report_service)):
    """Starts PDF generation in the background and returns immediately with a report id to poll.

    Every call creates a **new** report; earlier reports are kept. If a report for this analysis is already being
    generated, that one is returned (HTTP 200) instead of starting a duplicate."""
    meta, created = svc.submit(analysis_id)
    if not created:
        response.status_code = 200
    return with_url(meta)


@router.get("/analysis/{analysis_id}/reports", response_model=list[ReportMeta], responses=NOT_FOUND,
            summary="Reports generated for an analysis, newest first")
def list_reports(analysis_id: str, analyses: AnalysisStore = Depends(get_store), reports: ReportStore = Depends(get_report_store)):
    analyses.path(analysis_id)
    return [with_url(m) for m in reports.list_for_analysis(analysis_id)]


@router.get("/reports/{report_id}", response_model=ReportMeta, responses=NOT_FOUND, summary="Status of one report")
def report_status(report_id: str, reports: ReportStore = Depends(get_report_store)):
    return with_url(reports.meta(report_id))


@router.get("/reports/{report_id}/download", response_class=Response, summary="Download the PDF",
            responses={**NOT_FOUND, 409: {"model": ErrorResponse, "description": "Report is not complete"}})
def download_report(report_id: str, inline: Annotated[bool, Query(description="Show in the browser instead of downloading")] = False,
                    reports: ReportStore = Depends(get_report_store)):
    meta = reports.meta(report_id)
    if meta["status"] != "complete":
        raise ApiError(409, "report_not_ready", f"Report is {meta['status']}; it can be downloaded once complete.",
                       {"status": meta["status"], "stage": meta["stage"]})
    path = reports.pdf_path(report_id)
    if not path.is_file():
        raise ApiError(404, "report_not_found", "The report file is missing.")
    name = f"eagle-dar-report-{re.sub('[^0-9a-f]', '', meta['analysis_id'])[:8]}-{report_id[:8]}.pdf"   # server-built, never user input
    return FileResponse(path, media_type="application/pdf", filename=name,
                        content_disposition_type="inline" if inline else "attachment")


# ---------------------------------------------------------------- quick HTML/JSON view

@router.get("/analysis/{analysis_id}/report", response_model=None, responses={
    200: {"description": "HTML quick view (default) or JSON when format=json", "model": ReportResponse, "content": {"text/html": {}}},
    404: {"model": ErrorResponse, "description": "Unknown analysis id"}},
    summary="Quick report view (HTML or JSON)")
def report(analysis_id: str,
           format: Annotated[Literal["html", "json"], Query()] = "html",
           refresh: Annotated[bool, Query(description="Regenerate instead of using the cached HTML")] = False,
           store: AnalysisStore = Depends(get_store)):
    """Lightweight preview built from the same findings and recommendations as the PDF. The full report is the PDF."""
    d = store.path(analysis_id)
    cache, qfile = d / "report.html", d / "questions.json"
    if format == "html" and not refresh and cache.is_file() and (
            not qfile.is_file() or cache.stat().st_mtime >= qfile.stat().st_mtime):
        return Response(cache.read_text(encoding="utf-8"), media_type="text/html; charset=utf-8")
    document, html_text = report_service.build_report(store, analysis_id)
    cache.write_text(html_text, encoding="utf-8")
    if format == "json":
        return document
    return Response(html_text, media_type="text/html; charset=utf-8")
