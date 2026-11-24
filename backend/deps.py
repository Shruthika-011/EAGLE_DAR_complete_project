from __future__ import annotations

from fastapi import Request

from backend.config import Settings
from backend.services.inference_service import InferenceService
from backend.services.report_jobs import ReportService
from backend.utils.report_store import ReportStore
from backend.utils.storage import AnalysisStore


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_store(request: Request) -> AnalysisStore:
    return request.app.state.store


def get_inference(request: Request) -> InferenceService:
    return request.app.state.inference


def get_report_store(request: Request) -> ReportStore:
    return request.app.state.report_store


def get_report_service(request: Request) -> ReportService:
    return request.app.state.report_service
