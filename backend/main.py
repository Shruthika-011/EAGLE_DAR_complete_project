"""EAGLE-DAR API.

    python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000      (from the project root)
    Swagger UI: http://127.0.0.1:8000/docs
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from backend import API_VERSION
from backend.config import Settings, get_settings
from backend.errors import install_error_handlers
from backend.routes import analysis, health, questions, reports
from backend.services.inference_service import InferenceService
from backend.services.report_jobs import ReportService
from backend.utils.report_store import ReportStore
from backend.utils.storage import AnalysisStore

DESCRIPTION = """
**EAGLE-DAR** — Evidence-Aware Geospatial Learning and Explanation for Disaster Assessment and Reasoning.

1. `POST /api/analyze` — upload a before/after pair; YOLOv8-seg finds buildings, a Siamese ResNet18 grades each
   one into three damage classes with a building-level confidence.
2. `POST /api/questions` — ask official-style questions; answers and evidence regions are computed by deterministic
   spatial reasoning, never by a language model.
3. `GET /api/analysis/{id}/map | report | export` — three-class damage map, assessment report, official answer JSON.

Predictions are model estimates. Building recall of the segmentation model is limited; see the report's limitations.
"""

TAGS = [{"name": "health", "description": "Status, model availability, CUDA"},
        {"name": "analysis", "description": "Upload imagery, retrieve results and visualisations"},
        {"name": "questions", "description": "Official question types and PS-04 answer export"},
        {"name": "reports", "description": "Disaster assessment report"}]


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Models are loaded exactly once, here; request handlers only use the loaded instances.
        app.state.inference.load()
        interrupted = app.state.report_service.recover()
        if interrupted:
            logging.getLogger("eagle.reports").warning("Marked %d interrupted report(s) as failed", interrupted)
        yield
        app.state.report_service.shutdown()

    app = FastAPI(title="EAGLE-DAR API", version=API_VERSION, description=DESCRIPTION, openapi_tags=TAGS,
                  lifespan=lifespan, docs_url="/docs", redoc_url="/redoc")
    app.state.settings = settings
    app.state.store = AnalysisStore(settings.storage_dir)
    app.state.inference = InferenceService(settings)
    app.state.report_store = ReportStore(settings.reports_dir)
    app.state.report_service = ReportService(settings, app.state.store, app.state.report_store)

    app.add_middleware(CORSMiddleware, allow_origins=list(settings.cors_origins), allow_methods=["GET", "POST"],
                       allow_headers=["*"], expose_headers=["X-Analysis-Id", "X-Record-Count"])
    install_error_handlers(app)
    for r in (health.router, analysis.router, questions.router, reports.router):
        app.include_router(r, prefix="/api")

    @app.get("/", include_in_schema=False)
    def root():
        return RedirectResponse("/docs")

    return app


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
app = create_app()
