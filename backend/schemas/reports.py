from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ReportSection(BaseModel):
    title: str
    paragraphs: list[str] = Field(default_factory=list)


class ReportResponse(BaseModel):
    analysis_id: str
    generated_utc: str
    generation: str = Field(description="How the text was produced.")
    sections: list[ReportSection]
    facts: dict[str, Any] = Field(description="The deterministic facts every number in the report comes from.")


class ReportMeta(BaseModel):
    """Status of one generated PDF report. Each generation creates a new report; earlier reports are never overwritten."""
    report_id: str
    analysis_id: str
    status: Literal["queued", "generating", "complete", "failed"]
    stage: str = Field(description="What the generator is doing now (real progress, not simulated).")
    progress: float = Field(ge=0, le=1)
    created_utc: str
    started_utc: str | None = None
    finished_utc: str | None = None
    generation_seconds: float | None = None
    bytes: int | None = None
    sha256: str | None = None
    pages: int | None = None
    explanation_source: str | None = Field(default=None, description="'deterministic-template' or an LLM whose numbers were verified.")
    fallback_reason: str | None = Field(default=None, description="Why the LLM was not used, when it was requested.")
    error: str | None = None
    download_url: str | None = Field(default=None, description="Present only when status is complete.")
