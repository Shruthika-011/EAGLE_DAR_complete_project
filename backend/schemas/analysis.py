from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

DamageName = Literal["no-damage", "major-damage", "destroyed"]


class ErrorBody(BaseModel):
    code: str = Field(description="Stable machine-readable error code.")
    message: str
    details: dict[str, Any] | None = None


class ErrorResponse(BaseModel):
    error: ErrorBody


class UploadInfo(BaseModel):
    display_name: str = Field(description="Sanitised original filename (informational only).")
    bytes: int
    sha256: str


class ImageInfo(BaseModel):
    width: int
    height: int
    before: UploadInfo
    after: UploadInfo


class AnalysisParameters(BaseModel):
    conf: float = Field(description="YOLO confidence threshold.")
    imgsz: int = Field(description="YOLO inference image size.")
    alignment: Literal["check", "register"]
    device: str
    inference_seconds: float


class BuildingOut(BaseModel):
    building_id: int
    predicted_damage: DamageName
    class_id: int = Field(ge=0, le=4)
    building_confidence: float = Field(description="detection_confidence x damage_confidence (Novelty 1).")
    detection_confidence: float | None = None
    damage_confidence: float | None = None
    damage_margin: float | None = None
    confidence_level: Literal["high", "medium", "low"] | None = None
    class_probabilities: dict[str, float] | None = None
    centre_xy: list[float] = Field(description="Polygon centroid in image pixels.")
    bbox_xyxy: list[int]
    quadrant: Literal["top-left", "top-right", "bottom-left", "bottom-right"]
    area_px: float | None = None
    geometry_source: str | None = None
    local_alignment: dict[str, Any] | None = None
    polygon_xy: list[list[float]] | None = Field(default=None, description="Only with include_polygons=true.")


class AnalysisContext(BaseModel):
    disaster_name: str | None = None
    district: str | None = None
    state: str | None = None


class AnalysisLinks(BaseModel):
    self: str
    before_image: str
    after_image: str
    map: str = Field(description="Five-class colour-coded damage map (PNG).")
    comparison: str = Field(description="Before | after | damage map strip (JPEG).")
    report: str
    questions: str = Field(description="POST here to ask questions about this analysis.")
    export: str = Field(description="Official PS-04 answers for the questions asked so far.")


class AnalysisResponse(BaseModel):
    analysis_id: str
    created_utc: str
    status: Literal["complete"]
    image: ImageInfo
    context: AnalysisContext = Field(default_factory=AnalysisContext,
                                     description="User-supplied labels. Not derived from the imagery; there is no georeferencing.")
    parameters: AnalysisParameters
    alignment: dict[str, Any] = Field(description="Global pre/post alignment measurement.")
    statistics: dict[str, Any] = Field(description="Damage counts, per-quadrant/half statistics and confidence summary, "
                                                   "all computed by the deterministic reasoning engine.")
    buildings: list[BuildingOut]
    warnings: list[str]
    links: AnalysisLinks
    disclaimer: str
