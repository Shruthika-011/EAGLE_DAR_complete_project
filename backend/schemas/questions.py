from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from backend.schemas.analysis import ErrorResponse  # noqa: F401  (re-exported for routes)

Level = Literal["no-damage", "major-damage", "destroyed", "damaged"]
Half = Literal["left", "right", "top", "bottom"]
Quadrant = Literal["top-left", "top-right", "bottom-left", "bottom-right"]
ID_PATTERN = r"^[A-Za-z0-9_-]{1,64}$"


class StructuredScope(BaseModel):
    kind: Literal["image", "quadrant", "half"]
    name: str | None = Field(default=None, description="Quadrant or half name; omit for the whole image.")

    @model_validator(mode="after")
    def _check(self):
        if self.kind == "image" and self.name is not None:
            raise ValueError("scope.name must be omitted when kind is 'image'")
        if self.kind == "quadrant" and self.name not in ("top-left", "top-right", "bottom-left", "bottom-right"):
            raise ValueError("scope.name must be top-left, top-right, bottom-left or bottom-right")
        if self.kind == "half" and self.name not in ("left", "right", "top", "bottom"):
            raise ValueError("scope.name must be left, right, top or bottom")
        return self


class StructuredQuestion(BaseModel):
    """The four official question types.

    type 1: count   -> level + scope        type 2: most damaged quadrant -> level ('damaged'|'destroyed')
    type 3: halves  -> halves [first, second] (opposite halves)          type 4: unanswerable (no other fields)
    """
    type: Literal[1, 2, 3, 4]
    level: Level | None = None
    scope: StructuredScope | None = None
    halves: tuple[Half, Half] | None = None

    @model_validator(mode="after")
    def _check(self):
        t = self.type
        if t == 1 and (self.level is None or self.scope is None or self.halves is not None):
            raise ValueError("type 1 requires level and scope (and no halves)")
        if t == 2 and (self.level not in ("damaged", "destroyed") or self.scope is not None or self.halves is not None):
            raise ValueError("type 2 requires level 'damaged' or 'destroyed' (and no scope/halves)")
        if t == 3:
            if self.halves is None or self.level is not None or self.scope is not None:
                raise ValueError("type 3 requires halves only")
            a, b = self.halves
            if {a, b} not in ({"left", "right"}, {"top", "bottom"}):
                raise ValueError("halves must be opposite: left/right or top/bottom")
        if t == 4 and (self.level or self.scope or self.halves):
            raise ValueError("type 4 takes no other fields")
        return self


class QuestionItem(BaseModel):
    id: str | None = Field(default=None, pattern=ID_PATTERN,
                           description="Official question id. Generated if omitted. Re-using an id replaces the earlier answer.")
    question: str | None = Field(default=None, max_length=500, description="Natural-language question (official templates).")
    structured: StructuredQuestion | None = None

    @model_validator(mode="after")
    def _one_of(self):
        if (self.question is None) == (self.structured is None):
            raise ValueError("provide exactly one of 'question' or 'structured'")
        return self


class QuestionsRequest(BaseModel):
    analysis_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    questions: list[QuestionItem] = Field(min_length=1, max_length=100)
    min_confidence: float = Field(default=0.0, ge=0.0, le=1.0,
                                  description="Ignore buildings whose building_confidence is below this. 0 keeps all.")
    region_format: Literal["bbox", "name"] = Field(
        default="bbox", description="'bbox' -> [x1,y1,x2,y2]; 'name' -> quadrant/half name for types 2 and 3.")


class OfficialRecord(BaseModel):
    """Exactly the official PS-04 submission record. Nothing else belongs here."""
    model_config = {"extra": "forbid"}
    id: str
    answer: int | str
    region: list[int] | str | None


class EvidenceBuilding(BaseModel):
    building_id: int
    damage: str
    confidence: float | None
    centre: list[float]
    bbox: list[int]


class Evidence(BaseModel):
    region: list[int] | None
    region_name: str | None
    supporting_building_ids: list[int]
    supporting_buildings: list[EvidenceBuilding]
    map_url: str | None = Field(description="Damage map with the evidence region and supporting buildings highlighted.")
    highlight_truncated: bool = False


class InternalTrace(BaseModel):
    """Internal metadata. Never part of the official submission."""
    question_type: Literal[1, 2, 3, 4]
    question_text: str | None
    parsed: dict[str, Any]
    details: dict[str, Any]
    evidence: Evidence


class QuestionResult(BaseModel):
    official: OfficialRecord
    internal: InternalTrace


class QuestionsResponse(BaseModel):
    analysis_id: str
    results: list[QuestionResult]
