"""Bridge between stored analyses and the deterministic reasoning engine (Phase 2).

No counting, comparison or quadrant logic lives here: it is all in `reasoning/`.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

from backend.errors import ApiError
from backend.schemas.questions import QuestionItem, StructuredQuestion
from backend.utils.storage import AnalysisStore
from reasoning import spatial
from reasoning.engine import Answer, answer_query
from reasoning.loaders import scene_from_prediction_json
from reasoning.official import to_official, validate_official
from reasoning.questions import Query, parse_question
from reasoning.report_facts import build_report_facts
from reasoning.schema import Scene

LETTER_TO_TYPE = {"A": 1, "B": 2, "C": 3, "D": 4}


def load_scene_from_dir(directory, analysis_id: str) -> Scene:
    """Single code path that turns stored predictions into a Scene (also used while an analysis is being created)."""
    try:
        scene = scene_from_prediction_json(Path(directory) / "predictions.json")
    except (OSError, ValueError, KeyError) as exc:
        raise ApiError(500, "corrupt_analysis", "Stored analysis data could not be read.") from exc
    return dataclasses.replace(scene, scene_id=analysis_id)


def load_scene(store: AnalysisStore, analysis_id: str) -> Scene:
    return load_scene_from_dir(store.path(analysis_id), analysis_id)


def query_from_structured(sq: StructuredQuestion) -> Query:
    if sq.type == 1:
        return Query("A", level=sq.level, scope_kind=sq.scope.kind, scope_name=sq.scope.name)
    if sq.type == 2:
        return Query("B", level=sq.level)
    if sq.type == 3:
        return Query("C", halves=tuple(sq.halves))
    return Query("D", reason="question declared unanswerable by the caller")


def describe_structured(sq: StructuredQuestion) -> str:
    return "structured: " + sq.model_dump_json(exclude_none=True)


def answer_item(scene: Scene, item: QuestionItem, question_id: str, min_confidence: float, region_format: str):
    """-> (query, Answer, official_record, question_text)"""
    if item.structured is not None:
        query, text = query_from_structured(item.structured), describe_structured(item.structured)
    else:
        query, text = parse_question(item.question), item.question
    ans: Answer = answer_query(scene, query, min_confidence)
    record = to_official(question_id, ans, region_format)
    validate_official(record, scene.width, scene.height)       # compliance is enforced, not assumed
    return query, ans, record, text


def building_out(scene: Scene, b, include_polygon: bool) -> dict:
    m = b.metadata
    out = {
        "building_id": b.building_id, "predicted_damage": b.damage_name, "class_id": b.class_id,
        "building_confidence": b.confidence,
        "detection_confidence": m.get("detection_confidence"), "damage_confidence": m.get("damage_confidence"),
        "damage_margin": m.get("damage_margin"), "confidence_level": m.get("confidence_level"),
        "class_probabilities": m.get("class_probabilities"),
        "centre_xy": [round(b.centre[0], 2), round(b.centre[1], 2)], "bbox_xyxy": list(b.bbox),
        "quadrant": spatial.quadrant_of(b.centre[0], b.centre[1], scene.width, scene.height),
        "area_px": m.get("area_px"), "geometry_source": m.get("geometry_source"),
        "local_alignment": m.get("local_alignment"),
    }
    if include_polygon:
        out["polygon_xy"] = [list(p) for p in b.polygon]
    return out


STAT_KEYS = ("scene", "damage_counts", "damaged_total", "damaged_fraction", "class_fractions", "quadrants",
             "halves", "most_damaged_quadrant", "half_comparisons", "confidence")


def statistics(scene: Scene, alignment: dict | None, warnings) -> dict:
    facts = build_report_facts(scene, alignment, warnings)
    return {k: facts[k] for k in STAT_KEYS}


def analysis_links(analysis_id: str) -> dict:
    base = f"/api/analysis/{analysis_id}"
    return {"self": base, "before_image": f"{base}/image/before", "after_image": f"{base}/image/after",
            "map": f"{base}/map", "comparison": f"{base}/comparison", "report": f"{base}/report",
            "questions": "/api/questions", "export": f"{base}/export"}


def build_analysis_view(store: AnalysisStore, analysis_id: str, include_polygons: bool = False) -> dict:
    meta = store.meta(analysis_id)
    preds = store.predictions(analysis_id)
    scene = load_scene(store, analysis_id)
    return {
        "analysis_id": analysis_id, "created_utc": meta["created_utc"], "status": meta["status"],
        "image": meta["image"], "context": meta.get("context") or {}, "parameters": meta["parameters"],
        "alignment": preds["alignment"],
        "statistics": statistics(scene, preds["alignment"], preds["warnings"]),
        "buildings": [building_out(scene, b, include_polygons) for b in scene.buildings],
        "warnings": preds["warnings"], "links": analysis_links(analysis_id),
        "disclaimer": preds["disclaimer"],
    }
