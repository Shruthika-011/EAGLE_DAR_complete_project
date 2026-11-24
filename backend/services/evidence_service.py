"""Evidence payloads and evidence-highlighted maps.

Evidence regions and supporting building ids are produced by the reasoning
engine (`reasoning.engine.Answer`); this module only shapes them for the API and
renders the map. The three-class map itself is `reasoning.damage_map`.
"""
from __future__ import annotations

import cv2
import numpy as np

from backend.errors import ApiError
from backend.utils.storage import AnalysisStore
from reasoning.damage_map import render_damage_map
from reasoning.engine import Answer
from reasoning.schema import Scene

MAX_HIGHLIGHT_IN_URL = 150


def evidence_payload(analysis_id: str, scene: Scene, ans: Answer) -> dict:
    by_id = {b.building_id: b for b in scene.buildings}
    support = [by_id[i] for i in ans.supporting_ids]
    map_url, truncated = None, False
    if ans.region is not None:
        q = f"evidence={','.join(str(v) for v in ans.region)}"
        ids = [b.building_id for b in support]
        if ids and len(ids) <= MAX_HIGHLIGHT_IN_URL:
            q += "&highlight=" + ",".join(str(i) for i in ids)
        elif ids:
            truncated = True
        map_url = f"/api/analysis/{analysis_id}/map?{q}"
    return {
        "region": list(ans.region) if ans.region is not None else None,
        "region_name": ans.region_name,
        "supporting_building_ids": [b.building_id for b in support],
        "supporting_buildings": [
            {"building_id": b.building_id, "damage": b.damage_name, "confidence": b.confidence,
             "centre": [round(b.centre[0], 2), round(b.centre[1], 2)], "bbox": list(b.evidence_region)}
            for b in support],
        "map_url": map_url, "highlight_truncated": truncated,
    }


def parse_evidence_param(text: str | None, width: int, height: int):
    if not text:
        return None
    try:
        x1, y1, x2, y2 = (int(v) for v in text.split(","))
    except ValueError:
        raise ApiError(422, "invalid_evidence", "evidence must be four comma-separated integers: x1,y1,x2,y2") from None
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ApiError(422, "invalid_evidence", f"evidence region must satisfy 0<=x1<x2<={width} and 0<=y1<y2<={height}")
    return (x1, y1, x2, y2)


def parse_highlight_param(text: str | None, scene: Scene):
    if not text:
        return ()
    try:
        ids = [int(v) for v in text.split(",") if v.strip()]
    except ValueError:
        raise ApiError(422, "invalid_highlight", "highlight must be comma-separated building ids") from None
    if len(ids) > 2000:
        raise ApiError(422, "invalid_highlight", "too many ids in highlight")
    known = {b.building_id for b in scene.buildings}
    unknown = [i for i in ids if i not in known]
    if unknown:
        raise ApiError(422, "invalid_highlight", f"unknown building ids: {unknown[:10]}")
    return tuple(ids)


def load_after_image(store: AnalysisStore, analysis_id: str) -> np.ndarray:
    data = np.fromfile(str(store.image_path(analysis_id, "after")), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ApiError(500, "corrupt_analysis", "Stored after-image could not be decoded.")
    return img


def render_map_png(image: np.ndarray, scene: Scene, evidence=None, highlight=(), legend: bool = True) -> bytes:
    out = render_damage_map(image, scene, evidence_region=evidence, highlight_ids=highlight, legend=legend)
    ok, buf = cv2.imencode(".png", out)
    if not ok:
        raise ApiError(500, "render_failed", "Could not encode the damage map.")
    return buf.tobytes()
