"""Official PS-04 submission records. Standard library only.

An official record has EXACTLY three keys: id, answer, region. Everything else
(supporting building ids, counts, flags) lives in the separate internal trace.
"""
from __future__ import annotations

import json
from pathlib import Path

from reasoning import spatial
from reasoning.engine import Answer
from reasoning.schema import CANNOT_BE_DETERMINED

OFFICIAL_KEYS = ("id", "answer", "region")


def to_official(question_id: str, ans: Answer, region_format: str = "bbox") -> dict:
    """region_format: 'bbox' -> [x1,y1,x2,y2] (matches answers.jsonl `region`);
    'name' -> quadrant/half name for type 2/3 (README allows either)."""
    if region_format not in ("bbox", "name"):
        raise ValueError("region_format must be 'bbox' or 'name'")
    region = list(ans.region) if ans.region is not None else None
    if region_format == "name" and ans.region_name is not None:
        region = ans.region_name
    return {"id": question_id, "answer": ans.answer, "region": region}


def validate_official(rec: dict, width: int | None = None, height: int | None = None) -> None:
    """Raise ValueError if a record is not an exact official submission record."""
    if tuple(rec.keys()) != OFFICIAL_KEYS:
        raise ValueError(f"official record keys must be exactly {OFFICIAL_KEYS}, got {tuple(rec.keys())}")
    if not isinstance(rec["id"], str) or not rec["id"]:
        raise ValueError("id must be a non-empty string")
    a, r = rec["answer"], rec["region"]
    if isinstance(a, bool):
        raise ValueError("answer must not be a boolean (use 'yes'/'no')")
    if isinstance(a, int):
        if a < 0:
            raise ValueError("count answers must be >= 0")
    elif isinstance(a, str):
        if a not in ({CANNOT_BE_DETERMINED, "yes", "no"} | set(spatial.QUADRANTS)):
            raise ValueError(f"unexpected string answer {a!r}")
    else:
        raise ValueError(f"answer must be int or str, got {type(a).__name__}")
    if a == CANNOT_BE_DETERMINED and r is not None:
        raise ValueError("'cannot be determined' must have region null")
    if r is None:
        return
    if isinstance(r, str):
        if r not in set(spatial.QUADRANTS) | {f"{h} half" for h in spatial.HALVES}:
            raise ValueError(f"unknown region name {r!r}")
        return
    if not (isinstance(r, list) and len(r) == 4 and all(isinstance(v, int) and not isinstance(v, bool) for v in r)):
        raise ValueError("region must be null, a region name, or [x1, y1, x2, y2] integers")
    x1, y1, x2, y2 = r
    if not (0 <= x1 <= x2 and 0 <= y1 <= y2):
        raise ValueError(f"region {r} is not a valid box")
    if width is not None and height is not None and (x2 > width or y2 > height):
        raise ValueError(f"region {r} extends outside the {width}x{height} image")


def trace_record(question_id: str, question: str, query, ans: Answer, scene, buildings_by_id: dict) -> dict:
    """Internal traceability record: answer -> supporting buildings. NOT for submission."""
    support = []
    for bid in ans.supporting_ids:
        b = buildings_by_id[bid]
        support.append({"building_id": bid, "damage": b.damage_name, "confidence": b.confidence,
                        "centre": list(b.centre), "evidence_region": list(b.evidence_region)})
    return {
        "id": question_id, "question": question, "question_type": ans.qtype,
        "parsed": {"level": query.level, "scope": [query.scope_kind, query.scope_name],
                   "halves": list(query.halves) if query.halves else None, "reason": query.reason},
        "answer": ans.answer, "region": list(ans.region) if ans.region else None,
        "region_name": ans.region_name,
        "supporting_building_ids": list(ans.supporting_ids),
        "supporting_buildings": support, "details": ans.details,
        "scene_id": scene.scene_id,
    }


def write_jsonl(path, records) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="\n") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
