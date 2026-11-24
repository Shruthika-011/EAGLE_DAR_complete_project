"""Deterministic answer + evidence engine. Standard library only.

Conventions taken from the 50 official practice answers (verified, see
docs/PHASE2_SPATIAL_RULES.md):
  * Type 1 region  = union bbox of the counted buildings (floor/ceil, clipped);
                     if the count is 0 it is the queried quadrant/half/image rectangle.
  * Type 2 region  = rectangle of the winning quadrant.
  * Type 3 region  = rectangle of the half with MORE damaged buildings, which is
                     not necessarily the first half named in the question.
  * Type 4         = "cannot be determined", region null.

NOT covered by any official example (so handled conservatively and flagged in
the internal trace, never silently resolved):
  * Type 2 tie (two or more quadrants share the maximum, including 0 damaged
    everywhere): answer "cannot be determined", region null. No quadrant is
    picked by an invented tie-break rule.
  * Type 3 equal counts: answer "no" (strictly-more rule). The region has no
    official precedent; we return the union of the two compared halves.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from reasoning import spatial
from reasoning.questions import LEVELS, Query
from reasoning.schema import CANNOT_BE_DETERMINED, Building, Scene

RULES_VERSION = "1.0.0"


@dataclass(frozen=True)
class Answer:
    qtype: str                        # "A" | "B" | "C" | "D"
    answer: Any                       # int | quadrant str | "yes"/"no" | "cannot be determined"
    region: tuple | None              # [x1, y1, x2, y2] in image pixels, or None
    region_name: str | None = None    # "top-left" / "left half" when the region is a named part
    supporting_ids: tuple = ()        # building ids that support the answer
    details: dict = field(default_factory=dict)   # internal only, never in the official JSON


def _in_level(b: Building, level: str) -> bool:
    return b.class_id in LEVELS[level]


def _cannot(reason: str, **details) -> Answer:
    return Answer("D", CANNOT_BE_DETERMINED, None, None, (), {"reason": reason, **details})


def _ids(buildings) -> tuple:
    return tuple(b.building_id for b in buildings)


def count_by_quadrant(scene: Scene, level: str) -> dict:
    parts = spatial.assign_quadrants(scene.buildings, scene.width, scene.height)
    return {q: sum(1 for b in bs if _in_level(b, level)) for q, bs in parts.items()}


def count_by_half(scene: Scene, level: str) -> dict:
    return {h: sum(1 for b in spatial.buildings_in_region(scene.buildings, "half", h, scene.width, scene.height)
                   if _in_level(b, level)) for h in spatial.HALVES}


def answer_count(scene: Scene, q: Query) -> Answer:
    w, h = scene.width, scene.height
    in_scope = spatial.buildings_in_region(scene.buildings, q.scope_kind, q.scope_name, w, h)
    chosen = [b for b in in_scope if _in_level(b, q.level)]
    box = spatial.union_bbox((b.evidence_region for b in chosen), w, h)
    basis = "union_of_counted_buildings"
    if box is None:
        box = spatial.region_rect(q.scope_name or "image", w, h)
        basis = "scope_rectangle_because_count_is_zero"
    return Answer("A", len(chosen), box, None, _ids(chosen), {
        "level": q.level, "scope": [q.scope_kind, q.scope_name],
        "buildings_in_scope": len(in_scope), "region_basis": basis,
    })


def answer_most_damaged_quadrant(scene: Scene, q: Query) -> Answer:
    w, h = scene.width, scene.height
    if not scene.buildings:
        return _cannot("image has no buildings; quadrant questions are not answerable")
    counts = count_by_quadrant(scene, q.level)
    top = max(counts.values())
    winners = [qd for qd in spatial.QUADRANTS if counts[qd] == top]
    details = {"level": q.level, "quadrant_counts": counts, "max_count": top}
    if len(winners) != 1:
        return Answer("D", CANNOT_BE_DETERMINED, None, None, (), {
            **details, "reason": "tie for most buildings; no official tie-break rule exists",
            "tied_quadrants": winners, "tie_unresolved": True})
    win = winners[0]
    support = [b for b in spatial.buildings_in_region(scene.buildings, "quadrant", win, w, h) if _in_level(b, q.level)]
    return Answer("B", win, spatial.region_rect(win, w, h), win, _ids(support), {**details, "winner": win})


def answer_compare_halves(scene: Scene, q: Query) -> Answer:
    w, h = scene.width, scene.height
    if not scene.buildings:
        return _cannot("image has no buildings; half questions are not answerable")
    a, b = q.halves
    sel = {hn: [x for x in spatial.buildings_in_region(scene.buildings, "half", hn, w, h) if _in_level(x, "damaged")]
           for hn in (a, b)}
    ca, cb = len(sel[a]), len(sel[b])
    details = {"first_half": a, "second_half": b, "damaged_counts": {a: ca, b: cb}}
    if ca > cb:
        return Answer("C", "yes", spatial.region_rect(a, w, h), spatial.region_label(a), _ids(sel[a]),
                      {**details, "region_basis": "half_with_more_damaged"})
    if cb > ca:
        return Answer("C", "no", spatial.region_rect(b, w, h), spatial.region_label(b), _ids(sel[b]),
                      {**details, "region_basis": "half_with_more_damaged"})
    box = spatial.union_bbox([spatial.region_rect(a, w, h), spatial.region_rect(b, w, h)], w, h)
    return Answer("C", "no", box, None, _ids(sel[a] + sel[b]),
                  {**details, "tie": True, "region_basis": "union_of_both_halves_no_official_precedent"})


def answer_query(scene: Scene, query: Query, min_confidence: float = 0.0) -> Answer:
    """Answer one parsed query on one scene. Pure function of its inputs."""
    scene = scene.with_min_confidence(min_confidence)
    if query.qtype == "A":
        ans = answer_count(scene, query)
    elif query.qtype == "B":
        ans = answer_most_damaged_quadrant(scene, query)
    elif query.qtype == "C":
        ans = answer_compare_halves(scene, query)
    else:
        ans = _cannot(query.reason or "unsupported question")
    return Answer(ans.qtype, ans.answer, ans.region, ans.region_name, ans.supporting_ids,
                  {**ans.details, "min_confidence": min_confidence, "rules_version": RULES_VERSION,
                   "image_size": [scene.width, scene.height], "total_buildings": len(scene.buildings)})
