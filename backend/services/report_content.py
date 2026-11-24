"""Deterministic report content: findings, recommendations, executive-summary numbers.

Every sentence is computed from `reasoning.report_facts.build_report_facts(...)` (itself computed from the stored
predictions). Each finding carries a `source` naming the fact it came from, so it is traceable. Nothing here calls a
model; the optional LLM layer (explain.py) only receives these facts and may only rephrase them.
"""
from __future__ import annotations

from reasoning.schema import CLASS_NAMES

QUADS = ("top-left", "top-right", "bottom-left", "bottom-right")
LABEL = {"no-damage": "no damage", "major-damage": "major damage", "destroyed": "destroyed"}

CLASS_DEFINITIONS = {   # paraphrase of the xBD joint damage scale used by the PS-04 labels
    "no-damage": "No visible damage to the building in the after image.",
    "major-damage": "Visible structural damage, from partial roof damage or debris/water around an otherwise intact structure to partial wall or roof collapse (minor and major damage are reported together).",
    "destroyed": "Building collapsed, burnt or no longer present as a standing structure.",
}


def pct(x, digits: int = 1) -> str:
    return "n/a" if x is None else f"{100 * x:.{digits}f}%"


def affected_counts(facts: dict) -> dict:
    c = facts["damage_counts"]
    total = facts["scene"]["total_buildings"]
    return {"total": total, "affected": facts["damaged_total"], "unaffected": c["no-damage"],
            "severe": c["destroyed"], "damaged_fraction": facts["damaged_fraction"]}


def build_findings(facts: dict) -> list[dict]:
    total = facts["scene"]["total_buildings"]
    c = facts["damage_counts"]
    out: list[dict] = []
    if total == 0:
        return [{"text": "No buildings were detected in the after image, so no damage statistics can be reported. "
                         "This does not show that the area has no buildings.", "source": "scene.total_buildings"}]
    out.append({"text": f"{facts['damaged_total']} of {total} detected buildings ({pct(facts['damaged_fraction'])}) are classified as "
                        f"affected (major or destroyed): {c['major-damage']} major, "
                        f"{c['destroyed']} destroyed.", "source": "damage_counts, damaged_total"})
    out.append({"text": f"{c['destroyed']} building(s) are classified as destroyed "
                        f"({pct(c['destroyed'] / total)} of detected buildings).", "source": "damage_counts.destroyed"})
    top = max(CLASS_NAMES, key=lambda n: c[n])
    out.append({"text": f"The largest class is {LABEL[top]}: {c[top]} buildings ({pct(c[top] / total)} of detections).",
                "source": "class_fractions"})
    mq = facts["most_damaged_quadrant"]
    qc = mq.get("quadrant_counts") or {}
    if facts["damaged_total"] == 0:
        out.append({"text": "No affected buildings were detected in any quadrant.", "source": "quadrants.*.damaged"})
    elif mq["answer"] in QUADS:
        out.append({"text": f"The {mq['answer']} quadrant contains the most affected buildings ({qc.get(mq['answer'])} of "
                            f"{facts['damaged_total']}).", "source": "most_damaged_quadrant"})
    elif mq.get("tie_unresolved") and qc:
        top_n = max(qc.values())
        out.append({"text": f"No single quadrant contains the most affected buildings: "
                            f"{', '.join(q for q in QUADS if qc.get(q) == top_n)} are tied at {top_n}.",
                    "source": "most_damaged_quadrant.tie_unresolved"})
    for key, (a, b) in (("left_vs_right", ("left", "right")), ("top_vs_bottom", ("top", "bottom"))):
        dc = facts["half_comparisons"][key].get("damaged_counts")
        if dc and facts["damaged_total"] > 0:
            verdict = (f"the {a} half has more" if dc[a] > dc[b] else f"the {b} half has more" if dc[b] > dc[a] else "the halves are equal")
            out.append({"text": f"Affected buildings by half: {a} {dc[a]}, {b} {dc[b]}; {verdict}.",
                        "source": f"half_comparisons.{key}"})
    conf = facts["confidence"]
    low = len(conf["low_confidence_building_ids"])
    if conf["mean"] is not None:
        out.append({"text": f"{low} building(s) have model confidence below {conf['low_confidence_threshold']} "
                            f"({pct(low / total, 0)} of detections); mean model confidence is {conf['mean']:.3f}.",
                    "source": "confidence"})
    al = (facts["data_quality"].get("alignment") or {})
    if al.get("status") in ("misaligned", "unverified"):
        out.append({"text": f"Before/after alignment is {al['status']}; classification of small buildings may be less reliable.",
                    "source": "data_quality.alignment"})
    return out


def _ids(ids, limit=15) -> str:
    ids = list(ids)
    return ", ".join(str(i) for i in ids[:limit]) + (f" (+{len(ids) - limit} more)" if len(ids) > limit else "")


def build_recommendations(facts: dict) -> list[dict]:
    """AI-assisted guidance. Each item names the numbers that triggered it. All require human verification."""
    total = facts["scene"]["total_buildings"]
    c = facts["damage_counts"]
    recs: list[dict] = []
    if total == 0:
        return [{"text": "Re-check the imagery (coverage, cloud, resolution) or run field/aerial verification: the model found "
                         "no buildings, which does not establish that there are none.", "trigger": "total_buildings = 0"}]
    pri = [p["building_id"] for p in facts["priority_buildings"] if p["damage"] == "destroyed"]
    if c["destroyed"]:
        recs.append({"text": f"Prioritise field verification of the {c['destroyed']} building(s) classified as destroyed"
                             f"{' (highest-confidence IDs: ' + _ids(pri, 10) + ')' if pri else ''}.",
                     "trigger": f"destroyed = {c['destroyed']}"})
    if c["major-damage"]:
        recs.append({"text": f"Schedule inspection of the {c['major-damage']} building(s) classified as major damage.",
                     "trigger": f"major-damage = {c['major-damage']}"})
    mq = facts["most_damaged_quadrant"]
    qc = mq.get("quadrant_counts") or {}
    if mq["answer"] in QUADS and facts["damaged_total"] >= 3 and qc.get(mq["answer"], 0) / facts["damaged_total"] >= 0.4:
        recs.append({"text": f"Review the {mq['answer']} quadrant first: it holds {qc[mq['answer']]} of the {facts['damaged_total']} "
                             "affected buildings.", "trigger": "most_damaged_quadrant share >= 40%"})
    low = facts["confidence"]["low_confidence_building_ids"]
    if low:
        recs.append({"text": f"Manually inspect the {len(low)} low-confidence building(s) before relying on their class "
                             "(listed in the confidence section).", "trigger": f"low-confidence = {len(low)}"})
    ev = facts["data_quality"].get("evaluation_context") or {}
    yv = ev.get("yolo_validation")
    if yv and yv["mask_recall"] < 0.5:
        recs.append({"text": "Conduct additional assessment (ground survey or higher-resolution imagery) where image evidence is "
                             "insufficient: the building detector's measured recall is low, so some buildings are not in these counts.",
                     "trigger": f"detector mask recall = {yv['mask_recall']:.2f}"})
    al = (facts["data_quality"].get("alignment") or {})
    if al.get("status") in ("misaligned", "unverified"):
        recs.append({"text": "Confirm that the before and after images are co-registered before acting on building-level labels.",
                     "trigger": f"alignment = {al['status']}"})
    recs.append({"text": "Treat all results as decision support: verify with ground reports or professional assessment before "
                         "dispatching resources.", "trigger": "always"})
    return recs
