"""Explanation layer for the report.

The deterministic explainer is the default and always works. An LLM may be used ONLY to rephrase structured facts,
and only when explicitly enabled (EAGLE_LLM_ENABLED=1 plus ANTHROPIC_API_KEY). Whatever it returns is rejected, and the
deterministic text used instead, if it contains any number that is not present in the facts it was given. It never
sees images, filenames, or user-entered labels such as district or state.
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Callable

import httpx

from backend.services.report_content import QUADS

log = logging.getLogger("eagle.explain")

SECTIONS = ("overall", "spatial", "reliability")


@dataclass
class Explanation:
    paragraphs: dict          # SECTIONS -> text
    source: str               # "deterministic-template" | "llm:<model> (numbers verified)"
    fallback_reason: str | None = None


# --------------------------------------------------------------------------- facts handed to the explainer

def explanation_facts(facts: dict, seconds: float) -> dict:
    """The ONLY data an explainer sees. Aggregates and ids; no images, names or user labels."""
    c = facts["damage_counts"]
    total = facts["scene"]["total_buildings"]
    conf = facts["confidence"]
    ev = (facts["data_quality"].get("evaluation_context") or {})
    out = {
        "total_buildings": total, "damage_counts": c, "affected_buildings": facts["damaged_total"],
        "affected_percent": None if facts["damaged_fraction"] is None else round(100 * facts["damaged_fraction"], 1),
        "class_percent": {k: round(100 * v, 1) for k, v in (facts["class_fractions"] or {}).items()},
        "affected_by_quadrant": {q: facts["quadrants"][q]["damaged"] for q in QUADS},
        "most_damaged_quadrant": facts["most_damaged_quadrant"]["answer"],
        "quadrant_tie": bool(facts["most_damaged_quadrant"].get("tie_unresolved")),
        "affected_by_half": {h: v["damaged"] for h, v in facts["halves"].items()},
        "mean_model_confidence": conf["mean"], "min_model_confidence": conf["min"], "max_model_confidence": conf["max"],
        "low_confidence_threshold": conf["low_confidence_threshold"],
        "low_confidence_count": len(conf["low_confidence_building_ids"]),
        "alignment_status": (facts["data_quality"].get("alignment") or {}).get("status"),
        "analysis_seconds": round(seconds, 1),
    }
    if ev.get("yolo_validation"):
        out["detector_validation_mask_recall"] = round(ev["yolo_validation"]["mask_recall"], 2)
    if ev.get("siamese_validation"):
        out["classifier_validation_macro_f1"] = round(ev["siamese_validation"]["macro_f1"], 2)
        out["classifier_validation_recall_by_class"] = {k: round(v, 2) for k, v in ev["siamese_validation"]["recall_by_class"].items()}
    return out


# --------------------------------------------------------------------------- deterministic explainer

def deterministic_explanation(f: dict) -> dict:
    total = f["total_buildings"]
    c = f["damage_counts"]
    if total == 0:
        overall = ("No buildings were detected in the after image, so there is nothing to assess. Absence of detections is not "
                   "evidence that the area has no buildings.")
        return {"overall": overall, "spatial": "No spatial breakdown is available because no buildings were detected.",
                "reliability": "The detector may miss buildings; consider re-checking the imagery."}
    overall = (f"Of {total} detected buildings, {f['affected_buildings']} ({f['affected_percent']}%) are classified as affected "
               f"(major damage or destroyed), including {c['destroyed']} classified as destroyed. {c['no-damage']} are classified "
               f"as no damage. These are model "
               f"predictions from the supplied imagery, produced in {f['analysis_seconds']} seconds.")
    q, per = f["most_damaged_quadrant"], f["affected_by_quadrant"]
    if f["affected_buildings"] == 0:
        spatial = "No affected buildings were detected in any quadrant."
    elif q in QUADS:
        spatial = (f"Affected buildings by quadrant: " + ", ".join(f"{k} {per[k]}" for k in QUADS) +
                   f". The {q} quadrant has the most. Quadrants split the image at its centre and each building is assigned by its centre.")
    else:
        spatial = ("Affected buildings by quadrant: " + ", ".join(f"{k} {per[k]}" for k in QUADS) +
                   ". No single quadrant has the most (tie).")
    rel = (f"Mean model confidence is {f['mean_model_confidence']:.3f}, and {f['low_confidence_count']} building(s) fall below the "
           f"{f['low_confidence_threshold']} low-confidence threshold. Model confidence is an uncalibrated score of how strongly "
           f"the model supports a prediction; it is not the probability that the prediction is correct and not model accuracy.")
    if "detector_validation_mask_recall" in f:
        rel += (f" On held-out validation data the building detector's mask recall was {f['detector_validation_mask_recall']}, so "
                "buildings it misses are absent from every count.")
    return {"overall": overall, "spatial": spatial, "reliability": rel}


# --------------------------------------------------------------------------- number guard

def _fmt_variants(v: float) -> set[str]:
    out = {str(v)}
    for d in range(0, 4):
        out.add(f"{v:.{d}f}")
    if float(v).is_integer():
        out.add(str(int(v)))
    return out


def allowed_numbers(payload) -> set[str]:
    allowed: set[str] = set()

    def walk(x):
        if isinstance(x, bool):
            return
        if isinstance(x, (int, float)):
            allowed.update(_fmt_variants(float(x)))
        elif isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, (list, tuple)):
            for v in x:
                walk(v)
    walk(payload)
    return allowed


NUM_RE = re.compile(r"(?<![\w.])\d+(?:\.\d+)?")


def unverified_numbers(text: str, payload) -> list[str]:
    allowed = allowed_numbers(payload)
    return [n for n in NUM_RE.findall(text) if n not in allowed and n.rstrip("0").rstrip(".") not in allowed]


# --------------------------------------------------------------------------- LLM provider (optional)

SYSTEM_PROMPT = (
    "You explain a disaster damage assessment to a non-technical reader. You are given a JSON object of computed facts. "
    "Write three short plain-English paragraphs: overall, spatial, reliability. Rules: use ONLY numbers that appear in the JSON; "
    "never compute new counts, percentages or totals; never invent building IDs, locations, place names, dates or causes; "
    "do not contradict the JSON; describe predictions as model estimates, not confirmed damage; say that model confidence is "
    "not accuracy; damaged means major damage or destroyed. "
    'Reply with JSON only: {"overall": "...", "spatial": "...", "reliability": "..."}.'
)


def anthropic_call(payload: dict, model: str, timeout: float) -> str:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")
    r = httpx.post("https://api.anthropic.com/v1/messages", timeout=timeout,
                   headers={"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
                   json={"model": model, "max_tokens": 900, "system": SYSTEM_PROMPT,
                         "messages": [{"role": "user", "content": json.dumps(payload)}]})
    r.raise_for_status()
    return "".join(b.get("text", "") for b in r.json().get("content", []))


def explain(facts: dict, seconds: float, *, enabled: bool = False, model: str = "claude-sonnet-5-5", timeout: float = 30.0,
            call: Callable[[dict, str, float], str] | None = None) -> Explanation:
    payload = explanation_facts(facts, seconds)
    base = deterministic_explanation(payload)
    if not enabled:
        return Explanation(base, "deterministic-template")
    try:
        raw = (call or anthropic_call)(payload, model, timeout)
        data = json.loads(raw[raw.index("{"): raw.rindex("}") + 1])
        paragraphs = {k: str(data[k]).strip() for k in SECTIONS}
        if any(not v for v in paragraphs.values()):
            raise ValueError("empty section in LLM output")
        bad = unverified_numbers(" ".join(paragraphs.values()), payload)
        if bad:
            raise ValueError(f"LLM text contains numbers not present in the facts: {sorted(set(bad))[:6]}")
        return Explanation(paragraphs, f"llm:{model} (numbers verified against computed facts)")
    except Exception as exc:                                       # any failure -> deterministic template
        reason = f"{type(exc).__name__}: {str(exc)[:160]}"
        log.warning("LLM explanation unavailable, using template: %s", reason)
        return Explanation(base, "deterministic-template", fallback_reason=reason)
