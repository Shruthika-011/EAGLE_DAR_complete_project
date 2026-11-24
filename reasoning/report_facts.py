"""Novelty 3 (data layer): structured evidence and statistics for the AI-generated
disaster assessment report. Standard library only.

Every number the report may state is produced here by deterministic code.
The LLM that writes the narrative (Phase 3) may only quote this block; it must
not count, compare or classify anything itself.

    python -m reasoning.report_facts --predictions outputs/predictions/X_predictions.json \
        --out outputs/phase2/X_report_facts.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

from reasoning import spatial
from reasoning.engine import answer_query, count_by_half, count_by_quadrant
from reasoning.loaders import scene_from_prediction_json
from reasoning.questions import Query
from reasoning.schema import CLASS_NAMES, Scene

ROOT = Path(__file__).resolve().parent.parent
LOW_CONFIDENCE = 0.35
PRIORITY_CLASSES = (2, 1)          # destroyed, then major damage
MAX_PRIORITY = 10

GUARDRAILS = (
    "Quote only numbers present in this document.",
    "Do not count, compare, rank or classify buildings yourself.",
    "Counts are model predictions, not ground truth; the segmentation model misses many buildings.",
    "Confidence values are uncalibrated softmax/detection scores, not probabilities of being correct.",
    "Damaged means major damage or destroyed; minor damage is merged into major damage.",
)


def _collapse_siamese_metrics(m: dict, source: Path) -> dict:
    """Collapse the model-level 5-class validation matrix to the app's three classes.

    True 'un-classified' rows are dropped (not an app class); minor-damage is merged into major-damage; a model
    prediction of 'un-classified' is counted as a miss, so the figures are conservative (the pipeline itself
    never outputs that class)."""
    cm = m["confusion_matrix"]
    labels, rows = cm["labels"], cm["rows_true_cols_pred"]
    app = {"no-damage": "no-damage", "minor-damage": "major-damage", "major-damage": "major-damage", "destroyed": "destroyed"}
    names = list(CLASS_NAMES)
    n = len(names)
    mat = [[0] * (n + 1) for _ in range(n)]            # last column: predicted un-classified
    for i, tl in enumerate(labels):
        if tl not in app:
            continue
        ti = names.index(app[tl])
        for j, pl in enumerate(labels):
            pj = names.index(app[pl]) if pl in app else n
            mat[ti][pj] += int(rows[i][j])
    support = {nm: sum(mat[i]) for i, nm in enumerate(names)}
    total = sum(support.values())
    f1s, recall = [], {}
    for i, nm in enumerate(names):
        tp = mat[i][i]
        col = sum(mat[r][i] for r in range(n))
        rec = tp / support[nm] if support[nm] else 0.0
        prec = tp / col if col else 0.0
        recall[nm] = rec
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return {"source": str(source), "n_val": total,
            "accuracy": sum(mat[i][i] for i in range(n)) / total if total else 0.0,
            "macro_f1": sum(f1s) / n, "recall_by_class": recall, "support_by_class": support,
            "note": "model-level validation collapsed to 3 classes (minor merged into major; un-classified rows dropped, "
                    "un-classified predictions counted as misses)"}


def _evaluation_context(eval_dir: Path) -> dict:
    """Measured model limitations, read from evaluation outputs if they exist.
    Nothing is hard-coded: a missing file yields no entry."""
    ctx = {}
    yolo = eval_dir / "yolo_eagle_buildings-2_best_metrics.json"
    if yolo.exists():
        m = json.loads(yolo.read_text(encoding="utf-8"))
        ctx["yolo_validation"] = {"source": str(yolo), "mask_precision": m["mask"]["precision"],
                                  "mask_recall": m["mask"]["recall"], "mask_mAP50": m["mask"]["mAP50"]}
    sia = eval_dir / "siamese_damage_metrics.json"
    if sia.exists():
        m = json.loads(sia.read_text(encoding="utf-8"))
        ctx["siamese_validation"] = _collapse_siamese_metrics(m, sia)
    for f in sorted(eval_dir.glob("pipeline_practice_*.json")):
        if f.name.endswith("_per_image.csv"):
            continue
        m = json.loads(f.read_text(encoding="utf-8"))
        if "detection" in m:
            ctx.setdefault("pipeline_practice", {})[m.get("alignment_mode", f.stem)] = {
                "source": str(f), "building_recall_iou50": m["detection"]["recall"],
                "building_precision_iou50": m["detection"]["precision"]}
    return ctx


def build_report_facts(scene: Scene, alignment: dict | None = None, warnings=(), eval_dir: Path | None = None) -> dict:
    total = len(scene.buildings)
    counts = {n: 0 for n in CLASS_NAMES}
    for b in scene.buildings:
        counts[b.damage_name] += 1
    damaged = sum(counts[n] for n in ("major-damage", "destroyed"))

    parts = spatial.assign_quadrants(scene.buildings, scene.width, scene.height)
    quads = {"damaged": count_by_quadrant(scene, "damaged")}
    halves = count_by_half(scene, "damaged")
    most = answer_query(scene, Query("B", level="damaged"))
    comparisons = {}
    for a, b in (("left", "right"), ("top", "bottom")):
        r = answer_query(scene, Query("C", halves=(a, b)))
        comparisons[f"{a}_vs_{b}"] = {"answer": r.answer, "region": list(r.region) if r.region else None,
                                      "damaged_counts": r.details.get("damaged_counts"),
                                      "supporting_building_ids": list(r.supporting_ids)}

    confs = [b.confidence for b in scene.buildings if b.confidence is not None]
    low = [b.building_id for b in scene.buildings if b.confidence is not None and b.confidence < LOW_CONFIDENCE]
    priority = sorted((b for b in scene.buildings if b.class_id in PRIORITY_CLASSES),
                      key=lambda b: (-b.class_id, -(b.confidence or 0.0), str(b.building_id)))[:MAX_PRIORITY]

    return {
        "schema": "eagle-dar/report-facts/v1",
        "scene": {"scene_id": scene.scene_id, "width": scene.width, "height": scene.height,
                  "total_buildings": total},
        "damage_counts": counts,
        "damaged_total": damaged,
        "damaged_fraction": round(damaged / total, 4) if total else None,
        "class_fractions": {n: round(c / total, 4) for n, c in counts.items()} if total else None,
        "quadrants": {q: {**{n: sum(1 for b in parts[q] if b.damage_name == n) for n in CLASS_NAMES},
                          "damaged": quads["damaged"][q],
                          "region": list(spatial.region_rect(q, scene.width, scene.height))}
                      for q in spatial.QUADRANTS},
        "halves": {h: {"damaged": halves[h], "region": list(spatial.region_rect(h, scene.width, scene.height))}
                   for h in spatial.HALVES},
        "most_damaged_quadrant": {"answer": most.answer, "region": list(most.region) if most.region else None,
                                  "supporting_building_ids": list(most.supporting_ids),
                                  "tie_unresolved": bool(most.details.get("tie_unresolved")),
                                  "quadrant_counts": most.details.get("quadrant_counts")},
        "half_comparisons": comparisons,
        "confidence": {
            "mean": round(statistics.fmean(confs), 4) if confs else None,
            "min": round(min(confs), 4) if confs else None,
            "max": round(max(confs), 4) if confs else None,
            "low_confidence_threshold": LOW_CONFIDENCE,
            "low_confidence_building_ids": low,
            "levels": {lv: sum(1 for b in scene.buildings if b.metadata.get("confidence_level") == lv)
                       for lv in ("high", "medium", "low")},
        },
        # Buildings the report should illustrate with before/after crops.
        "priority_buildings": [{**b.to_dict(), "polygon": None} for b in priority],
        "colour_legend": {"no-damage": "green", "major-damage": "orange", "destroyed": "red"},
        "data_quality": {"alignment": alignment, "warnings": list(warnings),
                         "evaluation_context": _evaluation_context(eval_dir or ROOT / "outputs" / "evaluation")},
        "llm_guardrails": list(GUARDRAILS),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    scene = scene_from_prediction_json(args.predictions)
    facts = build_report_facts(scene, scene.metadata.get("alignment"), scene.metadata.get("warnings") or ())
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(facts, indent=2), encoding="utf-8")
    print(f"Saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
