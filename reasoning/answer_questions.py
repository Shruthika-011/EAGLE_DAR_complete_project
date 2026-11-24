"""Answer a PS-04 questions.jsonl deterministically.

Outputs
  --out    official submission JSONL: exactly {"id","answer","region"} per line
  --trace  internal JSONL: parsed query, supporting building ids, counts, flags

Scene source
  predictions   (default) Phase 1 JSON files <predictions-dir>/<pair_id>_predictions.json
  ground-truth  official labels, to validate the reasoning logic in isolation
                from model error. NOT a model-performance number.

Examples
  python -m reasoning.answer_questions --questions dataset_source/practice/questions.jsonl \
      --source ground-truth --labels-dir dataset_source/practice/labels \
      --answers dataset_source/practice/answers.jsonl --out outputs/phase2/practice_gt_submission.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from reasoning.engine import answer_query
from reasoning.loaders import scene_from_ground_truth, scene_from_prediction_json
from reasoning.official import to_official, trace_record, validate_official, write_jsonl
from reasoning.questions import parse_question
from reasoning.schema import CANNOT_BE_DETERMINED

ROOT = Path(__file__).resolve().parent.parent


def _read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _region_iou(a, b):
    if a is None or b is None:
        return None
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--source", choices=["predictions", "ground-truth"], default="predictions")
    ap.add_argument("--predictions-dir", default=str(ROOT / "outputs" / "predictions"))
    ap.add_argument("--labels-dir", default=None, help="label JSON directory (ground-truth source)")
    ap.add_argument("--out", required=True, help="official submission JSONL")
    ap.add_argument("--trace", default=None, help="internal trace JSONL (default: <out>.trace.jsonl)")
    ap.add_argument("--region-format", choices=["bbox", "name"], default="bbox")
    ap.add_argument("--min-confidence", type=float, default=0.0,
                    help="ignore buildings whose building_confidence is below this (default 0 = keep all)")
    ap.add_argument("--answers", default=None, help="official answers.jsonl, to score (practice set only)")
    ap.add_argument("--on-missing", choices=["error", "cannot-determine"], default="error",
                    help="when a pair has no scene data")
    args = ap.parse_args(argv)

    questions = _read_jsonl(args.questions)
    scenes, official, traces, errors = {}, [], [], []

    for q in questions:
        pid = q["pair_id"]
        if pid not in scenes:
            try:
                if args.source == "ground-truth":
                    if not args.labels_dir:
                        raise SystemExit("--labels-dir is required with --source ground-truth")
                    scenes[pid] = scene_from_ground_truth(Path(args.labels_dir) / f"{pid}.json")
                else:
                    scenes[pid] = scene_from_prediction_json(Path(args.predictions_dir) / f"{pid}_predictions.json")
            except (FileNotFoundError, ValueError, KeyError) as exc:
                scenes[pid] = None
                errors.append(f"{pid}: {exc}")
        scene = scenes[pid]
        query = parse_question(q["question"])
        if scene is None:
            if args.on_missing == "error":
                continue
            rec = {"id": q["id"], "answer": CANNOT_BE_DETERMINED, "region": None}
            official.append(rec)
            traces.append({"id": q["id"], "question": q["question"], "note": "no scene data; defaulted"})
            continue
        ans = answer_query(scene, query, args.min_confidence)
        rec = to_official(q["id"], ans, args.region_format)
        validate_official(rec, scene.width, scene.height)
        official.append(rec)
        traces.append(trace_record(q["id"], q["question"], query, ans, scene, {b.building_id: b for b in scene.buildings}))

    if errors and args.on_missing == "error":
        print("ERROR: missing/invalid scene data for:\n  " + "\n  ".join(errors), file=sys.stderr)
        return 2

    out = Path(args.out)
    write_jsonl(out, official)
    write_jsonl(args.trace or out.with_suffix(".trace.jsonl"), traces)
    print(f"Wrote {len(official)} official records -> {out}")

    if args.answers:
        key = {a["id"]: a for a in _read_jsonl(args.answers)}
        stats = Counter(); region_ious = []
        for rec in official:
            k = key.get(rec["id"])
            if k is None:
                continue
            t = k["type"]; stats[t, "n"] += 1
            if rec["answer"] == k["answer"]:
                stats[t, "answer_ok"] += 1
            if t in (1, 2, 3) and k["region"] is not None and isinstance(rec["region"], list):
                iou = _region_iou(rec["region"], k["region"])
                region_ious.append(iou)
                if rec["region"] == k["region"]:
                    stats[t, "region_exact"] += 1
        print(f"\nScored against {args.answers} (source={args.source}):")
        for t in (1, 2, 3, 4):
            n = stats[t, "n"]
            if n:
                print(f"  type {t}: n={n} answer {stats[t, 'answer_ok']}/{n}"
                      + (f"  region exact {stats[t, 'region_exact']}/{n}" if t != 4 else ""))
        tot = sum(stats[t, "n"] for t in (1, 2, 3, 4)); ok = sum(stats[t, "answer_ok"] for t in (1, 2, 3, 4))
        print(f"  overall answer match: {ok}/{tot} = {ok / tot:.1%}" if tot else "  no matching ids")
        if region_ious:
            print(f"  mean region IoU (types 1-3): {sum(region_ious) / len(region_ious):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
