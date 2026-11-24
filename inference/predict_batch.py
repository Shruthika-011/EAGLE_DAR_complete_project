"""Run the Phase 1 pipeline over every pair named in a questions.jsonl (loads the models once).

    python -m inference.predict_batch --questions dataset_source/practice/questions.jsonl \
        --images-dir dataset_source/practice --out-dir outputs/predictions

Writes <pair_id>_predictions.json (+ damage map and comparison image) per pair,
which `python -m reasoning.answer_questions` then consumes.
"""
import argparse
import json
import sys
from pathlib import Path

from config import OUTPUTS_DIR, SIAMESE_WEIGHTS, YOLO_CONF, YOLO_IMGSZ, YOLO_IOU, YOLO_MAX_DET, YOLO_WEIGHTS
from inference.pipeline import DamagePipeline, InferenceError, PipelineConfig, save_outputs


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", required=True)
    ap.add_argument("--images-dir", required=True, help="directory containing the questions' pre_image/post_image paths")
    ap.add_argument("--out-dir", default=str(OUTPUTS_DIR / "predictions"))
    ap.add_argument("--yolo", default=str(YOLO_WEIGHTS))
    ap.add_argument("--siamese", default=str(SIAMESE_WEIGHTS))
    ap.add_argument("--conf", type=float, default=YOLO_CONF)
    ap.add_argument("--imgsz", type=int, default=YOLO_IMGSZ)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--alignment", choices=["check", "register"], default="check")
    args = ap.parse_args(argv)

    pairs = {}
    for line in Path(args.questions).read_text(encoding="utf-8").splitlines():
        if line.strip():
            q = json.loads(line)
            pairs.setdefault(q["pair_id"], (q["pre_image"], q["post_image"]))
    try:
        pipe = DamagePipeline(PipelineConfig(
            yolo_weights=Path(args.yolo), siamese_weights=Path(args.siamese), conf=args.conf, imgsz=args.imgsz,
            iou=YOLO_IOU, max_det=YOLO_MAX_DET, device=args.device, alignment=args.alignment))
    except InferenceError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    failed = 0
    for i, (pid, (pre, post)) in enumerate(sorted(pairs.items()), 1):
        try:
            report, images = pipe.run(Path(args.images_dir) / pre, Path(args.images_dir) / post)
            save_outputs(report, images, Path(args.out_dir), pid)
            print(f"[{i}/{len(pairs)}] {pid}: {report['total_detected_buildings']} buildings, "
                  f"alignment={report['alignment']['status']}", flush=True)
        except InferenceError as exc:
            failed += 1
            print(f"[{i}/{len(pairs)}] {pid}: FAILED {exc}", file=sys.stderr)
    print(f"Done. {len(pairs) - failed}/{len(pairs)} pairs written to {args.out_dir}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
