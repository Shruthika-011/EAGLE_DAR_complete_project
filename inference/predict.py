"""EAGLE-DAR building damage inference (CLI).

Example:
    python -m inference.predict \
        --before dataset_source/practice/images/p11b4ad02_pre.jpg \
        --after  dataset_source/practice/images/p11b4ad02_post.jpg

The original implementation is preserved in inference/predict_legacy.py.
"""
import argparse
import sys
from pathlib import Path

from config import OUTPUTS_DIR, SIAMESE_WEIGHTS, YOLO_CONF, YOLO_IMGSZ, YOLO_IOU, YOLO_MAX_DET, YOLO_WEIGHTS
from inference.pipeline import DamagePipeline, InferenceError, PipelineConfig, save_outputs


def default_name(after_path: str) -> str:
    stem = Path(after_path).stem
    for suffix in ("_post_disaster", "_post", "-post", "_after"):
        if stem.lower().endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="EAGLE-DAR building damage inference")
    p.add_argument("--before", required=True, help="Pre-disaster image")
    p.add_argument("--after", required=True, help="Post-disaster image")
    p.add_argument("--yolo", default=str(YOLO_WEIGHTS), help="YOLOv8-seg weights")
    p.add_argument("--siamese", default=str(SIAMESE_WEIGHTS), help="Siamese damage weights")
    p.add_argument("--conf", type=float, default=YOLO_CONF, help="YOLO confidence threshold")
    p.add_argument("--imgsz", type=int, default=YOLO_IMGSZ, help="YOLO inference size (multiple of 32)")
    p.add_argument("--iou", type=float, default=YOLO_IOU, help="YOLO NMS IoU threshold")
    p.add_argument("--max-det", type=int, default=YOLO_MAX_DET)
    p.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:N")
    p.add_argument("--alignment", choices=["check", "register"], default="check",
                   help="check: measure and report pre/post offset; "
                        "register: also translate the pre image onto the post image when offset")
    p.add_argument("--out-dir", default=str(OUTPUTS_DIR / "predictions"))
    p.add_argument("--name", default=None, help="Output file prefix (default: scene id)")
    p.add_argument("--save-crops", action="store_true", help="Save per-building before|after crops")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        pipe = DamagePipeline(PipelineConfig(
            yolo_weights=Path(args.yolo), siamese_weights=Path(args.siamese),
            conf=args.conf, imgsz=args.imgsz, iou=args.iou, max_det=args.max_det,
            device=args.device, alignment=args.alignment))
        report, images = pipe.run(args.before, args.after)
        paths = save_outputs(report, images, Path(args.out_dir),
                             args.name or default_name(args.after), save_crops=args.save_crops)
    except InferenceError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    s = report["summary"]
    al = report["alignment"]
    print(f"\nDevice: {report['parameters']['device']}")
    print(f"Alignment: {al['status']} (shift {al['shift_px']:.1f}px via {al['method']}"
          f"{', registered' if al['registered'] else ''})")
    print(f"Buildings detected: {report['total_detected_buildings']}")
    for name, n in s["damage_counts"].items():
        print(f"  {name:14s} {n}")
    print(f"  damaged total  {s['damaged_total']}")
    print(f"Confidence levels: {s['confidence_levels']}")
    for w in report["warnings"]:
        print(f"WARNING: {w}")
    for k, v in paths.items():
        print(f"Saved {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
