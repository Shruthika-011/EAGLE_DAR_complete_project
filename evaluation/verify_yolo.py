"""Verify the existing YOLO building-segmentation weights. Does not train.

Usage (needs the interpreter that has ultralytics installed):
    python -m evaluation.verify_yolo --weights runs/eagle_buildings-2/weights/best.pt
"""
import argparse
import json
from pathlib import Path

import torch
from ultralytics import YOLO

from config import ROOT, OUTPUTS_DIR


def ckpt_metadata(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    train_args = ckpt.get("train_args") or {}
    return {
        "epoch_index_saved": ckpt.get("epoch"),  # 0-based; -1 means a stripped/final ckpt
        "best_fitness": None if ckpt.get("best_fitness") is None else float(ckpt["best_fitness"]),
        "date": ckpt.get("date"),
        "version": ckpt.get("version"),
        "has_optimizer": ckpt.get("optimizer") is not None,
        "train_args": {k: train_args.get(k) for k in
                       ("model", "data", "epochs", "patience", "batch", "imgsz", "name")},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="runs/eagle_buildings-2/weights/best.pt")
    parser.add_argument("--data", default="processed/yolo/data.yaml")
    parser.add_argument("--out-dir", default=str(OUTPUTS_DIR / "evaluation"))
    args = parser.parse_args()

    weights = (ROOT / args.weights).resolve()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = ckpt_metadata(weights)
    model = YOLO(str(weights))
    task = model.task
    names = model.names

    res = model.val(data=str(ROOT / args.data), split="val", imgsz=640, batch=2,
                    device=0 if torch.cuda.is_available() else "cpu",
                    project=str(out_dir), name="yolo_val", exist_ok=True,
                    plots=True, verbose=False)

    out = {
        "weights": str(weights),
        "task": task,
        "names": names,
        "checkpoint": meta,
        "box": {"precision": float(res.box.mp), "recall": float(res.box.mr),
                "mAP50": float(res.box.map50), "mAP50_95": float(res.box.map)},
        "mask": {"precision": float(res.seg.mp), "recall": float(res.seg.mr),
                 "mAP50": float(res.seg.map50), "mAP50_95": float(res.seg.map)},
    }
    stem = Path(args.weights).parent.parent.name + "_" + weights.stem
    (out_dir / f"yolo_{stem}_metrics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
