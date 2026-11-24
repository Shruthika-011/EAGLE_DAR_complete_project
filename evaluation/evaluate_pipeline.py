"""End-to-end evaluation of the inference pipeline against ground-truth polygons.

Only scenes unseen by BOTH models are used:
  * practice  - the 40 PS-04 practice pairs (never used for training), and
  * joint_val - train-folder scenes that are in the YOLO val split AND the
                Siamese val split (the two preprocessing scripts split
                differently, so most YOLO-val scenes were Siamese-train scenes).

Predicted building polygons are matched to GT polygons greedily by IoU (>=0.5).
Reports detection P/R/F1, 5-class damage metrics on matched buildings, and
per-image count errors (the quantity PS-04 counting questions depend on).
This is NOT the official PS-04 answer-accuracy evaluation.

Usage:
    python -m evaluation.evaluate_pipeline --set practice
    python -m evaluation.evaluate_pipeline --set joint_val
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from shapely import wkt
from shapely.geometry import Polygon
from shapely.strtree import STRtree
from shapely.validation import make_valid
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

from config import (ROOT, OUTPUTS_DIR, SIAMESE_WEIGHTS, YOLO_WEIGHTS, YOLO_CONF, YOLO_IMGSZ,
                    APP_DAMAGE_CLASSES, SUBTYPE_TO_APP_NAME, DAMAGED_CLASSES)
from inference.pipeline import DamagePipeline, PipelineConfig, CLASS_NAMES, _largest_polygon

IOU_THR = 0.5


def load_gt(label_path):
    data = json.loads(Path(label_path).read_text(encoding="utf-8"))
    out = []
    for f in data.get("features", {}).get("xy", []):
        props = f.get("properties", {})
        if props.get("feature_type") != "building":
            continue
        dmg = SUBTYPE_TO_APP_NAME.get(str(props.get("subtype", "un-classified")).lower())
        if dmg is None:      # un-classified is not an app class; minor-damage maps to major-damage
            continue
        try:
            g = wkt.loads(f["wkt"])
        except Exception:
            continue
        if not g.is_valid:
            g = make_valid(g)
        g = _largest_polygon(g)
        if g is not None and g.area > 0:
            out.append((g, APP_DAMAGE_CLASSES[dmg]))
    return out


def scenes_for(set_name):
    if set_name == "practice":
        img_dir = ROOT / "dataset_source" / "practice" / "images"
        lbl_dir = ROOT / "dataset_source" / "practice" / "labels"
        ids = sorted(p.stem for p in lbl_dir.glob("*.json"))
    elif set_name == "joint_val":
        img_dir = ROOT / "dataset" / "raw" / "train" / "images"
        lbl_dir = ROOT / "dataset" / "raw" / "train" / "labels"
        yolo_val = {p.stem.rsplit("_", 1)[0] for p in (ROOT / "processed" / "yolo" / "images" / "val").iterdir()}
        with (ROOT / "processed" / "siamese" / "pairs_val.csv").open(encoding="utf-8") as f:
            siam_val = {r["scene"] for r in csv.DictReader(f)}
        with (ROOT / "processed" / "siamese" / "pairs_train.csv").open(encoding="utf-8") as f:
            siam_train = {r["scene"] for r in csv.DictReader(f)}
        ids = sorted((yolo_val & siam_val) - siam_train)
    else:
        raise SystemExit(f"unknown set {set_name}")
    return [(i, img_dir / f"{i}_pre.jpg", img_dir / f"{i}_post.jpg", lbl_dir / f"{i}.json") for i in ids]


def match(preds, gts):
    """Greedy one-to-one matching by IoU. Returns list of (pred_idx, gt_idx, iou)."""
    if not preds or not gts:
        return []
    tree = STRtree([g for g, _ in gts])
    cands = []
    for pi, pp in enumerate(preds):
        for gi in tree.query(pp):
            gp = gts[int(gi)][0]
            inter = pp.intersection(gp).area
            if inter <= 0:
                continue
            iou = inter / (pp.area + gp.area - inter)
            if iou >= IOU_THR:
                cands.append((iou, pi, int(gi)))
    cands.sort(reverse=True)
    used_p, used_g, out = set(), set(), []
    for iou, pi, gi in cands:
        if pi in used_p or gi in used_g:
            continue
        used_p.add(pi); used_g.add(gi); out.append((pi, gi, iou))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["practice", "joint_val"], default="practice")
    ap.add_argument("--conf", type=float, default=YOLO_CONF)
    ap.add_argument("--imgsz", type=int, default=YOLO_IMGSZ)
    ap.add_argument("--siamese", default=str(SIAMESE_WEIGHTS))
    ap.add_argument("--modes", default="check,register", help="alignment modes to compare")
    ap.add_argument("--out-dir", default=str(OUTPUTS_DIR / "evaluation"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    scenes = scenes_for(args.set)
    print(f"{args.set}: {len(scenes)} scenes")
    pipe = DamagePipeline(PipelineConfig(yolo_weights=YOLO_WEIGHTS, siamese_weights=Path(args.siamese),
                                         conf=args.conf, imgsz=args.imgsz, local_alignment=False))
    all_results = {}
    for mode in args.modes.split(","):
        pipe.cfg.alignment = mode
        tp = fp = fn = 0
        y_true, y_pred, per_image, ious = [], [], [], []
        align_status = {}
        for sid, pre, post, lbl in scenes:
            gts = load_gt(lbl)
            report, _ = pipe.run(pre, post)
            preds = [Polygon(b["polygon_xy"]) for b in report["buildings"]]
            pred_cls = [b["class_id"] for b in report["buildings"]]
            m = match(preds, gts)
            tp += len(m); fp += len(preds) - len(m); fn += len(gts) - len(m)
            for pi, gi, iou in m:
                y_true.append(gts[gi][1]); y_pred.append(pred_cls[pi]); ious.append(iou)
            st = report["alignment"]["status"]
            align_status[st] = align_status.get(st, 0) + 1
            row = {"scene": sid, "gt_buildings": len(gts), "pred_buildings": len(preds), "matched": len(m),
                   "alignment": st, "shift_px": report["alignment"]["shift_px"],
                   "registered": report["alignment"]["registered"]}
            for ci, name in enumerate(CLASS_NAMES):
                row[f"gt_{name}"] = sum(1 for _, c in gts if c == ci)
                row[f"pred_{name}"] = report["summary"]["damage_counts"][name]
            row["gt_damaged"] = sum(row[f"gt_{c}"] for c in DAMAGED_CLASSES)
            row["pred_damaged"] = report["summary"]["damaged_total"]
            per_image.append(row)

        labels = list(range(len(CLASS_NAMES)))
        prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
        res = {
            "set": args.set, "alignment_mode": mode, "n_scenes": len(scenes),
            "yolo_conf": args.conf, "yolo_imgsz": args.imgsz, "iou_threshold": IOU_THR,
            "alignment_status_counts": align_status,
            "detection": {"tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec,
                          "f1": 2 * prec * rec / max(prec + rec, 1e-12),
                          "mean_matched_iou": float(np.mean(ious)) if ious else None},
            "damage_on_matched": None,
            "counts": {},
        }
        if y_true:
            res["damage_on_matched"] = {
                "n": len(y_true),
                "accuracy": accuracy_score(y_true, y_pred),
                "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
                "weighted_f1": f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0),
                "report": classification_report(y_true, y_pred, labels=labels, target_names=CLASS_NAMES,
                                                output_dict=True, zero_division=0),
                "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
                "gt_present_classes": sorted({CLASS_NAMES[c] for c in y_true}),
            }
        for key in ["buildings"] + CLASS_NAMES + ["damaged"]:
            g = np.array([r["gt_buildings" if key == "buildings" else f"gt_{key}"] for r in per_image])
            p = np.array([r["pred_buildings" if key == "buildings" else f"pred_{key}"] for r in per_image])
            res["counts"][key] = {"gt_total": int(g.sum()), "pred_total": int(p.sum()),
                                  "mae_per_image": float(np.abs(g - p).mean()),
                                  "exact_match_rate": float((g == p).mean())}
        all_results[mode] = res

        tag = f"pipeline_{args.set}_{mode}"
        (out_dir / f"{tag}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
        with (out_dir / f"{tag}_per_image.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(per_image[0].keys())); w.writeheader(); w.writerows(per_image)

        d = res["detection"]
        print(f"\n=== {args.set} | alignment={mode} ===")
        print(f"alignment status: {align_status}")
        print(f"Detection @IoU{IOU_THR}: P={d['precision']:.3f} R={d['recall']:.3f} F1={d['f1']:.3f} "
              f"(TP={tp} FP={fp} FN={fn})")
        if res["damage_on_matched"]:
            dm = res["damage_on_matched"]
            print(f"Damage on {dm['n']} matched: acc={dm['accuracy']:.3f} macroF1={dm['macro_f1']:.3f} "
                  f"weightedF1={dm['weighted_f1']:.3f}")
            print("confusion (rows=true):", dm["confusion_matrix"])
        for k, c in res["counts"].items():
            print(f"  count {k:14s} gt={c['gt_total']:5d} pred={c['pred_total']:5d} "
                  f"MAE/img={c['mae_per_image']:.2f} exact={c['exact_match_rate']:.2f}")


if __name__ == "__main__":
    main()
