"""EAGLE-DAR inference pipeline: building segmentation -> per-building damage.

Stages
  1. Validate and load the pre/post image pair.
  2. Measure pre/post alignment (optionally register pre onto post).
  3. Segment buildings on the POST image with YOLOv8-seg (as trained).
  4. For each building polygon: centre, bbox, area, quadrant/half.
  5. Cut pre/post crops exactly as in training (polygon bounds + padding,
     resized to CROP_SIZE with INTER_AREA) and classify with the Siamese model.
  6. Emit building-level confidence, a 5-class colour-coded damage map and
     structured JSON.

The trained models are used unchanged.
"""
from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from shapely.geometry import Polygon, box as shapely_box
from shapely.validation import make_valid
from torchvision import transforms

from config import (
    CROP_PADDING,
    CROP_SIZE,
    APP_DAMAGE_NAMES,
    DAMAGE_COLORS_BGR,
    DAMAGED_CLASSES,
    IMAGE_EXTENSIONS,
    MODEL_TO_APP_CLASS,
)
from inference import alignment as align
from training.siamese_model import SiameseDamageClassifier

PIPELINE_VERSION = "1.1.0"
MODEL_NUM_CLASSES = 5          # outputs of the trained Siamese head (unchanged)
NUM_CLASSES = len(APP_DAMAGE_NAMES)   # app classes: no-damage, major-damage, destroyed
CLASS_NAMES = [APP_DAMAGE_NAMES[i] for i in range(NUM_CLASSES)]

# Identical to the transform in training/train_siamese.py
SIAMESE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

CONFIDENCE_DEFINITION = {
    "detection_confidence": "YOLO segmentation score for the building instance",
    "damage_confidence": "softmax probability of the predicted damage class (uncalibrated)",
    "damage_margin": "top-1 minus top-2 class probability",
    "building_confidence": "detection_confidence * damage_confidence",
    "confidence_level": "high if building_confidence >= 0.60, medium if >= 0.35, else low "
                        "(heuristic thresholds, not calibrated probabilities)",
}


class InferenceError(Exception):
    """A user-facing error with a clear message."""


# --------------------------------------------------------------------------
# Image loading / validation
# --------------------------------------------------------------------------

def read_image(path: str | Path, role: str) -> np.ndarray:
    p = Path(path)
    if not p.exists():
        raise InferenceError(f"{role} image not found: {p}")
    if not p.is_file():
        raise InferenceError(f"{role} image path is not a file: {p}")
    if p.suffix.lower() not in IMAGE_EXTENSIONS:
        raise InferenceError(f"{role} image has unsupported extension '{p.suffix}'. "
                             f"Supported: {', '.join(IMAGE_EXTENSIONS)}")
    # np.fromfile + imdecode handles non-ASCII Windows paths, unlike cv2.imread.
    data = np.fromfile(str(p), dtype=np.uint8)
    if data.size == 0:
        raise InferenceError(f"{role} image is empty: {p}")
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise InferenceError(f"{role} image could not be decoded (corrupt or not an image): {p}")
    if img.shape[0] < 32 or img.shape[1] < 32:
        raise InferenceError(f"{role} image is too small ({img.shape[1]}x{img.shape[0]}).")
    return img


def validate_pair(pre: np.ndarray, post: np.ndarray) -> list[str]:
    warnings = []
    if pre.shape[:2] != post.shape[:2]:
        raise InferenceError(
            f"Before/after dimensions differ: {pre.shape[1]}x{pre.shape[0]} vs "
            f"{post.shape[1]}x{post.shape[0]}. Register/resize them deliberately first.")
    h, w = post.shape[:2]
    if (w, h) != (1024, 1024):
        warnings.append(f"image size {w}x{h} differs from the 1024x1024 training data")
    for role, img in (("before", pre), ("after", post)):
        if float(img.std()) < 2.0:
            warnings.append(f"{role} image is nearly uniform (std<2); it may be blank")
    if np.array_equal(pre, post):
        warnings.append("before and after images are identical")
    return warnings


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------

def select_device(name: str = "auto") -> torch.device:
    name = (name or "auto").lower()
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name.startswith("cuda"):
        if not torch.cuda.is_available():
            raise InferenceError("CUDA was requested but is not available. Use --device cpu.")
        return torch.device(name)
    if name == "cpu":
        return torch.device("cpu")
    raise InferenceError(f"Unknown device '{name}'. Use auto, cpu or cuda[:N].")


def _sha256_short(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def load_yolo(path: str | Path):
    p = Path(path)
    if not p.exists():
        raise InferenceError(f"YOLO weights not found: {p}")
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise InferenceError("ultralytics is not installed in this Python environment "
                             "(pip install ultralytics).") from exc
    model = YOLO(str(p))
    if model.task != "segment":
        raise InferenceError(f"YOLO weights are a '{model.task}' model; a segmentation model is required.")
    return model


def load_siamese(path: str | Path, device: torch.device) -> SiameseDamageClassifier:
    p = Path(path)
    if not p.exists():
        raise InferenceError(f"Siamese weights not found: {p}")
    try:
        state = torch.load(p, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise InferenceError(f"Could not read Siamese checkpoint {p}: {exc}") from exc
    # Accept a bare state_dict (current format) or a wrapped training checkpoint.
    if isinstance(state, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
    state = {k.removeprefix("module."): v for k, v in state.items()}
    model = SiameseDamageClassifier(num_classes=MODEL_NUM_CLASSES, pretrained=False)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise InferenceError(f"Siamese checkpoint does not match the model architecture: {exc}") from exc
    return model.to(device).eval()


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------

def _largest_polygon(geom):
    if geom.is_empty:
        return None
    if geom.geom_type == "Polygon":
        return geom
    polys = [g for g in getattr(geom, "geoms", []) if g.geom_type == "Polygon" and not g.is_empty]
    polys += [p for g in getattr(geom, "geoms", []) if g.geom_type == "MultiPolygon" for p in g.geoms]
    return max(polys, key=lambda g: g.area) if polys else None


def region_of(cx: float, cy: float, w: int, h: int) -> tuple[str, str, str]:
    """PS-04 rule: split at image centre; the part containing the centre wins.

    Points exactly on a centre line go to the right/bottom part, matching the
    half-open [0,512) / [512,1024) ranges in question_templates.json.
    """
    right, bottom = cx >= w / 2, cy >= h / 2
    quadrant = f"{'bottom' if bottom else 'top'}-{'right' if right else 'left'}"
    return quadrant, ("right" if right else "left"), ("bottom" if bottom else "top")


@dataclass
class Detection:
    polygon: Polygon
    detection_confidence: float
    geometry_source: str            # "mask_polygon" | "bbox_fallback"
    extras: dict = field(default_factory=dict)


def extract_buildings(result, img_w: int, img_h: int, min_area_px: float) -> tuple[list[Detection], int]:
    """Convert a YOLO result into validated building polygons in image pixels."""
    if result.boxes is None or len(result.boxes) == 0:
        return [], 0
    confs = result.boxes.conf.cpu().numpy()
    boxes = result.boxes.xyxy.cpu().numpy()
    polys_xy = result.masks.xy if result.masks is not None else [None] * len(confs)
    frame = shapely_box(0, 0, img_w, img_h)

    out, dropped = [], 0
    for i, conf in enumerate(confs):
        pts = polys_xy[i] if i < len(polys_xy) else None
        source = "mask_polygon"
        if pts is not None and len(pts) >= 3:
            geom = Polygon(pts)
        else:
            x1, y1, x2, y2 = boxes[i]
            geom = shapely_box(x1, y1, x2, y2)
            source = "bbox_fallback"
        if not geom.is_valid:
            geom = make_valid(geom)
        geom = _largest_polygon(geom.intersection(frame))
        if geom is None or geom.area < min_area_px:
            dropped += 1
            continue
        out.append(Detection(geom, float(conf), source))
    return out, dropped


def crop_like_training(image: np.ndarray, polygon: Polygon) -> np.ndarray | None:
    """Exact replica of preprocessing/build_siamese_pairs.py::crop_polygon."""
    h, w = image.shape[:2]
    minx, miny, maxx, maxy = polygon.bounds
    x1 = max(0, int(minx) - CROP_PADDING)
    y1 = max(0, int(miny) - CROP_PADDING)
    x2 = min(w, int(maxx) + CROP_PADDING + 1)
    y2 = min(h, int(maxy) + CROP_PADDING + 1)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_AREA)


def _to_tensor(bgr_crop: np.ndarray) -> torch.Tensor:
    # Training crops were saved by cv2 (BGR) and read with PIL as RGB.
    return SIAMESE_TRANSFORM(Image.fromarray(cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)))


def collapse_probs(model_probs: np.ndarray) -> np.ndarray:
    """(N, 5) model softmax -> (N, 3) app-class probabilities.

    The 'un-classified' output (index 4) is ignored and the remaining four are
    renormalised; 'minor-damage' probability is added to 'major-damage'."""
    model_probs = np.asarray(model_probs, dtype=np.float64)
    out = np.zeros((model_probs.shape[0], NUM_CLASSES), dtype=np.float64)
    for src, dst in MODEL_TO_APP_CLASS.items():
        out[:, dst] += model_probs[:, src]
    total = out.sum(axis=1, keepdims=True)
    return out / np.where(total > 0, total, 1.0)


@torch.no_grad()
def classify_pairs(model, pre_crops, post_crops, device, batch_size=64) -> np.ndarray:
    probs = []
    for s in range(0, len(pre_crops), batch_size):
        b = torch.stack([_to_tensor(c) for c in pre_crops[s:s + batch_size]]).to(device)
        a = torch.stack([_to_tensor(c) for c in post_crops[s:s + batch_size]]).to(device)
        with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            logits = model(b, a)
        probs.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
    raw = np.concatenate(probs) if probs else np.zeros((0, MODEL_NUM_CLASSES), dtype=np.float32)
    return collapse_probs(raw)


def confidence_level(score: float) -> str:
    return "high" if score >= 0.60 else "medium" if score >= 0.35 else "low"


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    yolo_weights: Path
    siamese_weights: Path
    conf: float = 0.25
    imgsz: int = 640
    iou: float = 0.7
    max_det: int = 1000
    device: str = "auto"
    alignment: str = "check"        # check | register
    min_area_px: float = 4.0
    local_alignment: bool = True


class DamagePipeline:
    def __init__(self, cfg: PipelineConfig):
        if not 0.0 < cfg.conf < 1.0:
            raise InferenceError("--conf must be between 0 and 1.")
        if cfg.imgsz < 32 or cfg.imgsz % 32:
            raise InferenceError("--imgsz must be a positive multiple of 32.")
        if cfg.alignment not in ("check", "register"):
            raise InferenceError("--alignment must be 'check' or 'register'.")
        self.cfg = cfg
        self.device = select_device(cfg.device)
        self.yolo = load_yolo(cfg.yolo_weights)
        self.siamese = load_siamese(cfg.siamese_weights, self.device)
        self.model_info = {
            "yolo": {"path": str(Path(cfg.yolo_weights).resolve()),
                     "sha256_16": _sha256_short(Path(cfg.yolo_weights)),
                     "classes": {int(k): v for k, v in self.yolo.names.items()}},
            "siamese": {"path": str(Path(cfg.siamese_weights).resolve()),
                        "sha256_16": _sha256_short(Path(cfg.siamese_weights)),
                        "architecture": "SiameseDamageClassifier(ResNet18, |f_pre - f_post|, MLP)",
                        "output_mapping": "un-classified output ignored; minor-damage merged into major-damage",
                        "classes": CLASS_NAMES},
        }

    def run(self, before_path, after_path) -> tuple[dict, dict]:
        pre = read_image(before_path, "before")
        post = read_image(after_path, "after")
        warnings = validate_pair(pre, post)
        h, w = post.shape[:2]

        # ---- alignment ------------------------------------------------------
        glob_al = align.estimate_global_alignment(pre, post, tolerance_px=CROP_PADDING)
        pre_used = pre
        if (self.cfg.alignment == "register" and glob_al.method != "none"
                and glob_al.status in ("minor_offset", "misaligned")):
            pre_used = align.register_pre_to_post(pre, glob_al.dx, glob_al.dy)
            glob_al.registered = True
        if glob_al.note:
            warnings.append(f"alignment: {glob_al.note}")
        if glob_al.status == "misaligned" and not glob_al.registered:
            warnings.append("pre/post images are globally offset beyond the crop padding; "
                            "damage labels for small buildings are less reliable")

        # ---- segmentation ---------------------------------------------------
        yolo_device = 0 if self.device.type == "cuda" else "cpu"
        results = self.yolo.predict(source=post, conf=self.cfg.conf, iou=self.cfg.iou,
                                    imgsz=self.cfg.imgsz, max_det=self.cfg.max_det,
                                    device=yolo_device, retina_masks=True, verbose=False)
        detections, dropped = extract_buildings(results[0], w, h, self.cfg.min_area_px)
        if results[0].masks is None and detections:
            warnings.append("YOLO returned boxes without masks; bbox geometry used")

        # ---- classification -------------------------------------------------
        kept, pre_crops, post_crops = [], [], []
        for det in detections:
            cb, ca = crop_like_training(pre_used, det.polygon), crop_like_training(post, det.polygon)
            if cb is None or ca is None:
                dropped += 1
                continue
            kept.append(det); pre_crops.append(cb); post_crops.append(ca)
        probs = classify_pairs(self.siamese, pre_crops, post_crops, self.device)

        # ---- assemble buildings --------------------------------------------
        buildings = []
        for idx, (det, p) in enumerate(zip(kept, probs)):
            order = np.argsort(p)[::-1]
            cls = int(order[0])
            dmg_conf = float(p[order[0]])
            margin = float(p[order[0]] - p[order[1]])
            bconf = det.detection_confidence * dmg_conf
            poly = det.polygon
            c = poly.centroid
            minx, miny, maxx, maxy = poly.bounds
            bbox = [int(np.floor(minx)), int(np.floor(miny)), int(np.ceil(maxx)), int(np.ceil(maxy))]
            quadrant, half_lr, half_tb = region_of(c.x, c.y, w, h)

            b = {
                "building_id": idx,
                "predicted_damage": CLASS_NAMES[cls],
                "class_id": cls,
                "is_damaged": CLASS_NAMES[cls] in DAMAGED_CLASSES,
                "class_probabilities": {n: round(float(p[i]), 4) for i, n in enumerate(CLASS_NAMES)},
                "detection_confidence": round(det.detection_confidence, 4),
                "damage_confidence": round(dmg_conf, 4),
                "damage_margin": round(margin, 4),
                "building_confidence": round(bconf, 4),
                "confidence_level": confidence_level(bconf),
                # 3 decimals: 1-decimal rounding can move a centre across the
                # quadrant boundary (e.g. 511.96 -> 512.0).
                "centre_xy": [round(c.x, 3), round(c.y, 3)],
                "bbox_xyxy": bbox,
                "polygon_xy": [[round(x, 2), round(y, 2)] for x, y in list(poly.exterior.coords)[:-1]],
                "area_px": round(poly.area, 1),
                "geometry_source": det.geometry_source,
                "quadrant": quadrant,
                "half_horizontal": half_lr,
                "half_vertical": half_tb,
            }
            if self.cfg.local_alignment:
                ls = align.local_shift(pre_used, post, bbox)
                if ls is None:
                    b["local_alignment"] = {"status": "not_measured"}
                else:
                    ldx, ldy, lresp = ls
                    shift = float(np.hypot(ldx, ldy))
                    if lresp < 0.10:
                        st = "uncertain"   # low texture or real change (e.g. destroyed)
                    elif shift > CROP_PADDING:
                        st = "offset_exceeds_padding"
                    else:
                        st = "consistent"
                    b["local_alignment"] = {"status": st, "dx": round(ldx, 2), "dy": round(ldy, 2),
                                            "response": round(lresp, 3)}
            buildings.append(b)

        summary = self._summarise(buildings, w, h)
        report = {
            "schema": "eagle-dar/building-damage/v1",
            "pipeline_version": PIPELINE_VERSION,
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "inputs": {"before_image": str(Path(before_path).resolve()),
                       "after_image": str(Path(after_path).resolve()),
                       "width": w, "height": h},
            "models": self.model_info,
            "parameters": {"yolo_conf": self.cfg.conf, "yolo_imgsz": self.cfg.imgsz,
                           "yolo_iou": self.cfg.iou, "yolo_max_det": self.cfg.max_det,
                           "segmentation_image": "after", "crop_padding_px": CROP_PADDING,
                           "crop_size_px": CROP_SIZE, "alignment_mode": self.cfg.alignment,
                           "device": str(self.device), "torch": torch.__version__,
                           "python": platform.python_version()},
            "alignment": glob_al.to_dict(),
            "total_detected_buildings": len(buildings),
            "dropped_detections": dropped,
            "summary": summary,
            "confidence_definition": CONFIDENCE_DEFINITION,
            "buildings": buildings,
            "warnings": warnings,
            "disclaimer": "Model estimates only. Building recall of the segmentation model is "
                          "limited (see evaluation report); counts are likely underestimates. "
                          "Verify visually before operational use.",
        }
        images = {"pre": pre, "pre_used": pre_used, "post": post,
                  "pre_crops": pre_crops, "post_crops": post_crops}
        return report, images

    @staticmethod
    def _summarise(buildings, w, h):
        counts = {n: 0 for n in CLASS_NAMES}
        quad = {q: {n: 0 for n in CLASS_NAMES} for q in ("top-left", "top-right", "bottom-left", "bottom-right")}
        halves = {hn: {n: 0 for n in CLASS_NAMES} for hn in ("left", "right", "top", "bottom")}
        levels = {"high": 0, "medium": 0, "low": 0}
        for b in buildings:
            n = b["predicted_damage"]
            counts[n] += 1
            quad[b["quadrant"]][n] += 1
            halves[b["half_horizontal"]][n] += 1
            halves[b["half_vertical"]][n] += 1
            levels[b["confidence_level"]] += 1

        def damaged(d):
            return sum(d[c] for c in DAMAGED_CLASSES)

        return {
            "damage_counts": counts,
            "damaged_total": damaged(counts),
            "quadrant_counts": {q: {**d, "damaged": damaged(d)} for q, d in quad.items()},
            "half_counts": {hn: {**d, "damaged": damaged(d)} for hn, d in halves.items()},
            "confidence_levels": levels,
            "mean_building_confidence": (round(float(np.mean([b["building_confidence"] for b in buildings])), 4)
                                         if buildings else None),
        }


# --------------------------------------------------------------------------
# Rendering: three-class colour-coded damage map
# --------------------------------------------------------------------------

def render_damage_map(post: np.ndarray, report: dict, alpha: float = 0.40,
                      draw_ids: bool = True, draw_centre_lines: bool = True) -> np.ndarray:
    h, w = post.shape[:2]
    fill = post.copy()
    for b in report["buildings"]:
        pts = np.round(np.array(b["polygon_xy"], dtype=np.float32)).astype(np.int32)
        color = DAMAGE_COLORS_BGR[b["predicted_damage"]]
        cv2.fillPoly(fill, [pts], color)
    canvas = cv2.addWeighted(fill, alpha, post, 1 - alpha, 0)
    for b in report["buildings"]:
        pts = np.round(np.array(b["polygon_xy"], dtype=np.float32)).astype(np.int32)
        color = DAMAGE_COLORS_BGR[b["predicted_damage"]]
        thickness = 2 if b["confidence_level"] != "low" else 1
        cv2.polylines(canvas, [pts], True, color, thickness, cv2.LINE_AA)
        if draw_ids:
            cx, cy = (int(v) for v in b["centre_xy"])
            cv2.putText(canvas, str(b["building_id"]), (cx - 6, cy + 4), cv2.FONT_HERSHEY_SIMPLEX,
                        0.35, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(canvas, str(b["building_id"]), (cx - 6, cy + 4), cv2.FONT_HERSHEY_SIMPLEX,
                        0.35, (0, 0, 0), 1, cv2.LINE_AA)
    if draw_centre_lines:
        for x in range(0, w, 12):
            cv2.line(canvas, (x, h // 2), (min(x + 6, w), h // 2), (230, 230, 230), 1)
        for y in range(0, h, 12):
            cv2.line(canvas, (w // 2, y), (w // 2, min(y + 6, h)), (230, 230, 230), 1)

    # Legend panel
    panel_w = 300
    panel = np.full((h, panel_w, 3), 250, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    y = 34
    cv2.putText(panel, "EAGLE-DAR damage map", (14, y), font, 0.6, (30, 30, 30), 2, cv2.LINE_AA)
    y += 34
    counts = report["summary"]["damage_counts"]
    for name in CLASS_NAMES:
        cv2.rectangle(panel, (14, y - 14), (36, y + 4), DAMAGE_COLORS_BGR[name], -1)
        cv2.rectangle(panel, (14, y - 14), (36, y + 4), (60, 60, 60), 1)
        cv2.putText(panel, f"{name}: {counts[name]}", (46, y), font, 0.5, (30, 30, 30), 1, cv2.LINE_AA)
        y += 30
    y += 6
    lines = [
        f"buildings detected: {report['total_detected_buildings']}",
        f"damaged (major+destroyed): {report['summary']['damaged_total']}",
        f"mean building conf: {report['summary']['mean_building_confidence']}",
        f"alignment: {report['alignment']['status']} "
        f"({report['alignment']['shift_px']:.1f}px"
        f"{', registered' if report['alignment']['registered'] else ''})",
        "thin outline = low confidence",
        "dashed lines = quadrant split",
        "Model estimates - verify visually",
    ]
    for line in lines:
        cv2.putText(panel, line, (14, y), font, 0.42, (50, 50, 50), 1, cv2.LINE_AA)
        y += 24
    return np.hstack([canvas, panel])


def render_comparison(pre: np.ndarray, post: np.ndarray, damage_map: np.ndarray, width: int = 1800) -> np.ndarray:
    """Before | After | Damage map strip for reports."""
    h = post.shape[0]
    strip = np.hstack([pre, post, damage_map])
    header = np.full((40, strip.shape[1], 3), 255, dtype=np.uint8)
    for x, text in ((10, "BEFORE"), (post.shape[1] + 10, "AFTER"), (2 * post.shape[1] + 10, "DAMAGE MAP")):
        cv2.putText(header, text, (x, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2, cv2.LINE_AA)
    strip = np.vstack([header, strip])
    scale = width / strip.shape[1]
    return cv2.resize(strip, (width, int(strip.shape[0] * scale)), interpolation=cv2.INTER_AREA)


def write_image(path: Path, img: np.ndarray):
    ok, buf = cv2.imencode(path.suffix, img)
    if not ok:
        raise InferenceError(f"Could not encode image for {path}")
    buf.tofile(str(path))


def save_outputs(report: dict, images: dict, out_dir: Path, name: str, save_crops: bool = False) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": out_dir / f"{name}_predictions.json",
        "damage_map": out_dir / f"{name}_damage_map.png",
        "comparison": out_dir / f"{name}_comparison.jpg",
    }
    dmap = render_damage_map(images["post"], report)
    write_image(paths["damage_map"], dmap)
    write_image(paths["comparison"], render_comparison(images["pre"], images["post"], dmap))
    if save_crops and images["pre_crops"]:
        crop_dir = out_dir / f"{name}_crops"
        crop_dir.mkdir(exist_ok=True)
        for b, cb, ca in zip(report["buildings"], images["pre_crops"], images["post_crops"]):
            write_image(crop_dir / f"b{b['building_id']:04d}_{b['predicted_damage']}.png", np.hstack([cb, ca]))
        paths["crops"] = crop_dir
    report["outputs"] = {k: str(v) for k, v in paths.items()}
    paths["json"].write_text(json.dumps(report, indent=2), encoding="utf-8")
    return paths
