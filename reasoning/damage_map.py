"""Novelty 2: five-class colour-coded building damage map.

Renders from structured buildings only (no model needed), so the map is
guaranteed to show exactly the buildings the reasoning engine counts.

    python -m reasoning.damage_map --predictions outputs/predictions/X_predictions.json \
        --image dataset_source/practice/images/X_post.jpg --out outputs/phase2/X_map.png \
        [--evidence 0,0,512,512] [--highlight 3,7,12]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from reasoning.schema import CLASS_NAMES, Scene

# BGR. green / orange / red, one fixed colour per class.
CLASS_COLORS_BGR = {
    "no-damage": (80, 175, 76),
    "major-damage": (0, 152, 255),
    "destroyed": (54, 67, 244),
}
CLASS_COLOR_LABELS = {"no-damage": "green", "major-damage": "orange", "destroyed": "red"}
assert tuple(CLASS_COLORS_BGR) == CLASS_NAMES


def _polygon_points(b) -> np.ndarray:
    if b.polygon:
        return np.round(np.array(b.polygon, dtype=np.float32)).astype(np.int32)
    x1, y1, x2, y2 = b.bbox
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)


def render_damage_map(image: np.ndarray, scene: Scene, alpha: float = 0.40, draw_ids: bool = True,
                      draw_split_lines: bool = True, evidence_region=None, highlight_ids=(),
                      legend: bool = True) -> np.ndarray:
    """Return a BGR image: the post image with every building filled in its class colour."""
    if image.shape[:2] != (scene.height, scene.width):
        raise ValueError(f"image is {image.shape[1]}x{image.shape[0]} but the scene is "
                         f"{scene.width}x{scene.height}")
    fill = image.copy()
    polys = [(b, _polygon_points(b)) for b in scene.buildings]
    for b, pts in polys:
        cv2.fillPoly(fill, [pts], CLASS_COLORS_BGR[b.damage_name])
    canvas = cv2.addWeighted(fill, alpha, image, 1 - alpha, 0)
    highlight = set(highlight_ids)
    for b, pts in polys:
        thick = 3 if b.building_id in highlight else (1 if b.confidence is not None and b.confidence < 0.35 else 2)
        cv2.polylines(canvas, [pts], True, CLASS_COLORS_BGR[b.damage_name], thick, cv2.LINE_AA)
        if b.building_id in highlight:
            cv2.polylines(canvas, [pts], True, (255, 255, 255), 1, cv2.LINE_AA)
        if draw_ids:
            cx, cy = int(b.centre[0]), int(b.centre[1])
            label = str(b.building_id)
            cv2.putText(canvas, label, (cx - 6, cy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(canvas, label, (cx - 6, cy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1, cv2.LINE_AA)
    if draw_split_lines:
        for x in range(0, scene.width, 12):
            cv2.line(canvas, (x, scene.height // 2), (min(x + 6, scene.width), scene.height // 2), (235, 235, 235), 1)
        for y in range(0, scene.height, 12):
            cv2.line(canvas, (scene.width // 2, y), (scene.width // 2, min(y + 6, scene.height)), (235, 235, 235), 1)
    if evidence_region is not None:
        x1, y1, x2, y2 = (int(v) for v in evidence_region)
        cv2.rectangle(canvas, (x1, y1), (max(x1, x2 - 1), max(y1, y2 - 1)), (255, 0, 255), 3)
    if not legend:
        return canvas
    return np.hstack([canvas, _legend_panel(scene, canvas.shape[0])])


def _legend_panel(scene: Scene, height: int, width: int = 310) -> np.ndarray:
    panel = np.full((height, width, 3), 250, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    counts = {n: 0 for n in CLASS_NAMES}
    for b in scene.buildings:
        counts[b.damage_name] += 1
    y = 32
    cv2.putText(panel, "EAGLE-DAR damage map", (14, y), font, 0.58, (30, 30, 30), 2, cv2.LINE_AA)
    y += 34
    for name in CLASS_NAMES:
        cv2.rectangle(panel, (14, y - 14), (36, y + 4), CLASS_COLORS_BGR[name], -1)
        cv2.rectangle(panel, (14, y - 14), (36, y + 4), (60, 60, 60), 1)
        cv2.putText(panel, f"{name}: {counts[name]}", (46, y), font, 0.5, (30, 30, 30), 1, cv2.LINE_AA)
        y += 30
    damaged = counts["major-damage"] + counts["destroyed"]
    confs = [b.confidence for b in scene.buildings if b.confidence is not None]
    for line in (f"buildings: {len(scene.buildings)}",
                 f"damaged (major/destroyed): {damaged}",
                 f"mean building conf: {np.mean(confs):.3f}" if confs else "building conf: n/a",
                 "thin outline = low confidence", "dashed = quadrant split",
                 "magenta box = evidence region", "Model estimates - verify visually"):
        y += 8
        cv2.putText(panel, line, (14, y), font, 0.42, (50, 50, 50), 1, cv2.LINE_AA)
        y += 16
    return panel


def main(argv=None):
    from reasoning.loaders import scene_from_prediction_json

    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--image", required=True, help="post-disaster image")
    ap.add_argument("--out", required=True)
    ap.add_argument("--evidence", default=None, help="x1,y1,x2,y2 evidence region to outline")
    ap.add_argument("--highlight", default="", help="comma-separated building ids to emphasise")
    args = ap.parse_args(argv)

    scene = scene_from_prediction_json(args.predictions)
    data = np.fromfile(args.image, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        print(f"ERROR: cannot read image {args.image}", file=sys.stderr)
        return 2
    ev = [int(v) for v in args.evidence.split(",")] if args.evidence else None
    ids = [int(v) if v.strip().lstrip("-").isdigit() else v.strip() for v in args.highlight.split(",") if v.strip()]
    out = render_damage_map(img, scene, evidence_region=ev, highlight_ids=ids)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(Path(args.out).suffix or ".png", out)
    buf.tofile(args.out)
    print(f"Saved {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
