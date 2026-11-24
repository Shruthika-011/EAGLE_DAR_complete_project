"""Spatial rules (PS-04). Standard library only.

* The image is split at its ACTUAL centre (width/2, height/2), not a hard-coded 512.
* A building belongs to the part that contains its CENTRE (polygon centroid).
* Each part is half-open: x in [lo, hi), with the far image edge belonging to the
  last part. The official templates do not say which side an exact-centre-line
  point goes to; we choose right/bottom. Float centroids almost never land
  exactly on the line, so this only matters for synthetic or integer centres.
* Quadrants are therefore an exact partition: every building is in exactly one.
"""
from __future__ import annotations

import math

QUADRANTS = ("top-left", "top-right", "bottom-left", "bottom-right")
HALVES = ("left", "right", "top", "bottom")
_OPPOSITE = {"left": "right", "right": "left", "top": "bottom", "bottom": "top"}


def polygon_centroid(points) -> tuple:
    """Area-weighted centroid (same as shapely's Polygon.centroid). Falls back to
    the vertex mean for degenerate polygons."""
    pts = [tuple(p) for p in points]
    if len(pts) > 1 and pts[0] == pts[-1]:
        pts = pts[:-1]
    if not pts:
        raise ValueError("empty polygon")
    n = len(pts)
    a = cx = cy = 0.0
    for i in range(n):
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % n]
        cross = x0 * y1 - x1 * y0
        a += cross
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross
    if abs(a) < 1e-9:
        return (sum(p[0] for p in pts) / n, sum(p[1] for p in pts) / n)
    return (cx / (3.0 * a), cy / (3.0 * a))


def polygon_bbox(points, width: int, height: int) -> tuple:
    """Integer bbox: floor of the minimum, ceil of the maximum (matches the
    official practice regions), clipped to the image."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (max(0, math.floor(min(xs))), max(0, math.floor(min(ys))),
            min(width, math.ceil(max(xs))), min(height, math.ceil(max(ys))))


def _check_centre(cx, cy, w, h):
    if not (0 <= cx <= w and 0 <= cy <= h):
        raise ValueError(f"centre ({cx}, {cy}) is outside the {w}x{h} image")


def quadrant_of(cx: float, cy: float, w: int, h: int) -> str:
    _check_centre(cx, cy, w, h)
    return f"{'bottom' if cy >= h / 2 else 'top'}-{'right' if cx >= w / 2 else 'left'}"


def in_half(cx: float, cy: float, w: int, h: int, half: str) -> bool:
    _check_centre(cx, cy, w, h)
    if half == "left":
        return cx < w / 2
    if half == "right":
        return cx >= w / 2
    if half == "top":
        return cy < h / 2
    if half == "bottom":
        return cy >= h / 2
    raise ValueError(f"unknown half {half!r}")


def opposite_half(half: str) -> str:
    return _OPPOSITE[half]


def is_valid_half_pair(a: str, b: str) -> bool:
    return a in _OPPOSITE and _OPPOSITE[a] == b


def region_rect(name: str, w: int, h: int) -> tuple:
    """Image-coordinate rectangle [x1, y1, x2, y2] of a quadrant or half.
    Matches question_templates.json for 1024x1024. For odd sizes the far edge of
    the lower part rounds up and the near edge of the upper part rounds down, so
    the rectangles always cover their areas."""
    lo_x, hi_x = math.ceil(w / 2), math.floor(w / 2)
    lo_y, hi_y = math.ceil(h / 2), math.floor(h / 2)
    rects = {
        "image": (0, 0, w, h),
        "left": (0, 0, lo_x, h), "right": (hi_x, 0, w, h),
        "top": (0, 0, w, lo_y), "bottom": (0, hi_y, w, h),
        "top-left": (0, 0, lo_x, lo_y), "top-right": (hi_x, 0, w, lo_y),
        "bottom-left": (0, hi_y, lo_x, h), "bottom-right": (hi_x, hi_y, w, h),
    }
    if name not in rects:
        raise ValueError(f"unknown region {name!r}")
    return rects[name]


def region_label(name: str) -> str:
    """Human/official-style name: halves are 'left half', quadrants stay as is."""
    return f"{name} half" if name in _OPPOSITE else name


def union_bbox(boxes, w: int, h: int):
    """Smallest box containing all boxes, clipped to the image. None if empty."""
    boxes = list(boxes)
    if not boxes:
        return None
    return (max(0, min(b[0] for b in boxes)), max(0, min(b[1] for b in boxes)),
            min(w, max(b[2] for b in boxes)), min(h, max(b[3] for b in boxes)))


def assign_quadrants(buildings, w: int, h: int) -> dict:
    """Partition buildings into the four quadrants (each building exactly once)."""
    out = {q: [] for q in QUADRANTS}
    for b in buildings:
        out[quadrant_of(b.centre[0], b.centre[1], w, h)].append(b)
    return out


def buildings_in_region(buildings, scope_kind: str, scope_name, w: int, h: int) -> list:
    if scope_kind == "image":
        return list(buildings)
    if scope_kind == "quadrant":
        return [b for b in buildings if quadrant_of(b.centre[0], b.centre[1], w, h) == scope_name]
    if scope_kind == "half":
        return [b for b in buildings if in_half(b.centre[0], b.centre[1], w, h, scope_name)]
    raise ValueError(f"unknown scope kind {scope_kind!r}")
