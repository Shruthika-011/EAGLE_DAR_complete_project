"""Build a Scene from Phase 1 prediction JSON or from ground-truth labels."""
from __future__ import annotations

import json
from pathlib import Path

from config import SUBTYPE_TO_APP_NAME
from reasoning import spatial
from reasoning.schema import CLASS_NAMES, Building, Scene

PREDICTION_SCHEMA = "eagle-dar/building-damage/v1"


def scene_from_prediction_json(path) -> Scene:
    """Phase 1 output -> Scene. The centre is recomputed from the polygon (full
    precision) rather than trusting a rounded stored value, and quadrant/half
    fields stored by Phase 1 are ignored: Phase 2 re-derives them."""
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    if data.get("schema") != PREDICTION_SCHEMA:
        raise ValueError(f"{p}: expected schema {PREDICTION_SCHEMA!r}, got {data.get('schema')!r}")
    w, h = int(data["inputs"]["width"]), int(data["inputs"]["height"])
    buildings = []
    for b in data["buildings"]:
        poly = tuple((float(x), float(y)) for x, y in b.get("polygon_xy") or ())
        if len(poly) >= 3:
            centre = spatial.polygon_centroid(poly)
            bbox = spatial.polygon_bbox(poly, w, h)
        elif b.get("bbox_xyxy"):
            x1, y1, x2, y2 = (int(v) for v in b["bbox_xyxy"])
            bbox, centre = (x1, y1, x2, y2), ((x1 + x2) / 2, (y1 + y2) / 2)
        else:
            raise ValueError(f"{p}: building {b.get('building_id')!r} has neither polygon nor bbox")
        meta = {k: b[k] for k in ("detection_confidence", "damage_confidence", "damage_margin",
                                  "confidence_level", "class_probabilities", "geometry_source",
                                  "local_alignment", "area_px") if k in b}
        name = b.get("predicted_damage")
        if name is not None:
            name = SUBTYPE_TO_APP_NAME.get(str(name).lower())      # also reads legacy 5-class files
            if name is None:
                continue                                           # legacy 'un-classified' building
            class_id = CLASS_NAMES.index(name)
        else:
            class_id = int(b["class_id"])
        buildings.append(Building(
            building_id=b["building_id"], class_id=class_id, centre=centre, bbox=bbox,
            polygon=poly, confidence=b.get("building_confidence"), metadata=meta))
    meta = {k: data.get(k) for k in ("alignment", "warnings", "models", "parameters", "pipeline_version")}
    meta["source_json"] = str(p)
    return Scene(w, h, tuple(buildings), scene_id=Path(data["inputs"]["after_image"]).stem.removesuffix("_post"),
                 metadata=meta)


def scene_from_ground_truth(label_path) -> Scene:
    """Official label JSON -> Scene (confidence None). For validating the
    reasoning logic in isolation from model error; never report it as model
    performance."""
    from shapely import wkt  # lazy: only needed here

    p = Path(label_path)
    data = json.loads(p.read_text(encoding="utf-8"))
    md = data.get("metadata", {})
    w, h = int(md.get("width", 1024)), int(md.get("height", 1024))
    buildings = []
    for f in data.get("features", {}).get("xy", []):
        props = f.get("properties", {})
        if props.get("feature_type") != "building":
            continue
        subtype = SUBTYPE_TO_APP_NAME.get(str(props.get("subtype", "un-classified")).lower())
        if subtype is None:        # un-classified: not an app class
            continue
        g = wkt.loads(f["wkt"])
        if g.is_empty:
            continue
        c = g.centroid
        minx, miny, maxx, maxy = g.bounds
        bbox = spatial.polygon_bbox([(minx, miny), (maxx, maxy)], w, h)
        outer = g if g.geom_type == "Polygon" else max(g.geoms, key=lambda x: x.area)
        buildings.append(Building(
            building_id=str(props.get("uid")), class_id=CLASS_NAMES.index(subtype),
            centre=(c.x, c.y), bbox=bbox, polygon=tuple((x, y) for x, y in list(outer.exterior.coords)[:-1]),
            confidence=None, metadata={"source": "ground_truth"}))
    return Scene(w, h, tuple(buildings), scene_id=p.stem, metadata={"source_json": str(p), "source": "ground_truth"})
