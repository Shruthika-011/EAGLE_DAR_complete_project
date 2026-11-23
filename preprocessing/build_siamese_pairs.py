import json
import re
import csv
from pathlib import Path

import cv2
from shapely import wkt
from shapely.geometry import Polygon, MultiPolygon

from config import (
    SOURCE_IMAGES, SOURCE_LABELS, IMAGE_EXTENSIONS,
    SIAMESE_ROOT, SIAMESE_CROPS, PAIR_RULES,
    CROP_SIZE, CROP_PADDING, DAMAGE_CLASSES,
)

def scene_key(filename, role):
    stem = Path(filename).stem.lower()
    for token in PAIR_RULES[role]:
        if token in stem:
            return stem.replace(token, "")
    return None

def find_image_by_stem(stem):
    for ext in IMAGE_EXTENSIONS:
        p = SOURCE_IMAGES / f"{stem}{ext}"
        if p.exists():
            return p
    return None

def get_polygons(geom):
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return list(geom.geoms)
    return []

def crop_polygon(image, polygon, size=CROP_SIZE):
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
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)

def main():
    SIAMESE_CROPS.mkdir(parents=True, exist_ok=True)
    SIAMESE_ROOT.mkdir(parents=True, exist_ok=True)

    image_files = [p for p in SOURCE_IMAGES.iterdir()
                   if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]

    before_map, after_map = {}, {}
    for p in image_files:
        bkey = scene_key(p.name, "before")
        akey = scene_key(p.name, "after")
        if bkey is not None:
            before_map[bkey] = p
        if akey is not None:
            after_map[akey] = p

    common = sorted(set(before_map) & set(after_map))
    if not common:
        print("No pre/post image pairs found.")
        print("Expected filenames like scene01_pre_disaster.png and scene01_post_disaster.png.")
        print("YOLO preparation can still be used; Siamese training cannot run without pairs.")
        return

    records = []
    for key in common:
        before_path = before_map[key]
        after_path = after_map[key]
        before = cv2.imread(str(before_path))
        after = cv2.imread(str(after_path))
        if before is None or after is None:
            print(f"Skip {key}: image read failure")
            continue
        if before.shape[:2] != after.shape[:2]:
            print(f"Skip {key}: before/after dimensions differ; register/resize deliberately first.")
            continue

        # The post-disaster annotation JSON must share the post image stem.
        label_path = SOURCE_LABELS / f"{key}.json"
        if not label_path.exists():
            print(f"Skip {key}: missing post-disaster JSON {label_path.name}")
            continue

        data = json.loads(label_path.read_text(encoding="utf-8"))
        features = data.get("features", {}).get("xy", [])
        count = 0

        for feature in features:
            props = feature.get("properties", {})
            if props.get("feature_type") != "building":
                continue
            damage = str(props.get("subtype", "un-classified")).lower()
            if damage not in DAMAGE_CLASSES:
                continue
            try:
                geom = wkt.loads(feature["wkt"])
            except Exception:
                continue

            for poly in get_polygons(geom):
                crop_before = crop_polygon(before, poly)
                crop_after = crop_polygon(after, poly)
                if crop_before is None or crop_after is None:
                    continue

                uid = re.sub(r"[^A-Za-z0-9_-]", "_", str(props.get("uid", f"building{count}")))
                prefix = f"{key}_{uid}"
                before_name = f"{prefix}_before.png"
                after_name = f"{prefix}_after.png"
                cv2.imwrite(str(SIAMESE_CROPS / before_name), crop_before)
                cv2.imwrite(str(SIAMESE_CROPS / after_name), crop_after)

                records.append({
                    "before": f"processed/siamese/crops/{before_name}",
                    "after": f"processed/siamese/crops/{after_name}",
                    "label": DAMAGE_CLASSES[damage],
                    "damage_name": damage,
                    "scene": key,
                    "uid": str(props.get("uid", "")),
                })
                count += 1

        print(f"{key}: created {count} building pairs")

    if not records:
        print("No Siamese crop pairs created. Check post JSON names, WKT and image pairs.")
        return

    # Split by scene to avoid near-duplicate scene leakage.
    scenes = sorted({r["scene"] for r in records})
    if len(scenes) >= 2:
        from sklearn.model_selection import train_test_split
        train_scenes, val_scenes = train_test_split(
            scenes, test_size=0.2, random_state=42
        )
        train_scenes, val_scenes = set(train_scenes), set(val_scenes)
    else:
        print("WARNING: only one scene found; no independent validation scene is possible.")
        train_scenes, val_scenes = set(scenes), set()

    for split, selected in [("train", train_scenes), ("val", val_scenes)]:
        path = SIAMESE_ROOT / f"pairs_{split}.csv"
        selected_records = [r for r in records if r["scene"] in selected]
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(selected_records)
        print(f"{split} pairs: {len(selected_records)}")

    print(f"\nTotal Siamese pairs created: {len(records)}")
    print("Review pair crops visually before training. Images must be geographically aligned.")

if __name__ == "__main__":
    main()
