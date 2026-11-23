import json
import shutil
from pathlib import Path

import cv2
import yaml
from shapely import wkt
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
from sklearn.model_selection import train_test_split

from config import (
    SOURCE_IMAGES, SOURCE_LABELS, IMAGE_EXTENSIONS,
    YOLO_IMAGES, YOLO_LABELS, METADATA_DIR,
    VAL_FRACTION, RANDOM_SEED, BUILDING_CLASS_ID,
)

def find_image(stem):
    for ext in IMAGE_EXTENSIONS:
        candidate = SOURCE_IMAGES / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None

def polygons_from_geometry(geom):
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return list(geom.geoms)
    if isinstance(geom, GeometryCollection):
        result = []
        for item in geom.geoms:
            result.extend(polygons_from_geometry(item))
        return result
    return []

def make_yolo_line(poly, width, height):
    # YOLO segmentation format: class x1 y1 x2 y2 ...
    coords = list(poly.exterior.coords)
    if len(coords) < 4:
        return None
    values = []
    for x, y in coords[:-1]:
        # Clip polygon points to image boundary.
        x = min(max(float(x), 0.0), width - 1.0)
        y = min(max(float(y), 0.0), height - 1.0)
        values.extend([x / width, y / height])
    if len(values) < 6:
        return None
    return str(BUILDING_CLASS_ID) + " " + " ".join(f"{v:.6f}" for v in values)

def process_one(json_path):
    data = json.loads(json_path.read_text(encoding="utf-8"))
    image_path = find_image(json_path.stem)
    if image_path is None:
        print(f"SKIP {json_path.name}: matching image not found")
        return None

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        print(f"SKIP {image_path.name}: OpenCV cannot read image")
        return None

    height, width = image.shape[:2]
    metadata = data.get("metadata", {})
    declared_w = metadata.get("width")
    declared_h = metadata.get("height")
    if declared_w and declared_h and (int(declared_w) != width or int(declared_h) != height):
        print(f"SKIP {json_path.name}: metadata says {declared_w}x{declared_h}, image is {width}x{height}")
        return None

    yolo_lines = []
    rows = []
    features = data.get("features", {}).get("xy", [])

    for feature in features:
        props = feature.get("properties", {})
        if props.get("feature_type") != "building":
            continue
        uid = str(props.get("uid", ""))
        damage = str(props.get("subtype", "un-classified")).lower()
        try:
            geom = wkt.loads(feature["wkt"])
        except Exception as exc:
            print(f"WARNING: bad WKT in {json_path.name}, uid={uid}: {exc}")
            continue

        for poly in polygons_from_geometry(geom):
            line = make_yolo_line(poly, width, height)
            if line:
                yolo_lines.append(line)
                minx, miny, maxx, maxy = poly.bounds
                rows.append({
                    "image": image_path.name,
                    "json": json_path.name,
                    "uid": uid,
                    "damage": damage,
                    "disaster": metadata.get("disaster", ""),
                    "disaster_type": metadata.get("disaster_type", ""),
                    "width": width,
                    "height": height,
                    "x1": minx, "y1": miny, "x2": maxx, "y2": maxy,
                    "wkt": poly.wkt,
                })

    return image_path, yolo_lines, rows

def main():
    for folder in [YOLO_IMAGES / "train", YOLO_IMAGES / "val",
                   YOLO_LABELS / "train", YOLO_LABELS / "val", METADATA_DIR]:
        folder.mkdir(parents=True, exist_ok=True)

    json_files = sorted(SOURCE_LABELS.glob("*.json"))
    if not json_files:
        raise SystemExit(f"No JSON files found in {SOURCE_LABELS}")

    valid = []
    for jf in json_files:
        item = process_one(jf)
        if item:
            valid.append((jf.stem, *item))

    if not valid:
        raise SystemExit("No valid image/JSON pairs. Check matching filenames and image dimensions.")

    stems = [x[0] for x in valid]
    if len(stems) >= 2:
        train_stems, val_stems = train_test_split(
            stems, test_size=VAL_FRACTION, random_state=RANDOM_SEED
        )
        train_stems, val_stems = set(train_stems), set(val_stems)
    else:
        print("WARNING: only one labeled image; placing it in train. Validation metrics will not be available.")
        train_stems, val_stems = set(stems), set()

    all_rows = []
    for stem, image_path, lines, rows in valid:
        split = "val" if stem in val_stems else "train"
        shutil.copy2(image_path, YOLO_IMAGES / split / image_path.name)
        label_path = YOLO_LABELS / split / f"{stem}.txt"
        label_path.write_text("\n".join(lines), encoding="utf-8")
        all_rows.extend(rows)
        print(f"{split}: {image_path.name} | building polygons: {len(lines)}")

    import csv
    csv_path = METADATA_DIR / "buildings.csv"
    fieldnames = list(all_rows[0].keys()) if all_rows else []
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(all_rows)

    yaml_data = {
        "path": str(YOLO_IMAGES.parent.parent.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "building"},
    }
    (YOLO_IMAGES.parent.parent / "data.yaml").write_text(
        yaml.safe_dump(yaml_data, sort_keys=False), encoding="utf-8"
    )

    print("\nYOLO preprocessing complete.")
    print(f"Images: {len(valid)}")
    print(f"Building polygons: {len(all_rows)}")
    print(f"Metadata: {csv_path}")
    print(f"YOLO config: {YOLO_IMAGES.parent.parent / 'data.yaml'}")

if __name__ == "__main__":
    main()
