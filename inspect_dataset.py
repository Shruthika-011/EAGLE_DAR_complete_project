import json
from collections import Counter
from config import SOURCE_IMAGES, SOURCE_LABELS, IMAGE_EXTENSIONS

def main():
    if not SOURCE_IMAGES.exists() or not SOURCE_LABELS.exists():
        raise SystemExit(
            f"Expected folders:\n{SOURCE_IMAGES}\n{SOURCE_LABELS}\n"
            "Create them and put your images and JSON labels inside."
        )

    images = [p for p in SOURCE_IMAGES.iterdir()
              if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    labels = list(SOURCE_LABELS.glob("*.json"))

    image_stems = {p.stem for p in images}
    print(f"Images found: {len(images)}")
    print(f"JSON labels found: {len(labels)}")
    print("\nFirst image filenames:")
    for p in images[:15]:
        print(" ", p.name)

    print("\nFirst JSON filenames:")
    for p in labels[:15]:
        print(" ", p.name)

    print("\nAnnotation summary:")
    total = Counter()
    for path in labels:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            features = data.get("features", {}).get("xy", [])
            counts = Counter(
                f.get("properties", {}).get("subtype", "missing")
                for f in features
                if f.get("properties", {}).get("feature_type") == "building"
            )
            total.update(counts)
            print(f"{path.name}: {sum(counts.values())} buildings | {dict(counts)}")
            if path.stem not in image_stems:
                print("  WARNING: no image with matching filename stem")
        except Exception as exc:
            print(f"ERROR reading {path.name}: {exc}")

    print("\nTotal damage-label distribution:", dict(total))
    print("\nNote: matching filename stems are required for YOLO preparation.")

if __name__ == "__main__":
    main()
