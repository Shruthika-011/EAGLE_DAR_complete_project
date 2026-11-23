
from pathlib import Path

ROOT = Path("dataset_source/train")

images_dir = ROOT / "images"
labels_dir = ROOT / "labels"

pre = list(images_dir.glob("*_pre.jpg"))
post = list(images_dir.glob("*_post.jpg"))
labels = list(labels_dir.glob("*.json"))

pre_ids = {p.stem.replace("_pre", "") for p in pre}
post_ids = {p.stem.replace("_post", "") for p in post}
label_ids = {p.stem for p in labels}

print("PRE IMAGES:", len(pre))
print("POST IMAGES:", len(post))
print("JSON LABELS:", len(labels))

print("COMPLETE IMAGE PAIRS:", len(pre_ids & post_ids))
print("MISSING POST:", len(pre_ids - post_ids))
print("MISSING PRE:", len(post_ids - pre_ids))

print("PAIRS WITH LABELS:", len((pre_ids & post_ids) & label_ids))

print("\nSAMPLE:")
for image_id in sorted(pre_ids & post_ids)[:5]:
    print(image_id)
