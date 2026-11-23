
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "dataset_source"

print("\nDATASET DIRECTORY STRUCTURE\n")

for path in sorted(ROOT.rglob("*")):
    relative = path.relative_to(ROOT)

    if path.is_dir():
        print(f"[DIR]  {relative}")
    else:
        print(f"[FILE] {relative}")

print("\nFILE COUNTS")

image_extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

images = [
    p for p in ROOT.rglob("*")
    if p.is_file() and p.suffix.lower() in image_extensions
]

json_files = list(ROOT.rglob("*.json"))

print("Images:", len(images))
print("JSON files:", len(json_files))
