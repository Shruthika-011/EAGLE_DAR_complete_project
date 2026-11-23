
from pathlib import Path

# ============================================================
# PROJECT ROOT
# ============================================================

ROOT = Path(__file__).resolve().parent

# ============================================================
# SOURCE DATASET - HUGGING FACE DATASET
# ============================================================

DATASET_ROOT = ROOT / "dataset_source"
TRAIN_DIR = DATASET_ROOT / "train"

SOURCE_IMAGES = ROOT / "dataset" / "raw" / "train" / "images"
SOURCE_LABELS = ROOT / "dataset" / "raw" / "train" / "labels"
# ============================================================
# GENERATED PROCESSED DATA
# ============================================================

PROCESSED = ROOT / "processed"

# YOLO dataset
YOLO_ROOT = PROCESSED / "yolo"
YOLO_IMAGES = YOLO_ROOT / "images"
YOLO_LABELS = YOLO_ROOT / "labels"

# Metadata
METADATA_DIR = PROCESSED / "metadata"

# Siamese CNN dataset
SIAMESE_ROOT = PROCESSED / "siamese"
SIAMESE_CROPS = SIAMESE_ROOT / "crops"

# ============================================================
# MODELS AND TRAINING OUTPUTS
# ============================================================

RUNS_DIR = ROOT / "runs"
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"

# ============================================================
# YOLO SEGMENTATION CONFIGURATION
# ============================================================

BUILDING_CLASS_ID = 0
BUILDING_CLASS_NAME = "building"

# ============================================================
# DAMAGE CLASS CONFIGURATION
# ============================================================

DAMAGE_CLASSES = {
    "no-damage": 0,
    "minor-damage": 1,
    "major-damage": 2,
    "destroyed": 3,
    "un-classified": 4,
}

DAMAGE_NAMES = {
    value: key
    for key, value in DAMAGE_CLASSES.items()
}

# ============================================================
# IMAGE CONFIGURATION
# ============================================================

IMAGE_EXTENSIONS = [
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".bmp",
]

# ============================================================
# DATASET SPLIT
# ============================================================

VAL_FRACTION = 0.20
RANDOM_SEED = 42

# ============================================================
# BUILDING CROP CONFIGURATION
# ============================================================

CROP_SIZE = 128
CROP_PADDING = 8

# ============================================================
# BEFORE / AFTER IMAGE PAIR RULES
# ============================================================

PAIR_RULES = {
    "before": ["_pre"],
    "after": ["_post"],
}

# ============================================================
# DATASET VALIDATION
# ============================================================

if not SOURCE_IMAGES.exists():
    print(f"WARNING: Image directory not found: {SOURCE_IMAGES}")

if not SOURCE_LABELS.exists():
    print(f"WARNING: Label directory not found: {SOURCE_LABELS}")

print("EAGLE-DAR CONFIGURATION LOADED")
print("Project root:", ROOT)
print("Source images:", SOURCE_IMAGES)
print("Source labels:", SOURCE_LABELS)
print("Processed directory:", PROCESSED)
print("YOLO directory:", YOLO_ROOT)
