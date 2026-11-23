from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Existing user dataset
SOURCE_IMAGES = ROOT / "train" / "images"
SOURCE_LABELS = ROOT / "train" / "labels"

# Generated data
PROCESSED = ROOT / "processed"
YOLO_ROOT = PROCESSED / "yolo"
YOLO_IMAGES = YOLO_ROOT / "images"
YOLO_LABELS = YOLO_ROOT / "labels"
METADATA_DIR = PROCESSED / "metadata"
SIAMESE_ROOT = PROCESSED / "siamese"
SIAMESE_CROPS = SIAMESE_ROOT / "crops"

# Models and outputs
RUNS_DIR = ROOT / "runs"
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"

# YOLO segmentation: one class only. Damage classes are retained for Siamese.
BUILDING_CLASS_ID = 0
BUILDING_CLASS_NAME = "building"

DAMAGE_CLASSES = {
    "no-damage": 0,
    "minor-damage": 1,
    "major-damage": 2,
    "destroyed": 3,
    "un-classified": 4,
}
DAMAGE_NAMES = {v: k for k, v in DAMAGE_CLASSES.items()}

IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]
VAL_FRACTION = 0.20
RANDOM_SEED = 42
CROP_SIZE = 128
CROP_PADDING = 8

# Expected pair tokens in image filenames. The scene key is the filename
# after removing one of these tokens and the extension.
PAIR_RULES = {
    "before": ["_pre_disaster", "-pre-disaster", "_before", "-before"],
    "after": ["_post_disaster", "-post-disaster", "_after", "-after"],
}
