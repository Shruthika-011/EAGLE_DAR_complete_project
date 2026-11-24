
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

# ------------------------------------------------------------
# APPLICATION-LEVEL CLASSES (what the pipeline, API, reports and UI show)
#
# The trained Siamese model still has the five outputs above. At inference the
# 'un-classified' output is ignored (argmax over the other four) and
# 'minor-damage' is merged into 'major-damage', so the app works with three
# classes. Training / evaluation scripts keep using DAMAGE_CLASSES above.
# ------------------------------------------------------------

APP_DAMAGE_CLASSES = {
    "no-damage": 0,
    "major-damage": 1,
    "destroyed": 2,
}

APP_DAMAGE_NAMES = {value: key for key, value in APP_DAMAGE_CLASSES.items()}

# model output index -> app class index (4 = un-classified has no app class)
MODEL_TO_APP_CLASS = {0: 0, 1: 1, 2: 1, 3: 2}

# dataset 'subtype' string -> app class name (anything else is not an app class)
SUBTYPE_TO_APP_NAME = {
    "no-damage": "no-damage",
    "minor-damage": "major-damage",
    "major-damage": "major-damage",
    "destroyed": "destroyed",
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
# INFERENCE CONFIGURATION
# ============================================================

# Trained weights actually produced by training (runs/eagle_buildings/ is empty;
# the completed run is eagle_buildings-2).
YOLO_WEIGHTS = RUNS_DIR / "eagle_buildings-2" / "weights" / "best.pt"
SIAMESE_WEIGHTS = MODELS_DIR / "siamese_damage.pth"

YOLO_CONF = 0.25
YOLO_IMGSZ = 640
YOLO_IOU = 0.7
YOLO_MAX_DET = 1000

# Three-class colour-coded damage map (BGR for OpenCV).
DAMAGE_COLORS_BGR = {
    "no-damage": (80, 175, 76),       # green
    "major-damage": (0, 152, 255),    # orange
    "destroyed": (54, 67, 244),       # red
}

# Quadrants and halves follow PS-04 rules: split at image centre,
# a building belongs to the part containing its centre.
DAMAGED_CLASSES = ("major-damage", "destroyed")

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
