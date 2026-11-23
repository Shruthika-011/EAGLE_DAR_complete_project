
from huggingface_hub import snapshot_download
from pathlib import Path

REPO_ID = "Tarakeshwaran/CIT_Hackathon_PS04_aerial"

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_ROOT / "dataset_source"

print("Starting Hugging Face dataset download...")
print("Dataset:", REPO_ID)
print("Destination:", DATASET_DIR)

snapshot_download(
    repo_id=REPO_ID,
    repo_type="dataset",
    local_dir=str(DATASET_DIR)
)

print("\nDataset download completed!")
print("Saved at:", DATASET_DIR)
