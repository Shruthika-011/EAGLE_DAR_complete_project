from ultralytics import YOLO
from config import ROOT, RUNS_DIR

def main():
    data_yaml = ROOT / "processed" / "yolo" / "data.yaml"
    if not data_yaml.exists():
        raise SystemExit("YOLO dataset config not found. Run: python preprocessing/prepare_yolo.py")

    # Nano segmentation model is a lighter starting point for 4 GB VRAM.
    model = YOLO("yolov8n-seg.pt")
    model.train(
        data=str(data_yaml),
        epochs=30,
        imgsz=640,
        batch=2,
        device=0,          # Change to "cpu" if CUDA is unavailable.
        workers=2,
        project=str(RUNS_DIR),
        name="eagle_buildings",
        patience=8,
        cache=False,
        pretrained=True,
    )
    print("Training finished. Check runs/eagle_buildings/weights/best.pt")

if __name__ == "__main__":
    main()
