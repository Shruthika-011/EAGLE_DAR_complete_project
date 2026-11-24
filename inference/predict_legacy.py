import argparse
import json
from pathlib import Path

import cv2
import torch
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO

from config import ROOT, DAMAGE_NAMES, RUNS_DIR, MODELS_DIR, OUTPUTS_DIR
from training.siamese_model import SiameseDamageClassifier

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

def classify_crop(model, before_crop, after_crop, device):
    before_rgb = cv2.cvtColor(before_crop, cv2.COLOR_BGR2RGB)
    after_rgb = cv2.cvtColor(after_crop, cv2.COLOR_BGR2RGB)
    before_tensor = TRANSFORM(Image.fromarray(before_rgb)).unsqueeze(0).to(device)
    after_tensor = TRANSFORM(Image.fromarray(after_rgb)).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(before_tensor, after_tensor)
        probs = torch.softmax(logits, dim=1)[0]
        confidence, class_id = probs.max(dim=0)
    return int(class_id.item()), float(confidence.item())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", required=True, help="Path to pre-disaster image")
    parser.add_argument("--after", required=True, help="Path to post-disaster image")
    parser.add_argument("--yolo", default=str(RUNS_DIR / "eagle_buildings" / "weights" / "best.pt"))
    parser.add_argument("--siamese", default=str(MODELS_DIR / "siamese_damage.pth"))
    args = parser.parse_args()

    before = cv2.imread(args.before)
    after = cv2.imread(args.after)
    if before is None or after is None:
        raise SystemExit("Could not read before/after image.")
    if before.shape[:2] != after.shape[:2]:
        raise SystemExit("Before and after image dimensions differ. Align/register them first.")

    yolo_path = Path(args.yolo)
    siamese_path = Path(args.siamese)
    if not yolo_path.exists():
        raise SystemExit(f"YOLO weights not found: {yolo_path}")
    if not siamese_path.exists():
        raise SystemExit(f"Siamese weights not found: {siamese_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector = YOLO(str(yolo_path))
    siamese = SiameseDamageClassifier(num_classes=5, pretrained=False).to(device)
    siamese.load_state_dict(torch.load(siamese_path, map_location=device))
    siamese.eval()

    results = detector.predict(source=after, conf=0.25, imgsz=640, verbose=False)
    annotated = after.copy()
    buildings = []

    result = results[0]
    if result.masks is None:
        print("YOLO returned no segmentation masks.")
    else:
        for idx, polygon in enumerate(result.masks.xy):
            points = polygon.astype("int32")
            x, y, w, h = cv2.boundingRect(points)
            h_img, w_img = after.shape[:2]
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(w_img, x + w), min(h_img, y + h)
            if x2 <= x1 or y2 <= y1:
                continue

            before_crop = before[y1:y2, x1:x2]
            after_crop = after[y1:y2, x1:x2]
            class_id, confidence = classify_crop(siamese, before_crop, after_crop, device)
            damage = DAMAGE_NAMES[class_id]

            cv2.polylines(annotated, [points], True, (0, 0, 255), 2)
            cv2.putText(
                annotated, f"{damage} {confidence:.2f}",
                (x1, max(15, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1
            )
            buildings.append({
                "building_index": idx,
                "predicted_damage": damage,
                "confidence": confidence,
                "bbox_xyxy": [x1, y1, x2, y2],
            })

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    image_out = OUTPUTS_DIR / "damage_predictions.png"
    json_out = OUTPUTS_DIR / "damage_predictions.json"
    cv2.imwrite(str(image_out), annotated)

    summary = {name: 0 for name in DAMAGE_NAMES.values()}
    for item in buildings:
        summary[item["predicted_damage"]] += 1

    payload = {
        "before_image": str(args.before),
        "after_image": str(args.after),
        "total_detected_buildings": len(buildings),
        "damage_summary": summary,
        "buildings": buildings,
        "warning": "Predictions are model estimates; verify visually and against ground truth.",
    }
    json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"\nSaved overlay: {image_out}")
    print(f"Saved JSON: {json_out}")

if __name__ == "__main__":
    main()
