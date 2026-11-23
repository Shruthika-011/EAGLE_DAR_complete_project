import csv
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from config import ROOT, DAMAGE_NAMES, MODELS_DIR
from training.siamese_model import SiameseDamageClassifier

class PairDataset(Dataset):
    def __init__(self, csv_path):
        self.csv_path = Path(csv_path)
        with self.csv_path.open("r", encoding="utf-8") as f:
            self.rows = list(csv.DictReader(f))
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        before_path = ROOT / row["before"]
        after_path = ROOT / row["after"]
        before = Image.open(before_path).convert("RGB")
        after = Image.open(after_path).convert("RGB")
        label = int(row["label"])
        return self.transform(before), self.transform(after), label

def main():
    train_csv = ROOT / "processed" / "siamese" / "pairs_train.csv"
    val_csv = ROOT / "processed" / "siamese" / "pairs_val.csv"
    if not train_csv.exists():
        raise SystemExit("Siamese pairs not found. Run python preprocessing/build_siamese_pairs.py")
    train_ds = PairDataset(train_csv)
    if len(train_ds) < 5:
        raise SystemExit("Too few Siamese training pairs. Collect more paired scenes before training.")
    val_ds = PairDataset(val_csv) if val_csv.exists() else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training device:", device)
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=0) if val_ds and len(val_ds) else None

    model = SiameseDamageClassifier(num_classes=5, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    best_val = float("inf")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(10):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for before, after, labels in train_loader:
            before, after = before.to(device), after.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(before, after)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_correct += (logits.argmax(1) == labels).sum().item()
            train_total += labels.size(0)

        avg_train = train_loss / max(len(train_loader), 1)
        train_acc = train_correct / max(train_total, 1)
        print(f"Epoch {epoch+1}/10 | train loss={avg_train:.4f} | train accuracy={train_acc:.3f}")

        if val_loader:
            model.eval()
            val_loss, correct, total = 0.0, 0, 0
            with torch.no_grad():
                for before, after, labels in val_loader:
                    before, after, labels = before.to(device), after.to(device), labels.to(device)
                    logits = model(before, after)
                    val_loss += criterion(logits, labels).item()
                    correct += (logits.argmax(1) == labels).sum().item()
                    total += labels.size(0)
            avg_val = val_loss / max(len(val_loader), 1)
            print(f"             val loss={avg_val:.4f} | val accuracy={correct/max(total,1):.3f}")
            if avg_val < best_val:
                best_val = avg_val
                torch.save(model.state_dict(), MODELS_DIR / "siamese_damage.pth")
        else:
            print("             No independent validation pairs; saving latest checkpoint.")
            torch.save(model.state_dict(), MODELS_DIR / "siamese_damage.pth")

    print("Saved:", MODELS_DIR / "siamese_damage.pth")
    print("Class IDs:", DAMAGE_NAMES)

if __name__ == "__main__":
    main()
