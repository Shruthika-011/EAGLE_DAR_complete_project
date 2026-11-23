
import csv
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from config import ROOT, DAMAGE_NAMES, MODELS_DIR
from training.siamese_model import SiameseDamageClassifier


BATCH_SIZE = 4
EPOCHS = 5
LEARNING_RATE = 1e-4


class PairDataset(Dataset):
    def __init__(self, csv_path):
        self.csv_path = Path(csv_path)

        with self.csv_path.open("r", encoding="utf-8") as f:
            self.rows = list(csv.DictReader(f))

        self.labels = [int(row["label"]) for row in self.rows]

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

        with Image.open(before_path) as img:
            before = self.transform(img.convert("RGB"))

        with Image.open(after_path) as img:
            after = self.transform(img.convert("RGB"))

        label = int(row["label"])

        return before, after, label


def main():
    train_csv = ROOT / "processed" / "siamese" / "pairs_train.csv"
    val_csv = ROOT / "processed" / "siamese" / "pairs_val.csv"

    if not train_csv.exists():
        raise SystemExit("Training pairs not found.")

    train_ds = PairDataset(train_csv)
    val_ds = PairDataset(val_csv) if val_csv.exists() else None

    if len(train_ds) < 5:
        raise SystemExit("Too few training pairs.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nTraining device: {device}", flush=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    val_loader = None

    if val_ds and len(val_ds):
        val_loader = DataLoader(
            val_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=(device.type == "cuda"),
        )

    # Moderate class weighting to reduce the effect of imbalance.
    class_counts = torch.bincount(
        torch.tensor(train_ds.labels),
        minlength=5
    ).float()

    class_weights = 1.0 / torch.sqrt(class_counts.clamp(min=1))
    class_weights = class_weights / class_weights.mean()

    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device)
    )

    print("\nTraining class distribution:", flush=True)

    for class_id in range(5):
        print(
            f"{class_id} - {DAMAGE_NAMES[class_id]}: "
            f"{int(class_counts[class_id])}",
            flush=True
        )

    model = SiameseDamageClassifier(
        num_classes=5,
        pretrained=True
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(device.type == "cuda")
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")

    for epoch in range(EPOCHS):

        print(f"\n========== EPOCH {epoch + 1}/{EPOCHS} ==========",
              flush=True)

        model.train()

        train_loss = 0.0
        train_correct = 0
        train_total = 0

        progress = tqdm(
            train_loader,
            total=len(train_loader),
            desc=f"Training {epoch + 1}/{EPOCHS}",
            dynamic_ncols=True
        )

        for before, after, labels in progress:

            before = before.to(device, non_blocking=True)
            after = after.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                device_type=device.type,
                enabled=(device.type == "cuda")
            ):
                logits = model(before, after)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_size = labels.size(0)

            train_loss += loss.item() * batch_size
            train_correct += (
                logits.argmax(1) == labels
            ).sum().item()

            train_total += batch_size

            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                accuracy=f"{train_correct / max(train_total, 1):.3f}"
            )

        avg_train = train_loss / max(train_total, 1)
        train_acc = train_correct / max(train_total, 1)

        print(
            f"\nTrain loss: {avg_train:.4f} | "
            f"Train accuracy: {train_acc:.4f}",
            flush=True
        )

        avg_val = None

        if val_loader:

            model.eval()

            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():

                for before, after, labels in tqdm(
                    val_loader,
                    desc="Validation",
                    dynamic_ncols=True
                ):

                    before = before.to(device, non_blocking=True)
                    after = after.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)

                    with torch.amp.autocast(
                        device_type=device.type,
                        enabled=(device.type == "cuda")
                    ):
                        logits = model(before, after)
                        loss = criterion(logits, labels)

                    batch_size = labels.size(0)

                    val_loss += loss.item() * batch_size
                    val_correct += (
                        logits.argmax(1) == labels
                    ).sum().item()

                    val_total += batch_size

            avg_val = val_loss / max(val_total, 1)
            val_acc = val_correct / max(val_total, 1)

            print(
                f"Validation loss: {avg_val:.4f} | "
                f"Validation accuracy: {val_acc:.4f}",
                flush=True
            )

            if avg_val < best_val:
                best_val = avg_val

                torch.save(
                    model.state_dict(),
                    MODELS_DIR / "siamese_damage.pth"
                )

                print("Best model checkpoint saved.", flush=True)

        # Always save the latest completed epoch.
        torch.save(
            model.state_dict(),
            MODELS_DIR / "siamese_damage_last.pth"
        )

        print(
            f"Epoch {epoch + 1} completed. Latest checkpoint saved.",
            flush=True
        )

    print("\nTraining completed.", flush=True)
    print("Best model:", MODELS_DIR / "siamese_damage.pth")
    print("Latest model:", MODELS_DIR / "siamese_damage_last.pth")
    print("Class IDs:", DAMAGE_NAMES)


if __name__ == "__main__":
    main()
