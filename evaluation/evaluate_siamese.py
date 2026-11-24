"""Evaluate a trained Siamese damage checkpoint on the validation pairs.

Read-only with respect to models: loads weights, never writes them.

Usage:
    python -m evaluation.evaluate_siamese --checkpoint models/siamese_damage.pth
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import ROOT, DAMAGE_NAMES, OUTPUTS_DIR
from training.siamese_model import SiameseDamageClassifier
from training.train_siamese import PairDataset

NUM_CLASSES = 5
LABELS = list(range(NUM_CLASSES))
NAMES = [DAMAGE_NAMES[i] for i in LABELS]


def class_weights_from_train(train_csv):
    """Same weighting as train_siamese.py so val loss is comparable."""
    labels = PairDataset(train_csv).labels
    counts = torch.bincount(torch.tensor(labels), minlength=NUM_CLASSES).float()
    w = 1.0 / torch.sqrt(counts.clamp(min=1))
    return w / w.mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="models/siamese_damage.pth")
    parser.add_argument("--val-csv", default="processed/siamese/pairs_val.csv")
    parser.add_argument("--train-csv", default="processed/siamese/pairs_train.csv")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--out-dir", default=str(OUTPUTS_DIR / "evaluation"))
    args = parser.parse_args()

    ckpt_path = (ROOT / args.checkpoint).resolve()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = ckpt_path.stem

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SiameseDamageClassifier(num_classes=NUM_CLASSES, pretrained=False)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)  # strict: architecture must match exactly
    model.to(device).eval()

    val_ds = PairDataset(ROOT / args.val_csv)
    loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, pin_memory=(device.type == "cuda"))

    weights = class_weights_from_train(ROOT / args.train_csv).to(device)
    weighted_ce = nn.CrossEntropyLoss(weight=weights, reduction="sum")
    plain_ce = nn.CrossEntropyLoss(reduction="sum")

    y_true, y_pred = [], []
    w_loss_sum = p_loss_sum = 0.0
    w_denominator = 0.0

    with torch.no_grad():
        for before, after, labels in tqdm(loader, desc=f"Eval {tag}", dynamic_ncols=True):
            before, after = before.to(device), after.to(device)
            labels = labels.to(device)
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                logits = model(before, after)
            logits = logits.float()
            w_loss_sum += weighted_ce(logits, labels).item()
            w_denominator += weights[labels].sum().item()
            p_loss_sum += plain_ce(logits, labels).item()
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(logits.argmax(1).cpu().tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    n = len(y_true)

    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    report_txt = classification_report(y_true, y_pred, labels=LABELS, target_names=NAMES,
                                       digits=4, zero_division=0)
    report = classification_report(y_true, y_pred, labels=LABELS, target_names=NAMES,
                                   output_dict=True, zero_division=0)

    metrics = {
        "checkpoint": str(ckpt_path),
        "n_val": int(n),
        # Global class-weighted mean CE. Comparable to, but not bit-identical with,
        # the per-epoch "Validation loss" in train_siamese.py, which averages
        # per-batch weighted means by batch size.
        "val_loss_weighted_ce": w_loss_sum / max(w_denominator, 1e-12),
        "val_loss_unweighted_ce": p_loss_sum / max(n, 1),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, labels=LABELS, average="weighted", zero_division=0),
        "macro_f1_excl_unclassified": f1_score(y_true, y_pred, labels=[0, 1, 2, 3],
                                               average="macro", zero_division=0),
        "macro_f1_damage_only_1_2_3": f1_score(y_true, y_pred, labels=[1, 2, 3],
                                               average="macro", zero_division=0),
        "per_class": {name: report[name] for name in NAMES},
        "confusion_matrix": {"labels": NAMES, "rows_true_cols_pred": cm.tolist()},
        "pred_distribution": {NAMES[i]: int((y_pred == i).sum()) for i in LABELS},
    }

    (out_dir / f"{tag}_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out_dir / f"{tag}_classification_report.txt").write_text(report_txt, encoding="utf-8")

    header = "true\\pred," + ",".join(NAMES)
    rows = [f"{NAMES[i]}," + ",".join(str(v) for v in cm[i]) for i in LABELS]
    (out_dir / f"{tag}_confusion_matrix.csv").write_text("\n".join([header, *rows]) + "\n",
                                                         encoding="utf-8")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(LABELS, NAMES, rotation=35, ha="right")
        ax.set_yticks(LABELS, NAMES)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{tag}: row-normalised confusion matrix")
        for i in LABELS:
            for j in LABELS:
                ax.text(j, i, f"{cm[i, j]}\n{cm_norm[i, j]:.2f}", ha="center", va="center",
                        fontsize=8, color="white" if cm_norm[i, j] > 0.5 else "black")
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        fig.savefig(out_dir / f"{tag}_confusion_matrix.png", dpi=150)
        plt.close(fig)
    except ImportError:
        print("matplotlib not available; skipped confusion matrix PNG.")

    print(f"\nCheckpoint: {ckpt_path}")
    print(f"Val pairs: {n}")
    print(f"Val loss (weighted CE, as in training): {metrics['val_loss_weighted_ce']:.4f}")
    print(f"Val loss (unweighted CE): {metrics['val_loss_unweighted_ce']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1 (5 classes): {metrics['macro_f1']:.4f}")
    print(f"Macro F1 (excl. un-classified): {metrics['macro_f1_excl_unclassified']:.4f}")
    print(f"Weighted F1: {metrics['weighted_f1']:.4f}")
    print("\n" + report_txt)
    print("Confusion matrix (rows = true, cols = predicted):")
    print(header)
    print("\n".join(rows))
    print(f"\nPrediction distribution: {metrics['pred_distribution']}")
    print(f"Saved to: {out_dir}")


if __name__ == "__main__":
    main()
