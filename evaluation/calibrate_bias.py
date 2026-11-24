"""Can per-class logit offsets (no retraining, weights untouched) make rare damage classes predictable?

Collects the Siamese logits on the held-out validation pairs once, then 2-fold cross-validates BY SCENE: offsets are fitted on
one half of the scenes to maximise macro F1 and scored on the other half, which the fit never saw. Nothing is written to the
model or the platform; this only measures whether the idea is worth adopting.

    python -m evaluation.calibrate_bias
"""
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, recall_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import ROOT, MODELS_DIR
from training.siamese_model import SiameseDamageClassifier
from training.train_siamese import PairDataset

NAMES = ["no-damage", "minor-damage", "major-damage", "destroyed", "un-classified"]
OUT = ROOT / "outputs" / "evaluation"


def collect_logits():
    cache = OUT / "val_logits.npz"
    if cache.exists():
        z = np.load(cache)
        return z["logits"], z["labels"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SiameseDamageClassifier(num_classes=5, pretrained=False)
    model.load_state_dict(torch.load(MODELS_DIR / "siamese_damage.pth", map_location="cpu", weights_only=True))
    model.to(device).eval()
    ds = PairDataset(ROOT / "processed/siamese/pairs_val.csv")
    L, Y = [], []
    with torch.no_grad():
        for b, a, y in tqdm(DataLoader(ds, batch_size=32, num_workers=0), desc="val logits"):
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                L.append(model(b.to(device), a.to(device)).float().cpu().numpy())
            Y.append(y.numpy())
    logits, labels = np.concatenate(L), np.concatenate(Y)
    np.savez(cache, logits=logits, labels=labels)
    return logits, labels


def macro_f1(logits, labels, off):
    return f1_score(labels, (logits + off).argmax(1), labels=range(5), average="macro", zero_division=0)


def fit_offsets(logits, labels, grid=np.arange(-3.0, 8.01, 0.25), passes=3):
    off = np.zeros(5)
    best = macro_f1(logits, labels, off)
    for _ in range(passes):
        for c in range(1, 5):                       # class 0 is the reference
            for g in grid:
                trial = off.copy()
                trial[c] = g
                s = macro_f1(logits, labels, trial)
                if s > best + 1e-9:
                    best, off = s, trial
    return off, best


def report(logits, labels, off):
    pred = (logits + off).argmax(1)
    rec = recall_score(labels, pred, labels=range(5), average=None, zero_division=0)
    return {"macro_f1": float(f1_score(labels, pred, labels=range(5), average="macro", zero_division=0)),
            "accuracy": float((pred == labels).mean()), "recall": {n: round(float(r), 3) for n, r in zip(NAMES, rec)},
            "predicted": {n: int((pred == i).sum()) for i, n in enumerate(NAMES)}}


def main():
    logits, labels = collect_logits()
    import csv
    with open(ROOT / "processed/siamese/pairs_val.csv", encoding="utf-8") as f:
        scenes = np.array([r["scene"] for r in csv.DictReader(f)])
    assert len(scenes) == len(labels)
    uniq = sorted(set(scenes))
    random.Random(0).shuffle(uniq)
    halves = [set(uniq[::2]), set(uniq[1::2])]
    folds = []
    for k in range(2):
        fit_mask = np.isin(scenes, list(halves[k]))
        test_mask = ~fit_mask
        off, fit_score = fit_offsets(logits[fit_mask], labels[fit_mask])
        folds.append({"offsets": [round(float(x), 2) for x in off], "fit_macro_f1": round(float(fit_score), 4),
                      "heldout_before": report(logits[test_mask], labels[test_mask], np.zeros(5)),
                      "heldout_after": report(logits[test_mask], labels[test_mask], off)})
    full_off, full_score = fit_offsets(logits, labels)
    result = {"note": "offsets fitted on half of the validation scenes and scored on the other half (2-fold by scene); weights untouched",
              "n_val": int(len(labels)), "folds": folds,
              "offsets_fitted_on_all_validation": [round(float(x), 2) for x in full_off], "all_validation_fit_macro_f1_OPTIMISTIC": round(float(full_score), 4)}
    (OUT / "bias_calibration.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    for i, f in enumerate(folds):
        b, a = f["heldout_before"], f["heldout_after"]
        print(f"fold {i}: offsets {f['offsets']}  held-out macro F1 {b['macro_f1']:.3f} -> {a['macro_f1']:.3f}   accuracy {b['accuracy']:.3f} -> {a['accuracy']:.3f}")
        print(f"   recall before {b['recall']}\n   recall after  {a['recall']}\n   predicted after {a['predicted']}")
    print("saved", OUT / "bias_calibration.json")


if __name__ == "__main__":
    main()
