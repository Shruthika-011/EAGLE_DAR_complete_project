# EAGLE-DAR: YOLO + Siamese damage assessment

## 1. Dataset input

Put your existing data here:

```text
train/
  images/
    sceneA_pre_disaster.png
    sceneA_post_disaster.png
  labels/
    sceneA_post_disaster.json
```

Image/JSON matching for YOLO requires the same filename stem:
`sceneA_post_disaster.png` <-> `sceneA_post_disaster.json`.

For Siamese training, filenames must contain `_pre_disaster` and
`_post_disaster` (or `pre_disaster` / `post_disaster`) and refer to the
same scene. The post-disaster JSON must contain building polygons and
damage subtypes. Both images must cover the same area and be aligned.

If your filenames differ, edit `config.py`'s `PAIR_RULES` or rename files
consistently. The scripts do not invent missing before-images.

## 2. Install

Use Python 3.10 or 3.11. In PowerShell, from this project folder:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, use:
```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

For an NVIDIA GPU, install the PyTorch build compatible with your CUDA
driver from https://pytorch.org/get-started/locally/. The project can run
on CPU, but training will be slower.

## 3. Run in this order

```powershell
python inspect_dataset.py
python preprocessing/prepare_yolo.py
python preprocessing/build_siamese_pairs.py
```

Then train the building segmentation model:

```powershell
python training/train_yolo.py
```

If paired data was found and enough labeled pairs exist, train Siamese:

```powershell
python training/train_siamese.py
```

Run inference:

```powershell
python inference/predict.py --before "train/images/sceneA_pre_disaster.png" --after "train/images/sceneA_post_disaster.png"
```

For a single post-disaster image without a before-image, YOLO can detect
buildings, but the Siamese temporal damage classifier cannot be used.

## 4. Important limitations

- YOLO is trained to segment buildings, not to predict damage categories.
- Siamese training requires actual before/after image pairs and valid
  post-disaster building labels.
- Pair matching assumes the two images are geographically aligned and
  the same pixel coordinates refer to the same location. If not, register
  images before extracting crops.
- A building polygon in the post-disaster annotation is used to crop the
  same location from both images. This requires same-area, aligned images.
- A tiny dataset can overfit. Review the train/validation metrics and
  collect more diverse labeled disaster scenes.
- Damage predictions are decision-support estimates, not proof of
  structural safety or human impact.

## 5. Output folders

- `processed/yolo/`: YOLO segmentation dataset.
- `processed/metadata/buildings.csv`: building labels and coordinates.
- `processed/siamese/crops/`: paired building crops.
- `processed/siamese/pairs_train.csv` and `pairs_val.csv`: Siamese splits.
- `runs/eagle_buildings/weights/best.pt`: trained YOLO model.
- `models/siamese_damage.pth`: trained Siamese model.
- `outputs/`: inference visualizations and JSON results.
