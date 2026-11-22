# EcoSort — Adaptive Multi-Tier Inference Gateway for Automated Waste Classification

**Course:** MLOps | **Instructor:** Sir Usman Anwer  
**Team:** Muhammad Basim Irfan (`23L-0846`) & Umer Karamat (`23L-0873`) — BCS-7B  
**Deliverable:** Deliverable 1 (Core ML Pipeline: Versioned, Leak-Free, Tracked & Reproducible)

---

## 1. Pipeline Architecture

EcoSort classifies waste imagery into three operational categories (**`Recycle`**, **`Compost`**, **`Landfill`**) using a 6-stage DVC pipeline orchestrated end-to-end via `dvc.yaml`:

1. **`prepare`** (`src/data/prepare_dataset.py`): Maps the 12-class Garbage Classification dataset into 3 EcoSort classes with deterministic stratified sampling (`seed=42`).
2. **`validate`** (`src/data/validate_data.py`): Automated quality gate verifying file integrity, `PIL.Image.verify()`, RGB compatibility, minimum resolution (≥ 64×64), and class counts before splitting. Outputs `reports/validation_report.json`.
3. **`split`** (`src/data/split_data.py`): Removes exact MD5 duplicates and clusters near-duplicates via Perceptual Hashing (`pHash`, Hamming distance ≤ 6) to enforce leakage-free, group-disjoint 70/15/15 splits with programmatic zero-overlap assertions.
4. **`train_baseline`** (`src/models/train_baseline.py`): Classical Random Forest baseline trained on 3D HSV color histograms + RGB stats + edge density, logged to MLflow before deep learning.
5. **`train_mobilenetv4`** (`src/models/train_mobilenetv4.py`): Custom edge model adaptation using `timm` (`mobilenetv4_conv_small.e2400_r224_in1k`) with configurable **partial unfreezing**, train-only augmentations, class-weighted `CrossEntropyLoss`, and differential learning rates.
6. **`evaluate_register`** (`src/models/evaluate_register.py`): Single-pass evaluation on the held-out `test` split with an automated promotion gate that registers **`EcoSort-Tier1-MobileNetV4`** (`@champion`) in MLflow Model Registry only if it beats the classical baseline.

---

## 2. Deliverable 1 Results Summary

| Experiment / Run | Trainable Params | Val Macro-F1 | Test Accuracy | Test Macro-F1 | Promotion Gate |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `baseline-random_forest` | Classical (HSV+RGB) | 0.8319 | 0.8089 | 0.8083 | Baseline |
| `mobilenetv4-head-only` (`unfreeze_blocks=0`) | 3,843 (0.2%) | 0.5717 | — | — | Not Promoted |
| **`mobilenetv4-partial-unfreeze-1`** (`unfreeze_blocks=1`) | **1,360,003 (54.5%)** | **0.9244** | **0.8800** | **0.8807** | **PASSED (+0.0724)** |

* **Registered Model:** `EcoSort-Tier1-MobileNetV4` (`@champion`)
* **MLflow Evidence:** See `docs/deliverable1/` (UI screenshots) and `reports/mlflow_runs_export.csv`.

---

## 3. Repository Structure

```text
EcoSort/
├── .dvc/                       # DVC configuration (AWS S3 remote)
├── data/
│   ├── raw.dvc                 # DVC pointer for versioned raw dataset
│   └── processed/              # Validated manifests, splits, baseline & MobileNetV4 checkpoints
├── docs/deliverable1/          # MLflow UI screenshots (experiments, run details, model registry)
├── reports/                    # Validation report, split summary, metrics JSONs, confusion matrices, MLflow CSV export
├── src/
│   ├── data/                   # prepare_dataset.py, validate_data.py, split_data.py
│   ├── models/                 # train_baseline.py, train_mobilenetv4.py, evaluate_register.py
│   └── utils/                  # tracking.py (seeding, DVC MD5 tagging, MLflow setup)
├── params.yaml                 # Single source of truth for all hyperparameters
├── dvc.yaml                    # 6-stage reproducible DVC DAG
├── dvc.lock                    # Locked exact hashes of data, code, and outputs
├── requirements-ml.txt         # Pinned Python 3.12 ML dependencies
├── Dockerfile.train            # Containerized training environment
└── reproduce.sh                # One-command reproduction entrypoint