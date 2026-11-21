import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib
import mlflow
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

matplotlib.use("Agg")  # Headless backend safe for Windows CLI and Linux Docker
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.tracking import (
    get_standard_run_tags,
    load_params,
    set_global_seed,
    setup_mlflow,
)

ID_TO_LABEL: Dict[int, str] = {0: "Recycle", 1: "Compost", 2: "Landfill"}
CLASS_NAMES: List[str] = [ID_TO_LABEL[i] for i in range(3)]


def extract_image_features(
    filepath: str,
    image_size: int,
    hsv_bins: Tuple[int, int, int],
) -> np.ndarray:
    """
    Extract a compact classical feature vector (524 dims for 8x8x8 HSV bins):
      - RGB channel mean & std (6 features)
      - Normalized 3D HSV color histogram (512 features)
      - Grayscale Sobel-like finite-difference edge density & stats (6 features)
    """
    with Image.open(filepath) as img:
        rgb_img = img.convert("RGB").resize((image_size, image_size), Image.BILINEAR)
        rgb_arr = np.asarray(rgb_img, dtype=np.float32) / 255.0

        # 1. RGB color statistics
        rgb_mean = rgb_arr.mean(axis=(0, 1))
        rgb_std = rgb_arr.std(axis=(0, 1))

        # 2. 3D HSV color histogram
        hsv_arr = np.asarray(rgb_img.convert("HSV"), dtype=np.float32)
        pixels = hsv_arr.reshape(-1, 3)
        hist, _ = np.histogramdd(
            pixels,
            bins=hsv_bins,
            range=((0, 256), (0, 256), (0, 256)),
        )
        hist = hist.astype(np.float32).flatten()
        hist_sum = hist.sum()
        if hist_sum > 0:
            hist /= hist_sum

        # 3. Grayscale edge density / texture cues
        gray = np.asarray(rgb_img.convert("L"), dtype=np.float32)
        gx = np.abs(np.diff(gray, axis=1))
        gy = np.abs(np.diff(gray, axis=0))
        edge_feats = np.array(
            [
                gx.mean() / 255.0,
                gx.std() / 255.0,
                float((gx > 20.0).mean()),
                gy.mean() / 255.0,
                gy.std() / 255.0,
                float((gy > 20.0).mean()),
            ],
            dtype=np.float32,
        )

    return np.concatenate([rgb_mean, rgb_std, hist, edge_feats]).astype(np.float32)


def build_feature_matrix(
    df: pd.DataFrame,
    image_size: int,
    hsv_bins: Tuple[int, int, int],
    split_name: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract features in parallel for a single split."""
    filepaths = df["filepath"].tolist()
    with ThreadPoolExecutor(max_workers=8) as executor:
        feats = list(
            tqdm(
                executor.map(
                    lambda fp: extract_image_features(fp, image_size, hsv_bins),
                    filepaths,
                ),
                total=len(filepaths),
                desc=f"Extracting features [{split_name}]",
            )
        )
    X = np.vstack(feats)
    y = df["label_id"].to_numpy(dtype=np.int64)
    return X, y


def build_classifier(baseline_cfg: Dict, seed: int):
    """Instantiate the configured classical baseline with balanced class weights."""
    model_type = baseline_cfg["model_type"]
    if model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=int(baseline_cfg["n_estimators"]),
            max_depth=int(baseline_cfg["max_depth"]),
            class_weight="balanced",
            random_state=seed,
            n_jobs=int(baseline_cfg.get("n_jobs", -1)),
        )
    if model_type == "logistic_regression":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=seed,
                        n_jobs=int(baseline_cfg.get("n_jobs", -1)),
                    ),
                ),
            ]
        )
    raise ValueError(f"Unsupported baseline.model_type: {model_type}")


def evaluate_split(y_true: np.ndarray, y_pred: np.ndarray, prefix: str) -> Dict[str, float]:
    """Compute overall and per-class metrics for a split."""
    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    metrics = {
        f"{prefix}_accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        f"{prefix}_macro_f1": round(float(f1_score(y_true, y_pred, average="macro")), 4),
        f"{prefix}_weighted_f1": round(float(f1_score(y_true, y_pred, average="weighted")), 4),
    }
    for cls in CLASS_NAMES:
        cls_key = cls.lower()
        metrics[f"{prefix}_f1_{cls_key}"] = round(float(report[cls]["f1-score"]), 4)
        metrics[f"{prefix}_precision_{cls_key}"] = round(float(report[cls]["precision"]), 4)
        metrics[f"{prefix}_recall_{cls_key}"] = round(float(report[cls]["recall"]), 4)
    return metrics


def save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, output_path: Path) -> None:
    """Plot and save a confusion matrix PNG with raw counts and row percentages."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    cm_norm = cm.astype(np.float64) / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm[i, j]}\n({cm_norm[i, j]:.1%})"

    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(
        cm_norm,
        annot=annot,
        fmt="",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        vmin=0.0,
        vmax=1.0,
        ax=ax,
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Classical Baseline — Test Set Confusion Matrix")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    params = load_params()
    seed = int(params["project"]["seed"])
    set_global_seed(seed)

    split_cfg = params["split"]
    baseline_cfg = params["baseline"]

    splits_csv = Path(split_cfg["splits_csv"])
    if not splits_csv.exists():
        raise FileNotFoundError(f"Splits CSV not found: {splits_csv}. Run split_data.py first.")

    df = pd.read_csv(splits_csv)
    image_size = int(baseline_cfg["image_size"])
    hsv_bins = tuple(int(b) for b in baseline_cfg["hsv_bins"])

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    X_train, y_train = build_feature_matrix(train_df, image_size, hsv_bins, "train")
    X_val, y_val = build_feature_matrix(val_df, image_size, hsv_bins, "val")
    X_test, y_test = build_feature_matrix(test_df, image_size, hsv_bins, "test")

    clf = build_classifier(baseline_cfg, seed)
    print(f"[train_baseline] Fitting {baseline_cfg['model_type']} on {len(X_train):,} samples...")
    clf.fit(X_train, y_train)

    val_preds = clf.predict(X_val)
    test_preds = clf.predict(X_test)

    metrics: Dict[str, float] = {}
    metrics.update(evaluate_split(y_val, val_preds, "val"))
    metrics.update(evaluate_split(y_test, test_preds, "test"))

    # Save model artifact, confusion matrix, and metrics JSON
    model_out = Path(baseline_cfg["model_out"])
    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, model_out)

    cm_path = Path(baseline_cfg["cm_path"])
    save_confusion_matrix(y_test, test_preds, cm_path)

    metrics_path = Path(baseline_cfg["metrics_path"])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Log everything to MLflow
    setup_mlflow(params)
    run_name = f"baseline-{baseline_cfg['model_type']}"
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags(get_standard_run_tags(params, stage="baseline"))
        mlflow.log_params(
            {
                "seed": seed,
                "model_type": baseline_cfg["model_type"],
                "image_size": image_size,
                "hsv_bins": str(hsv_bins),
                "feature_dim": int(X_train.shape[1]),
                "n_estimators": baseline_cfg.get("n_estimators"),
                "max_depth": baseline_cfg.get("max_depth"),
                "class_weight": "balanced",
                "train_samples": len(X_train),
                "val_samples": len(X_val),
                "test_samples": len(X_test),
            }
        )
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(cm_path), artifact_path="plots")
        mlflow.log_artifact(str(metrics_path), artifact_path="metrics")
        mlflow.log_artifact(str(model_out), artifact_path="model")

        print(f"[train_baseline] Logged MLflow run '{run_name}' (run_id={run.info.run_id})")

    print(
        f"[train_baseline] Val Accuracy: {metrics['val_accuracy']:.4f} | "
        f"Val Macro-F1: {metrics['val_macro_f1']:.4f} | "
        f"Test Accuracy: {metrics['test_accuracy']:.4f} | "
        f"Test Macro-F1: {metrics['test_macro_f1']:.4f}"
    )


if __name__ == "__main__":
    main()