import json
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import timm
import torch
from mlflow.tracking import MlflowClient
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.utils.tracking import (
    get_standard_run_tags,
    load_params,
    set_global_seed,
    setup_mlflow,
)

CLASS_NAMES = ["Recycle", "Compost", "Landfill"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class TestImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform):
        self.filepaths = df["filepath"].tolist()
        self.labels = df["label_id"].astype(int).tolist()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.filepaths)

    def __getitem__(self, idx: int):
        with Image.open(self.filepaths[idx]) as img:
            img = img.convert("RGB")
        return self.transform(img), self.labels[idx]


def save_confusion_matrix(y_true: List[int], y_pred: List[int], out_path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    fig.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(CLASS_NAMES)),
        yticks=np.arange(len(CLASS_NAMES)),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        title="MobileNetV4 (Partial Unfreeze) — Test Confusion Matrix",
        ylabel="True Label",
        xlabel="Predicted Label",
    )

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=11,
                fontweight="bold",
            )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    params = load_params("params.yaml")
    seed = int(params["project"]["seed"])
    set_global_seed(seed)

    cfg = params["mobilenetv4"]
    checkpoint_path = Path(cfg["checkpoint_path"])
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path.resolve()}")

    baseline_metrics_path = Path(params["baseline"]["metrics_path"])
    if not baseline_metrics_path.exists():
        raise FileNotFoundError(f"Baseline metrics not found: {baseline_metrics_path.resolve()}")

    with baseline_metrics_path.open("r", encoding="utf-8") as f:
        baseline_metrics = json.load(f)
    baseline_test_f1 = float(baseline_metrics["test_macro_f1"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = timm.create_model(
        ckpt["model_name"],
        pretrained=False,
        num_classes=int(ckpt["num_classes"]),
        drop_rate=float(ckpt.get("dropout", 0.2)),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    image_size = int(ckpt["image_size"])
    eval_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    splits_df = pd.read_csv(params["split"]["splits_csv"])
    test_df = splits_df[splits_df["split"] == "test"].reset_index(drop=True)
    test_ds = TestImageDataset(test_df, transform=eval_tf)
    test_loader = DataLoader(test_ds, batch_size=int(cfg["batch_size"]), shuffle=False, num_workers=0)

    y_true: List[int] = []
    y_pred: List[int] = []
    with torch.no_grad():
        for images, targets in test_loader:
            logits = model(images.to(device))
            preds = torch.argmax(logits, dim=1)
            y_pred.extend(preds.cpu().tolist())
            y_true.extend(targets.tolist())

    test_acc = accuracy_score(y_true, y_pred)
    test_macro_f1 = f1_score(y_true, y_pred, average="macro")
    test_weighted_f1 = f1_score(y_true, y_pred, average="weighted")
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True)

    metrics: Dict[str, float] = {
        "test_accuracy": round(test_acc, 4),
        "test_macro_f1": round(test_macro_f1, 4),
        "test_weighted_f1": round(test_weighted_f1, 4),
        "baseline_test_macro_f1": round(baseline_test_f1, 4),
        "f1_gain_over_baseline": round(test_macro_f1 - baseline_test_f1, 4),
    }
    for cls_name in CLASS_NAMES:
        cls_key = cls_name.lower()
        metrics[f"test_f1_{cls_key}"] = round(report[cls_name]["f1-score"], 4)
        metrics[f"test_precision_{cls_key}"] = round(report[cls_name]["precision"], 4)
        metrics[f"test_recall_{cls_key}"] = round(report[cls_name]["recall"], 4)

    cm_path = Path("reports/mobilenetv4_confusion_matrix.png")
    save_confusion_matrix(y_true, y_pred, cm_path)

    promoted = test_macro_f1 > baseline_test_f1
    registered_model_name = params["mlflow"]["registered_model_name"]
    registered_version = None

    setup_mlflow(params)
    with mlflow.start_run(run_name="evaluate-register-mobilenetv4") as run:
        mlflow.set_tags(
            {
                **get_standard_run_tags(params, stage="evaluate_register"),
                "promotion_gate_passed": str(promoted),
                "source_train_run_id": str(ckpt.get("mlflow_run_id", "unknown")),
            }
        )
        mlflow.log_params(
            {
                "model_name": ckpt["model_name"],
                "unfreeze_blocks": ckpt.get("unfreeze_blocks"),
                "best_epoch": ckpt.get("best_epoch"),
                "test_samples": len(test_df),
            }
        )
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(cm_path))

        if promoted:
            model_info = mlflow.pytorch.log_model(
                pytorch_model=model.cpu(),
                name="model",
                serialization_format="pickle",
                registered_model_name=registered_model_name,
            )
            client = MlflowClient()
            versions = client.search_model_versions(f"name='{registered_model_name}'")
            latest = max(versions, key=lambda v: int(v.version))
            registered_version = latest.version
            client.set_registered_model_alias(registered_model_name, "champion", registered_version)
            print(
                f"[evaluate_register] PROMOTION GATE PASSED "
                f"({test_macro_f1:.4f} > {baseline_test_f1:.4f}) -> "
                f"Registered '{registered_model_name}' v{registered_version} with alias @champion"
            )
        else:
            print(
                f"[evaluate_register] PROMOTION GATE FAILED "
                f"({test_macro_f1:.4f} <= {baseline_test_f1:.4f}) -> Model NOT registered."
            )

        summary = {
            "mlflow_run_id": run.info.run_id,
            "promoted": promoted,
            "registered_model_name": registered_model_name if promoted else None,
            "registered_version": registered_version,
            "alias": "champion" if promoted else None,
            "metrics": metrics,
        }
        eval_out = Path("reports/evaluation_summary.json")
        with eval_out.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        mlflow.log_artifact(str(eval_out))

    print(
        f"[evaluate_register] Test Accuracy: {test_acc:.4f} | "
        f"Test Macro-F1: {test_macro_f1:.4f} (Baseline: {baseline_test_f1:.4f})"
    )


if __name__ == "__main__":
    main()