import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import mlflow
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from src.utils.tracking import (
    get_standard_run_tags,
    load_params,
    set_global_seed,
    setup_mlflow,
)

CLASS_NAMES = ["Recycle", "Compost", "Landfill"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class WasteImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        self.filepaths = df["filepath"].tolist()
        self.labels = df["label_id"].astype(int).tolist()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.filepaths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.filepaths[idx]
        with Image.open(img_path) as img:
            img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, self.labels[idx]


def build_transforms(image_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    train_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return train_tf, eval_tf


def configure_partial_unfreezing(model: nn.Module, unfreeze_blocks: int) -> Dict[str, int]:
    """
    Freeze the entire backbone, then unfreeze:
      - Always: classifier head
      - If unfreeze_blocks > 0: conv_head, norm_head, and the last `unfreeze_blocks` stages of model.blocks
    """
    for param in model.parameters():
        param.requires_grad = False

    # Always unfreeze the 3-class classifier head
    for param in model.classifier.parameters():
        param.requires_grad = True

    if unfreeze_blocks > 0:
        for attr in ("conv_head", "norm_head"):
            if hasattr(model, attr):
                for param in getattr(model, attr).parameters():
                    param.requires_grad = True

        num_blocks = len(model.blocks)
        k = min(unfreeze_blocks, num_blocks)
        for block in model.blocks[num_blocks - k :]:
            for param in block.parameters():
                param.requires_grad = True

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "frozen_params": frozen_params,
    }


def compute_class_weights(train_labels: List[int], num_classes: int = 3) -> torch.Tensor:
    counts = np.bincount(train_labels, minlength=num_classes).astype(np.float64)
    total = counts.sum()
    weights = total / (num_classes * np.maximum(counts, 1.0))
    return torch.tensor(weights, dtype=torch.float32)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer = None,
    desc: str = "Train",
) -> Tuple[float, float, float]:
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    all_preds: List[int] = []
    all_targets: List[int] = []

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for images, targets in tqdm(loader, desc=desc, leave=False):
            images = images.to(device)
            targets = targets.to(device)

            if is_train:
                optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, targets)

            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(targets.cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro")
    return avg_loss, acc, macro_f1


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MobileNetV4 with configurable partial unfreezing.")
    parser.add_argument("--unfreeze-blocks", type=int, default=None, help="Override params.yaml unfreeze_blocks")
    parser.add_argument("--epochs", type=int, default=None, help="Override params.yaml epochs")
    parser.add_argument("--num-workers", type=int, default=None, help="Override params.yaml num_workers")
    parser.add_argument("--run-name", type=str, default=None, help="Custom MLflow run name")
    parser.add_argument("--no-overwrite-best", action="store_true", help="Do not overwrite default checkpoint/metrics")
    args = parser.parse_args()

    params = load_params("params.yaml")
    seed = int(params["project"]["seed"])
    set_global_seed(seed)

    cfg = params["mobilenetv4"]
    model_name = cfg["model_name"]
    image_size = int(cfg["image_size"])
    batch_size = int(cfg["batch_size"])
    epochs = args.epochs if args.epochs is not None else int(cfg["epochs"])
    unfreeze_blocks = args.unfreeze_blocks if args.unfreeze_blocks is not None else int(cfg["unfreeze_blocks"])
    lr_head = float(cfg["lr_head"])
    lr_backbone = float(cfg["lr_backbone"])
    weight_decay = float(cfg["weight_decay"])
    dropout = float(cfg["dropout"])
    num_workers = args.num_workers if args.num_workers is not None else int(cfg["num_workers"])

    checkpoint_path = Path(cfg["checkpoint_path"])
    metrics_path = Path(cfg["metrics_path"])
    if args.no_overwrite_best:
        checkpoint_path = checkpoint_path.with_name(f"mobilenetv4_unfreeze{unfreeze_blocks}.pt")
        metrics_path = metrics_path.with_name(f"mobilenetv4_unfreeze{unfreeze_blocks}_metrics.json")

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_mobilenetv4] Device: {device} | Model: {model_name} | unfreeze_blocks={unfreeze_blocks} | epochs={epochs}")

    splits_df = pd.read_csv(params["split"]["splits_csv"])
    train_df = splits_df[splits_df["split"] == "train"].reset_index(drop=True)
    val_df = splits_df[splits_df["split"] == "val"].reset_index(drop=True)

    train_tf, eval_tf = build_transforms(image_size)
    train_ds = WasteImageDataset(train_df, transform=train_tf)
    val_ds = WasteImageDataset(val_df, transform=eval_tf)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = timm.create_model(model_name, pretrained=True, num_classes=len(CLASS_NAMES), drop_rate=dropout)
    param_stats = configure_partial_unfreezing(model, unfreeze_blocks)
    model = model.to(device)

    trainable_pct = 100.0 * param_stats["trainable_params"] / param_stats["total_params"]
    print(
        f"[train_mobilenetv4] Params -> Total: {param_stats['total_params']:,} | "
        f"Trainable: {param_stats['trainable_params']:,} ({trainable_pct:.1f}%) | "
        f"Frozen: {param_stats['frozen_params']:,}"
    )

    class_weights = compute_class_weights(train_df["label_id"].tolist(), num_classes=len(CLASS_NAMES)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    head_params = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("classifier")]
    backbone_params = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("classifier")]

    param_groups = [{"params": head_params, "lr": lr_head}]
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": lr_backbone})

    optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    setup_mlflow(params)
    default_run_name = "mobilenetv4-head-only" if unfreeze_blocks == 0 else f"mobilenetv4-partial-unfreeze-{unfreeze_blocks}"
    run_name = args.run_name or default_run_name

    best_val_f1 = -1.0
    best_val_acc = 0.0
    best_epoch = 0
    history = []

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags(get_standard_run_tags(params, stage="train_mobilenetv4"))
        mlflow.log_params(
            {
                "model_name": model_name,
                "image_size": image_size,
                "batch_size": batch_size,
                "epochs": epochs,
                "unfreeze_blocks": unfreeze_blocks,
                "lr_head": lr_head,
                "lr_backbone": lr_backbone,
                "weight_decay": weight_decay,
                "dropout": dropout,
                "seed": seed,
                "device": str(device),
                **param_stats,
                "trainable_pct": round(trainable_pct, 2),
            }
        )

        start_time = time.time()
        for epoch in range(1, epochs + 1):
            train_loss, train_acc, train_f1 = run_epoch(
                model, train_loader, criterion, device, optimizer=optimizer, desc=f"Epoch {epoch}/{epochs} [Train]"
            )
            val_loss, val_acc, val_f1 = run_epoch(
                model, val_loader, criterion, device, optimizer=None, desc=f"Epoch {epoch}/{epochs} [Val]"
            )
            scheduler.step()

            mlflow.log_metrics(
                {
                    "train_loss": round(train_loss, 4),
                    "train_accuracy": round(train_acc, 4),
                    "train_macro_f1": round(train_f1, 4),
                    "val_loss": round(val_loss, 4),
                    "val_accuracy": round(val_acc, 4),
                    "val_macro_f1": round(val_f1, 4),
                },
                step=epoch,
            )

            history.append(
                {
                    "epoch": epoch,
                    "train_loss": round(train_loss, 4),
                    "val_loss": round(val_loss, 4),
                    "val_accuracy": round(val_acc, 4),
                    "val_macro_f1": round(val_f1, 4),
                }
            )

            print(
                f"Epoch {epoch:02d}/{epochs:02d} | "
                f"Train Loss: {train_loss:.4f} F1: {train_f1:.4f} | "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} Macro-F1: {val_f1:.4f}"
            )

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_val_acc = val_acc
                best_epoch = epoch
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "model_name": model_name,
                        "num_classes": len(CLASS_NAMES),
                        "class_names": CLASS_NAMES,
                        "image_size": image_size,
                        "dropout": dropout,
                        "unfreeze_blocks": unfreeze_blocks,
                        "best_val_macro_f1": round(best_val_f1, 4),
                        "best_val_accuracy": round(best_val_acc, 4),
                        "best_epoch": best_epoch,
                        "mlflow_run_id": run.info.run_id,
                    },
                    checkpoint_path,
                )

        elapsed_sec = round(time.time() - start_time, 1)
        summary = {
            "run_name": run_name,
            "mlflow_run_id": run.info.run_id,
            "unfreeze_blocks": unfreeze_blocks,
            "best_epoch": best_epoch,
            "best_val_accuracy": round(best_val_acc, 4),
            "best_val_macro_f1": round(best_val_f1, 4),
            "training_time_sec": elapsed_sec,
            "checkpoint_path": str(checkpoint_path),
            "history": history,
        }

        mlflow.log_metrics(
            {
                "best_val_accuracy": summary["best_val_accuracy"],
                "best_val_macro_f1": summary["best_val_macro_f1"],
                "best_epoch": best_epoch,
                "training_time_sec": elapsed_sec,
            }
        )

        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        mlflow.log_artifact(str(metrics_path))
        print(
            f"[train_mobilenetv4] Done in {elapsed_sec}s | Best Epoch: {best_epoch} | "
            f"Best Val Macro-F1: {best_val_f1:.4f} | Saved -> {checkpoint_path}"
        )


if __name__ == "__main__":
    main()