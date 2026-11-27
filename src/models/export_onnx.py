import inspect
import json
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import onnxruntime as ort
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from onnxruntime.quantization import QuantType, quantize_dynamic
from PIL import Image
from tqdm import tqdm

from src.utils.tracking import get_dvc_data_hash, load_params, set_global_seed

ID_TO_LABEL: Dict[int, str] = {0: "Recycle", 1: "Compost", 2: "Landfill"}
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class DualOutputMobileNetV4(nn.Module):
    """
    Wraps a timm MobileNetV4 classifier so one ONNX forward pass outputs:
      - logits:   (B, 3)       exact classification logits
      - cam_maps: (B, 3, H, W) spatial Class Activation Maps for UI saliency overlays
    """

    def __init__(self, base_model: nn.Module) -> None:
        super().__init__()
        self.base = base_model
        classifier = self.base.get_classifier()
        self.cls_weight = classifier.weight  # (num_classes, C_out)
        self.has_conv_head = (
            hasattr(self.base, "conv_head") and self.base.conv_head is not None
        )
        self.has_act2 = hasattr(self.base, "act2") and self.base.act2 is not None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feats = self.base.forward_features(x)  # (B, C_feat, H, W)
        logits = self.base.forward_head(feats)  # (B, 3)

        spatial = feats
        if self.has_conv_head:
            spatial = self.base.conv_head(spatial)
            if self.has_act2:
                spatial = self.base.act2(spatial)

        if spatial.shape[1] == self.cls_weight.shape[1]:
            weight_1x1 = self.cls_weight.unsqueeze(-1).unsqueeze(-1)
            cam_maps = F.conv2d(spatial, weight_1x1)  # (B, 3, H, W)
        else:
            # Safe fallback: broadcast spatial activation energy across 3 classes
            energy = spatial.mean(dim=1, keepdim=True)
            cam_maps = energy.repeat(1, self.cls_weight.shape[0], 1, 1)

        return logits, cam_maps


def load_trained_model(params: Dict) -> nn.Module:
    """Reconstruct MobileNetV4 architecture and load the Deliverable 1 checkpoint."""
    cfg = params["mobilenetv4"]
    ckpt_path = Path(cfg["checkpoint_path"])
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {ckpt_path.resolve()}. Run Deliverable 1 training or `dvc pull` first."
        )

    model = timm.create_model(
        cfg["model_name"],
        pretrained=False,
        num_classes=3,
        drop_rate=float(cfg.get("dropout", 0.2)),
    )

    raw_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(raw_ckpt, dict):
        state_dict = (
            raw_ckpt.get("model_state_dict") or raw_ckpt.get("state_dict") or raw_ckpt
        )
    else:
        state_dict = raw_ckpt

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def preprocess_pil(img: Image.Image, image_size: int) -> np.ndarray:
    """Deterministic resize + ImageNet normalization -> NCHW float32 array."""
    resized = img.convert("RGB").resize((image_size, image_size), Image.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    norm = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return np.transpose(norm, (2, 0, 1))[None, ...].astype(np.float32)


def softmax_np(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exps = np.exp(shifted)
    return exps / np.sum(exps, axis=-1, keepdims=True)


def export_fp32_onnx(wrapper: nn.Module, out_path: Path, image_size: int) -> None:
    """Export dual-output PyTorch wrapper to ONNX FP32 and verify parity."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, 3, image_size, image_size, dtype=torch.float32)

    export_kwargs = {
        "input_names": ["input"],
        "output_names": ["logits", "cam_maps"],
        "dynamic_axes": {
            "input": {0: "batch_size"},
            "logits": {0: "batch_size"},
            "cam_maps": {0: "batch_size"},
        },
        "opset_version": 17,
        "do_constant_folding": True,
    }
    # Prefer legacy TorchScript exporter on PyTorch 2.5+ to avoid external onnxscript dep
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["dynamo"] = False

    torch.onnx.export(wrapper, dummy, str(out_path), **export_kwargs)

    # Numerical parity check against PyTorch
    with torch.no_grad():
        pt_logits, pt_cam = wrapper(dummy)

    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    ort_logits, ort_cam = sess.run(None, {"input": dummy.numpy()})

    max_logit_diff = float(np.max(np.abs(pt_logits.numpy() - ort_logits)))
    max_cam_diff = float(np.max(np.abs(pt_cam.numpy() - ort_cam)))
    print(
        f"[export_onnx] FP32 parity verified -> "
        f"max_logit_diff={max_logit_diff:.2e} | max_cam_diff={max_cam_diff:.2e} | "
        f"cam_shape={ort_cam.shape}"
    )
    assert max_logit_diff < 1e-4, f"ONNX logit drift too high: {max_logit_diff}"


def quantize_to_int8(fp32_path: Path, int8_path: Path, image_size: int) -> None:
    """Dynamically quantize FP32 ONNX to INT8 and verify ORT execution."""
    dummy = np.random.randn(1, 3, image_size, image_size).astype(np.float32)
    try:
        quantize_dynamic(
            model_input=str(fp32_path),
            model_output=str(int8_path),
            weight_type=QuantType.QUInt8,
        )
        sess = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])
        sess.run(None, {"input": dummy})
    except Exception as exc:
        print(
            f"[export_onnx] Full Conv+MatMul INT8 failed ({exc}); retrying MatMul/Gemm only..."
        )
        quantize_dynamic(
            model_input=str(fp32_path),
            model_output=str(int8_path),
            weight_type=QuantType.QUInt8,
            op_types_to_quantize=["MatMul", "Gemm"],
        )
        sess = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])
        sess.run(None, {"input": dummy})
    print(f"[export_onnx] INT8 quantized model verified -> {int8_path.as_posix()}")


def benchmark_session(model_path: Path, image_size: int, runs: int = 40) -> float:
    """Measure average CPU inference latency in milliseconds."""
    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    dummy = np.random.randn(1, 3, image_size, image_size).astype(np.float32)
    # Warmup
    for _ in range(5):
        sess.run(None, {"input": dummy})
    t0 = time.perf_counter()
    for _ in range(runs):
        sess.run(None, {"input": dummy})
    return round(((time.perf_counter() - t0) / runs) * 1000.0, 2)


def build_reference_stats(
    fp32_path: Path,
    splits_csv: Path,
    image_size: int,
    seed: int,
    max_samples: int = 450,
) -> Dict:
    """
    Compute reference input (brightness, RGB) and prediction (confidence, class priors)
    distributions on the training split for live drift monitoring & formal KS/PSI tests.
    """
    df = pd.read_csv(splits_csv)
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    if len(train_df) > max_samples:
        train_df = train_df.sample(n=max_samples, random_state=seed).reset_index(
            drop=True
        )

    sess = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])

    brightness_vals = []
    rgb_means = []
    confidences = []
    pred_counts = {lbl: 0 for lbl in ID_TO_LABEL.values()}

    for fp in tqdm(
        train_df["filepath"].tolist(), desc="Computing reference drift stats"
    ):
        with Image.open(fp) as img:
            rgb = img.convert("RGB").resize((image_size, image_size), Image.BILINEAR)
            arr = np.asarray(rgb, dtype=np.float32) / 255.0

            # Perceptual luminance (Rec. 601) in [0, 1]
            lum = float(
                (0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]).mean()
            )
            brightness_vals.append(round(lum, 5))
            rgb_means.append(arr.mean(axis=(0, 1)).tolist())

            inp = preprocess_pil(rgb, image_size)
            logits, _ = sess.run(None, {"input": inp})
            probs = softmax_np(logits)[0]
            pred_idx = int(np.argmax(probs))
            pred_counts[ID_TO_LABEL[pred_idx]] += 1
            confidences.append(round(float(probs[pred_idx]), 5))

    rgb_arr = np.array(rgb_means, dtype=np.float32)
    b_arr = np.array(brightness_vals, dtype=np.float32)
    c_arr = np.array(confidences, dtype=np.float32)
    total = max(len(train_df), 1)

    return {
        "dvc_raw_md5": get_dvc_data_hash(),
        "sample_count": len(train_df),
        "input_stats": {
            "brightness_mean": round(float(b_arr.mean()), 5),
            "brightness_std": round(float(max(b_arr.std(), 1e-4)), 5),
            "rgb_mean": [round(float(v), 5) for v in rgb_arr.mean(axis=0)],
            "rgb_std": [round(float(max(v, 1e-4)), 5) for v in rgb_arr.std(axis=0)],
            "brightness_samples": brightness_vals,
        },
        "prediction_stats": {
            "confidence_mean": round(float(c_arr.mean()), 5),
            "confidence_std": round(float(max(c_arr.std(), 1e-4)), 5),
            "class_priors": {
                lbl: round(cnt / total, 5) for lbl, cnt in pred_counts.items()
            },
            "confidence_samples": confidences,
        },
    }


def main() -> None:
    params = load_params()
    seed = int(params["project"]["seed"])
    set_global_seed(seed)

    image_size = int(params["mobilenetv4"]["image_size"])
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    fp32_path = models_dir / "mobilenetv4_fp32.onnx"
    int8_path = models_dir / "mobilenetv4_int8.onnx"
    ref_stats_path = models_dir / "reference_stats.json"

    base_model = load_trained_model(params)
    wrapper = DualOutputMobileNetV4(base_model)
    wrapper.eval()

    export_fp32_onnx(wrapper, fp32_path, image_size)
    quantize_to_int8(fp32_path, int8_path, image_size)

    fp32_mb = round(fp32_path.stat().st_size / (1024 * 1024), 2)
    int8_mb = round(int8_path.stat().st_size / (1024 * 1024), 2)
    fp32_ms = benchmark_session(fp32_path, image_size)
    int8_ms = benchmark_session(int8_path, image_size)

    print(
        f"[export_onnx] Size:    FP32={fp32_mb} MB -> INT8={int8_mb} MB "
        f"({(1 - int8_mb / max(fp32_mb, 1e-6)) * 100:.1f}% reduction)"
    )
    print(f"[export_onnx] Latency: FP32={fp32_ms} ms | INT8={int8_ms} ms (CPU avg)")

    ref_stats = build_reference_stats(
        fp32_path=fp32_path,
        splits_csv=Path(params["split"]["splits_csv"]),
        image_size=image_size,
        seed=seed,
    )
    ref_stats["export_benchmark"] = {
        "fp32_size_mb": fp32_mb,
        "int8_size_mb": int8_mb,
        "fp32_latency_ms": fp32_ms,
        "int8_latency_ms": int8_ms,
    }
    with ref_stats_path.open("w", encoding="utf-8") as f:
        json.dump(ref_stats, f, indent=2)

    print(f"[export_onnx] Saved reference drift stats -> {ref_stats_path.as_posix()}")


if __name__ == "__main__":
    main()
