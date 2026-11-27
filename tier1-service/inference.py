import base64
import io
import threading
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import onnxruntime as ort
from PIL import Image

ID_TO_LABEL: Dict[int, str] = {0: "Recycle", 1: "Compost", 2: "Landfill"}
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exps = np.exp(shifted)
    return exps / np.sum(exps)


def _apply_thermal_colormap(cam_01: np.ndarray) -> np.ndarray:
    """
    Pure-NumPy thermal colormap (blue -> cyan -> green -> yellow -> red)
    avoiding matplotlib/OpenCV container bloat.
    Input: (H, W) float32 in [0, 1]. Output: (H, W, 3) uint8.
    """
    r = np.clip(1.5 - np.abs(4.0 * cam_01 - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * cam_01 - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * cam_01 - 1.0), 0.0, 1.0)
    return (np.stack([r, g, b], axis=-1) * 255.0).astype(np.uint8)


def _render_cam_overlay_b64(
    orig_rgb: Image.Image, cam_2d: np.ndarray, alpha: float = 0.45
) -> str:
    """Upsample 7x7 CAM to 224x224, blend over original image, and return data-URI PNG."""
    cam = np.maximum(cam_2d, 0.0)
    cam_max = float(cam.max())
    if cam_max > 1e-8:
        cam = cam / cam_max

    cam_pil = Image.fromarray((cam * 255.0).astype(np.uint8), mode="L").resize(
        orig_rgb.size, Image.BILINEAR
    )
    cam_norm = np.asarray(cam_pil, dtype=np.float32) / 255.0
    heatmap_rgb = _apply_thermal_colormap(cam_norm)

    base_arr = np.asarray(orig_rgb, dtype=np.float32)
    blended = np.clip(
        (1.0 - alpha) * base_arr + alpha * heatmap_rgb.astype(np.float32), 0, 255
    )
    out_img = Image.fromarray(blended.astype(np.uint8), mode="RGB")

    buf = io.BytesIO()
    out_img.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


class Tier1Engine:
    """Thread-safe ONNX Runtime manager for FP32 primary and INT8 shadow models."""

    def __init__(self, fp32_path: str, int8_path: str, image_size: int = 224) -> None:
        self.fp32_path = Path(fp32_path)
        self.int8_path = Path(int8_path)
        self.image_size = image_size
        self._lock = threading.RLock()
        self.fp32_sess: Optional[ort.InferenceSession] = None
        self.int8_sess: Optional[ort.InferenceSession] = None
        self.model_version_ts: float = 0.0
        self.reload()

    def reload(self) -> Dict[str, str]:
        """Hot-reload FP32 and INT8 ONNX artifacts from disk without container restart."""
        if not self.fp32_path.exists():
            raise FileNotFoundError(f"FP32 ONNX model not found: {self.fp32_path}")

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        new_fp32 = ort.InferenceSession(
            str(self.fp32_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        new_int8 = (
            ort.InferenceSession(
                str(self.int8_path),
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
            if self.int8_path.exists()
            else None
        )

        with self._lock:
            self.fp32_sess = new_fp32
            self.int8_sess = new_int8
            self.model_version_ts = time.time()

        return {
            "status": "reloaded",
            "fp32_model": str(self.fp32_path),
            "int8_loaded": str(self.int8_sess is not None),
        }

    def _preprocess(self, raw_bytes: bytes) -> Tuple[Image.Image, np.ndarray]:
        img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        resized = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        arr = np.asarray(resized, dtype=np.float32) / 255.0
        norm = (arr - IMAGENET_MEAN) / IMAGENET_STD
        tensor = np.transpose(norm, (2, 0, 1))[None, ...].astype(np.float32)
        return resized, tensor

    def predict(
        self, raw_bytes: bytes, explain: bool = True, shadow: bool = False
    ) -> Dict:
        t0 = time.perf_counter()
        resized, tensor = self._preprocess(raw_bytes)

        with self._lock:
            sess = self.int8_sess if shadow else self.fp32_sess
            if sess is None:
                raise RuntimeError("Requested ONNX session is not loaded.")
            outputs = sess.run(None, {"input": tensor})

        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        logits = outputs[0][0]
        probs = _softmax(logits)
        pred_idx = int(np.argmax(probs))
        category = ID_TO_LABEL[pred_idx]
        confidence = round(float(probs[pred_idx]), 4)

        result = {
            "category": category,
            "confidence": confidence,
            "probabilities": {
                ID_TO_LABEL[i]: round(float(probs[i]), 4) for i in range(3)
            },
            "latency_ms": latency_ms,
            "model_variant": (
                "mobilenetv4-int8-onnx" if shadow else "mobilenetv4-fp32-onnx"
            ),
        }

        if explain and not shadow and len(outputs) > 1:
            cam_map = outputs[1][0, pred_idx]  # (7, 7)
            result["explanation"] = {
                "method": "Spatial Class Activation Map (CAM)",
                "saliency_overlay_base64": _render_cam_overlay_b64(resized, cam_map),
            }

        return result
