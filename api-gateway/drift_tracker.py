import io
import json
import math
import threading
import time
from collections import deque
from pathlib import Path
from typing import Deque, Dict, Optional, Tuple

import numpy as np
from PIL import Image

from metrics import (
    DRIFT_ALERT_ACTIVE,
    INPUT_BRIGHTNESS_DRIFT_ZSCORE,
    PREDICTION_CONFIDENCE_ROLLING_MEAN,
    PREDICTION_DISTRIBUTION_PSI,
)

CLASSES = ("Recycle", "Compost", "Landfill")


def extract_image_stats(raw_bytes: bytes) -> Tuple[float, Tuple[float, float, float]]:
    """Compute Rec.601 luminance brightness and RGB channel means in [0, 1]."""
    with Image.open(io.BytesIO(raw_bytes)) as img:
        rgb = img.convert("RGB").resize((128, 128), Image.BILINEAR)
        arr = np.asarray(rgb, dtype=np.float32) / 255.0
    lum = float(
        (0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]).mean()
    )
    rgb_mean = tuple(round(float(v), 5) for v in arr.mean(axis=(0, 1)))
    return round(lum, 5), rgb_mean


class DriftTracker:
    """Sliding-window input/prediction drift monitor and JSONL telemetry writer."""

    def __init__(
        self,
        reference_stats_path: str = "/models/reference_stats.json",
        telemetry_path: str = "/reports/inference_telemetry.jsonl",
        window_size: int = 50,
    ) -> None:
        self._lock = threading.RLock()
        self.telemetry_path = Path(telemetry_path)
        self.telemetry_path.parent.mkdir(parents=True, exist_ok=True)

        self.ref_brightness_mean = 0.62
        self.ref_brightness_std = 0.175
        self.ref_conf_mean = 0.98
        self.ref_priors: Dict[str, float] = {c: 1.0 / 3.0 for c in CLASSES}
        self._load_reference(Path(reference_stats_path))

        self._brightness_win: Deque[float] = deque(maxlen=window_size)
        self._conf_win: Deque[float] = deque(maxlen=window_size)
        self._cat_win: Deque[str] = deque(maxlen=window_size)

    def _load_reference(self, path: Path) -> None:
        if not path.exists():
            return
        with path.open("r", encoding="utf-8") as f:
            ref = json.load(f)
        in_stats = ref.get("input_stats", {})
        pred_stats = ref.get("prediction_stats", {})
        self.ref_brightness_mean = float(
            in_stats.get("brightness_mean", self.ref_brightness_mean)
        )
        self.ref_brightness_std = max(
            float(in_stats.get("brightness_std", self.ref_brightness_std)), 1e-4
        )
        self.ref_conf_mean = float(
            pred_stats.get("confidence_mean", self.ref_conf_mean)
        )
        priors = pred_stats.get("class_priors")
        if isinstance(priors, dict) and all(c in priors for c in CLASSES):
            self.ref_priors = {c: max(float(priors[c]), 1e-4) for c in CLASSES}

    def _compute_psi(self) -> float:
        n = len(self._cat_win)
        if n == 0:
            return 0.0
        psi = 0.0
        for cls in CLASSES:
            expected = max(self.ref_priors.get(cls, 1.0 / 3.0), 1e-4)
            actual = max(sum(1 for c in self._cat_win if c == cls) / n, 1e-4)
            psi += (actual - expected) * math.log(actual / expected)
        return round(float(psi), 4)

    def record(
        self,
        request_id: str,
        brightness: float,
        rgb_mean: Tuple[float, float, float],
        tier1_confidence: float,
        final_category: str,
        tier_used: str,
        escalated: bool,
        fallback_triggered: bool,
        total_latency_ms: float,
    ) -> Dict[str, float | bool]:
        with self._lock:
            self._brightness_win.append(brightness)
            self._conf_win.append(tier1_confidence)
            self._cat_win.append(final_category)

            rolling_b = float(np.mean(self._brightness_win))
            b_zscore = round(
                abs(rolling_b - self.ref_brightness_mean) / self.ref_brightness_std, 4
            )
            rolling_conf = round(float(np.mean(self._conf_win)), 4)
            psi = self._compute_psi()

            # Trigger alert once we have at least 10 samples and a clear shift
            alert = len(self._brightness_win) >= 10 and (
                b_zscore >= 1.2 or rolling_conf < 0.78 or psi >= 0.25
            )

            INPUT_BRIGHTNESS_DRIFT_ZSCORE.set(b_zscore)
            PREDICTION_CONFIDENCE_ROLLING_MEAN.set(rolling_conf)
            PREDICTION_DISTRIBUTION_PSI.set(psi)
            DRIFT_ALERT_ACTIVE.set(1.0 if alert else 0.0)

            row = {
                "timestamp": round(time.time(), 3),
                "request_id": request_id,
                "brightness": brightness,
                "rgb_mean": list(rgb_mean),
                "tier1_confidence": tier1_confidence,
                "final_category": final_category,
                "tier_used": tier_used,
                "escalated": escalated,
                "fallback_triggered": fallback_triggered,
                "total_latency_ms": total_latency_ms,
            }
            try:
                with self.telemetry_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")
            except OSError:
                pass

            return {
                "brightness_zscore": b_zscore,
                "rolling_confidence": rolling_conf,
                "prediction_psi": psi,
                "drift_alert_active": alert,
            }
