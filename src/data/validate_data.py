import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
from PIL import Image, ImageFile
from tqdm import tqdm

from src.utils.tracking import get_dvc_data_hash, load_params

# Strictly reject truncated images instead of silently loading partial pixels
ImageFile.LOAD_TRUNCATED_IMAGES = False

VALID_LABELS = {"Recycle", "Compost", "Landfill"}


def validate_single_image(
    row: Dict[str, Any],
    allowed_exts: set,
    min_w: int,
    min_h: int,
) -> Tuple[bool, Dict[str, Any]]:
    """Validate a single image record for existence, extension, decodability, and resolution."""
    filepath = Path(row["filepath"])
    label = row["label"]

    if label not in VALID_LABELS:
        return False, {"filepath": row["filepath"], "reason": f"invalid_label:{label}"}

    if not filepath.exists():
        return False, {"filepath": row["filepath"], "reason": "file_missing"}

    if filepath.stat().st_size == 0:
        return False, {"filepath": row["filepath"], "reason": "zero_byte_file"}

    if filepath.suffix.lower() not in allowed_exts:
        return False, {"filepath": row["filepath"], "reason": f"disallowed_ext:{filepath.suffix}"}

    try:
        with Image.open(filepath) as img:
            img.verify()

        # Re-open to verify full pixel payload decode and RGB convertibility
        with Image.open(filepath) as img:
            width, height = img.size
            if width < min_w or height < min_h:
                return False, {
                    "filepath": row["filepath"],
                    "reason": f"below_min_resolution:{width}x{height}",
                }
            rgb = img.convert("RGB")
            rgb.load()
    except Exception as exc:
        return False, {"filepath": row["filepath"], "reason": f"corrupt_image:{type(exc).__name__}"}

    return True, {**row, "width": width, "height": height}


def main() -> None:
    params = load_params()
    processed_dir = Path(params["data"]["processed_dir"])
    val_cfg = params["validation"]

    manifest_path = processed_dir / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}. Run prepare_dataset.py first.")

    df = pd.read_csv(manifest_path)
    allowed_exts = {ext.lower() for ext in val_cfg["allowed_extensions"]}
    min_w = int(val_cfg["min_width"])
    min_h = int(val_cfg["min_height"])
    min_samples = int(val_cfg["min_samples_per_class"])
    max_corrupt_ratio = float(val_cfg["max_corrupt_ratio"])

    rows = df.to_dict(orient="records")
    valid_rows: List[Dict[str, Any]] = []
    quarantined: List[Dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(validate_single_image, r, allowed_exts, min_w, min_h)
            for r in rows
        ]
        for future in tqdm(futures, total=len(futures), desc="Validating images"):
            is_valid, payload = future.result()
            if is_valid:
                valid_rows.append(payload)
            else:
                quarantined.append(payload)

    total_checked = len(rows)
    corrupt_count = len(quarantined)
    corrupt_ratio = corrupt_count / total_checked if total_checked > 0 else 1.0

    valid_df = (
        pd.DataFrame(valid_rows).sort_values("filepath").reset_index(drop=True)
        if valid_rows
        else pd.DataFrame(columns=["filepath", "raw_class", "label", "label_id", "width", "height"])
    )

    class_counts = {
        cls: int((valid_df["label"] == cls).sum()) for cls in sorted(VALID_LABELS)
    }
    failure_reasons: Dict[str, int] = {}
    for item in quarantined:
        key = item["reason"].split(":")[0]
        failure_reasons[key] = failure_reasons.get(key, 0) + 1

    gate_failures: List[str] = []
    if corrupt_ratio > max_corrupt_ratio:
        gate_failures.append(
            f"Corrupt ratio {corrupt_ratio:.4f} exceeds max_corrupt_ratio {max_corrupt_ratio:.4f}."
        )
    for cls, count in class_counts.items():
        if count < min_samples:
            gate_failures.append(
                f"Class '{cls}' has {count} valid samples, below minimum {min_samples}."
            )

    report = {
        "dvc_raw_md5": get_dvc_data_hash(params["data"]["dvc_file"]),
        "status": "FAILED" if gate_failures else "PASSED",
        "total_checked": total_checked,
        "valid_count": len(valid_df),
        "quarantined_count": corrupt_count,
        "corrupt_ratio": round(corrupt_ratio, 6),
        "class_distribution": class_counts,
        "failure_reason_counts": failure_reasons,
        "gate_failures": gate_failures,
        "quarantined_samples_preview": quarantined[:25],
    }

    report_path = Path(val_cfg["report_path"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(
        f"[validate_data] Status: {report['status']} | "
        f"Valid: {len(valid_df):,}/{total_checked:,} | "
        f"Quarantined: {corrupt_count:,} ({corrupt_ratio:.2%})"
    )

    if gate_failures:
        for msg in gate_failures:
            print(f"[validate_data] GATE FAILURE: {msg}", file=sys.stderr)
        sys.exit(1)

    validated_path = processed_dir / "validated_manifest.csv"
    valid_df.to_csv(validated_path, index=False)
    print(f"[validate_data] Saved validated manifest to {validated_path.as_posix()}")


if __name__ == "__main__":
    main()