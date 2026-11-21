from pathlib import Path
from typing import Dict, List

import pandas as pd

from src.utils.tracking import load_params, set_global_seed

LABEL_TO_ID: Dict[str, int] = {
    "Recycle": 0,
    "Compost": 1,
    "Landfill": 2,
}


def build_reverse_mapping(class_mapping: Dict[str, List[str]]) -> Dict[str, str]:
    """Invert target -> [raw_classes] into raw_class -> target."""
    reverse_map: Dict[str, str] = {}
    for target_label, raw_classes in class_mapping.items():
        if target_label not in LABEL_TO_ID:
            raise ValueError(f"Unsupported target class in params.yaml: {target_label}")
        for raw_cls in raw_classes:
            if raw_cls in reverse_map:
                raise ValueError(f"Raw class '{raw_cls}' mapped more than once.")
            reverse_map[raw_cls] = target_label
    return reverse_map


def main() -> None:
    params = load_params()
    seed = params["project"]["seed"]
    set_global_seed(seed)

    raw_dir = Path(params["data"]["raw_dir"])
    processed_dir = Path(params["data"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir.resolve()}")

    reverse_map = build_reverse_mapping(params["data"]["class_mapping"])
    records = []
    unmapped_dirs = []

    for subdir in sorted(p for p in raw_dir.iterdir() if p.is_dir()):
        raw_cls = subdir.name
        if raw_cls not in reverse_map:
            unmapped_dirs.append(raw_cls)
            continue

        target_label = reverse_map[raw_cls]
        label_id = LABEL_TO_ID[target_label]

        for file_path in sorted(p for p in subdir.rglob("*") if p.is_file()):
            records.append(
                {
                    "filepath": file_path.as_posix(),
                    "raw_class": raw_cls,
                    "label": target_label,
                    "label_id": label_id,
                }
            )

    if unmapped_dirs:
        raise ValueError(
            f"Found unmapped directories in {raw_dir}: {unmapped_dirs}. "
            "Update params.yaml class_mapping before proceeding."
        )

    if not records:
        raise RuntimeError(f"No files discovered under {raw_dir.resolve()}.")

    df = pd.DataFrame(records).sort_values("filepath").reset_index(drop=True)

    max_per_class = params["data"].get("max_samples_per_class")
    if max_per_class is not None:
        df = (
            df.groupby("label", group_keys=False)
            .apply(lambda g: g.sample(n=min(len(g), int(max_per_class)), random_state=seed))
            .sort_values("filepath")
            .reset_index(drop=True)
        )

    manifest_path = processed_dir / "manifest.csv"
    df.to_csv(manifest_path, index=False)

    print(f"[prepare_dataset] Wrote {len(df):,} records to {manifest_path.as_posix()}")
    print("[prepare_dataset] Class distribution:")
    for label, count in df["label"].value_counts().items():
        print(f"  - {label:<10}: {count:,}")


if __name__ == "__main__":
    main()