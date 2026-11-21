import hashlib
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple

import imagehash
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from src.utils.tracking import get_dvc_data_hash, load_params, set_global_seed


class UnionFind:
    """Disjoint-set structure for clustering near-duplicate image indices."""

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def compute_hashes(filepath: str, hash_size: int) -> Tuple[str, str, np.ndarray]:
    """Compute exact MD5 hash and binary pHash vector for a single image."""
    path = Path(filepath)
    md5 = hashlib.md5(path.read_bytes()).hexdigest()
    with Image.open(path) as img:
        phash = imagehash.phash(img.convert("RGB"), hash_size=hash_size)
    bits = phash.hash.flatten().astype(np.int8)
    return md5, str(phash), bits


def cluster_near_duplicates(
    bit_matrix: np.ndarray,
    hamming_threshold: int,
    chunk_size: int = 1024,
) -> np.ndarray:
    """
    Cluster images whose pHash Hamming distance <= hamming_threshold.

    Uses bipolar encoding {-1, +1}: for B-bit vectors u and v,
    dot(u, v) = B - 2 * hamming_distance(u, v),
    so hamming_distance <= T  <=>  dot(u, v) >= B - 2*T.
    """
    n, num_bits = bit_matrix.shape
    min_dot = float(num_bits - 2 * hamming_threshold) - 0.5
    bipolar = np.where(bit_matrix > 0, 1.0, -1.0).astype(np.float32)

    uf = UnionFind(n)
    for start in tqdm(range(0, n, chunk_size), desc="Clustering near-duplicates"):
        end = min(start + chunk_size, n)
        # Only compute upper triangle (j > i) to avoid redundant work
        dots = bipolar[start:end] @ bipolar[start:].T
        local_rows, local_cols = np.where(dots >= min_dot)
        for r, c in zip(local_rows, local_cols):
            i = start + int(r)
            j = start + int(c)
            if j > i:
                uf.union(i, j)

    roots = np.array([uf.find(i) for i in range(n)], dtype=np.int64)
    # Remap arbitrary roots to compact sequential cluster IDs
    _, compact_ids = np.unique(roots, return_inverse=True)
    return compact_ids


def assign_group_stratified_splits(
    df: pd.DataFrame,
    ratios: Dict[str, float],
    seed: int,
) -> pd.Series:
    """
    Assign near-duplicate clusters to train/val/test while preserving
    target split ratios per class and guaranteeing zero cluster overlap.
    """
    rng = np.random.default_rng(seed)
    split_names = list(ratios.keys())
    labels = sorted(df["label"].unique())

    # Target sample counts per (split, label)
    class_totals = df["label"].value_counts().to_dict()
    targets = {
        sp: {lbl: class_totals[lbl] * ratios[sp] for lbl in labels}
        for sp in split_names
    }
    current = {sp: {lbl: 0 for lbl in labels} for sp in split_names}

    # Build cluster metadata: (cluster_id, size, {label: count})
    clusters = []
    for cid, grp in df.groupby("cluster_id"):
        counts = grp["label"].value_counts().to_dict()
        clusters.append((int(cid), len(grp), counts))

    # Shuffle deterministically, then sort descending by cluster size (greedy bin-packing)
    perm = rng.permutation(len(clusters))
    clusters = [clusters[i] for i in perm]
    clusters.sort(key=lambda x: x[1], reverse=True)

    cluster_to_split: Dict[int, str] = {}
    for cid, _, counts in clusters:
        # Score each split by normalized remaining capacity for the cluster's labels
        best_split = max(
            split_names,
            key=lambda sp: sum(
                (targets[sp][lbl] - current[sp][lbl]) / max(targets[sp][lbl], 1.0) * cnt
                for lbl, cnt in counts.items()
            ),
        )
        cluster_to_split[cid] = best_split
        for lbl, cnt in counts.items():
            current[best_split][lbl] += cnt

    return df["cluster_id"].map(cluster_to_split)


def assert_zero_leakage(df: pd.DataFrame) -> None:
    """Hard assertions verifying no filepath, MD5, or pHash cluster crosses split boundaries."""
    splits = ["train", "val", "test"]
    for col in ["filepath", "md5", "cluster_id"]:
        sets = {sp: set(df.loc[df["split"] == sp, col]) for sp in splits}
        for i in range(len(splits)):
            for j in range(i + 1, len(splits)):
                s1, s2 = splits[i], splits[j]
                overlap = sets[s1].intersection(sets[s2])
                assert not overlap, (
                    f"DATA LEAKAGE DETECTED on '{col}' between '{s1}' and '{s2}': "
                    f"{len(overlap)} overlapping items!"
                )


def main() -> None:
    params = load_params()
    seed = int(params["project"]["seed"])
    set_global_seed(seed)

    processed_dir = Path(params["data"]["processed_dir"])
    split_cfg = params["split"]

    ratios = {
        "train": float(split_cfg["train_ratio"]),
        "val": float(split_cfg["val_ratio"]),
        "test": float(split_cfg["test_ratio"]),
    }
    if abs(sum(ratios.values()) - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {ratios}")

    validated_path = processed_dir / "validated_manifest.csv"
    if not validated_path.exists():
        raise FileNotFoundError(
            f"Validated manifest not found: {validated_path}. Run validate_data.py first."
        )

    df = pd.read_csv(validated_path).sort_values("filepath").reset_index(drop=True)
    hash_size = int(split_cfg["phash_size"])
    hamming_threshold = int(split_cfg["phash_hamming_threshold"])

    # 1. Compute MD5 + pHash in parallel
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(
            tqdm(
                executor.map(lambda fp: compute_hashes(fp, hash_size), df["filepath"]),
                total=len(df),
                desc="Computing MD5 & pHashes",
            )
        )

    df["md5"] = [r[0] for r in results]
    df["phash"] = [r[1] for r in results]
    bits_list = [r[2] for r in results]

    # 2. Remove exact MD5 duplicates deterministically (keep first sorted filepath)
    before_dedup = len(df)
    keep_mask = ~df["md5"].duplicated(keep="first")
    exact_dupes_removed = int((~keep_mask).sum())
    df = df.loc[keep_mask].reset_index(drop=True)
    bit_matrix = np.vstack([b for b, keep in zip(bits_list, keep_mask) if keep])

    print(
        f"[split_data] Exact MD5 duplicates removed: {exact_dupes_removed:,} "
        f"({before_dedup:,} -> {len(df):,})"
    )

    # 3. Cluster near-duplicates via pHash Hamming distance
    df["cluster_id"] = cluster_near_duplicates(bit_matrix, hamming_threshold)
    num_clusters = int(df["cluster_id"].nunique())
    multi_image_clusters = int((df["cluster_id"].value_counts() > 1).sum())

    # 4. Group-disjoint stratified split
    df["split"] = assign_group_stratified_splits(df, ratios=ratios, seed=seed)

    # 5. Programmatic leakage verification
    assert_zero_leakage(df)
    print("[split_data] Leakage assertions PASSED (zero filepath, MD5, or pHash cluster overlap).")

    splits_csv = Path(split_cfg["splits_csv"])
    splits_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(splits_csv, index=False)

    # 6. Write split summary report
    split_breakdown: Dict[str, Dict[str, int]] = {}
    for sp in ["train", "val", "test"]:
        sub = df[df["split"] == sp]
        split_breakdown[sp] = {
            "total": int(len(sub)),
            **{lbl: int((sub["label"] == lbl).sum()) for lbl in sorted(df["label"].unique())},
        }

    summary = {
        "dvc_raw_md5": get_dvc_data_hash(params["data"]["dvc_file"]),
        "seed": seed,
        "input_valid_images": before_dedup,
        "exact_md5_duplicates_removed": exact_dupes_removed,
        "unique_images_after_md5": int(len(df)),
        "phash_hamming_threshold": hamming_threshold,
        "total_phash_clusters": num_clusters,
        "near_duplicate_clusters_gt_1": multi_image_clusters,
        "leakage_assertions_passed": True,
        "splits": split_breakdown,
    }

    summary_path = Path(split_cfg["summary_path"])
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[split_data] Saved splits to {splits_csv.as_posix()} and summary to {summary_path.as_posix()}")
    for sp, stats in split_breakdown.items():
        print(f"  - {sp:<5}: {stats}")


if __name__ == "__main__":
    main()