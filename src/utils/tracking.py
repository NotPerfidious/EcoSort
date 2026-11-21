import os
import random
import subprocess
from pathlib import Path
from typing import Any, Dict

import mlflow
import numpy as np
import yaml


def load_params(params_path: str = "params.yaml") -> Dict[str, Any]:
    """Load pipeline parameters from YAML configuration."""
    path = Path(params_path)
    if not path.exists():
        raise FileNotFoundError(f"Parameters file not found at: {path.resolve()}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_global_seed(seed: int) -> None:
    """Enforce deterministic behavior across Python, NumPy, and PyTorch."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        # Allows lightweight data scripts to run even if torch isn't installed yet
        pass


def get_dvc_data_hash(dvc_file_path: str = "data/raw.dvc") -> str:
    """
    Extract the DVC MD5 hash from a .dvc file so every MLflow run
    is cryptographically tied to the exact dataset version used.
    """
    path = Path(dvc_file_path)
    if not path.exists():
        return "untracked"

    with path.open("r", encoding="utf-8") as f:
        dvc_meta = yaml.safe_load(f) or {}

    outs = dvc_meta.get("outs", [])
    if outs and isinstance(outs, list):
        return str(outs[0].get("md5", "unknown"))
    return "unknown"


def get_git_commit_hash() -> str:
    """Return the current short Git commit SHA (with '-dirty' suffix if uncommitted)."""
    try:
        sha = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode("utf-8")
            .strip()
        )
        status = (
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                stderr=subprocess.DEVNULL,
            )
            .decode("utf-8")
            .strip()
        )
        return f"{sha}-dirty" if status else sha
    except Exception:
        return "no-git"


def setup_mlflow(params: Dict[str, Any]) -> str:
    """Configure MLflow tracking URI and experiment, returning the experiment ID."""
    mlflow_cfg = params["mlflow"]
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", mlflow_cfg["tracking_uri"])
    experiment_name = mlflow_cfg["experiment_name"]

    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.set_experiment(experiment_name)
    return experiment.experiment_id


def get_standard_run_tags(params: Dict[str, Any], stage: str) -> Dict[str, str]:
    """Build standard provenance tags for MLflow runs."""
    return {
        "project": params["project"]["name"],
        "stage": stage,
        "dvc_raw_md5": get_dvc_data_hash(params["data"]["dvc_file"]),
        "git_commit": get_git_commit_hash(),
    }