#!/usr/bin/env bash
set -euo pipefail

echo "=========================================="
echo " EcoSort Deliverable 1 — Reproducible Run"
echo "=========================================="

# Pull versioned data from DVC remote if data/raw is not already present locally
if [ ! -d "data/raw" ] || [ -z "$(ls -A data/raw 2>/dev/null)" ]; then
  echo "[reproduce] Pulling versioned dataset via DVC..."
  dvc pull -j 8
else
  echo "[reproduce] Local data/raw detected — skipping dvc pull."
fi

echo "[reproduce] Executing end-to-end DVC pipeline (prepare -> validate -> split -> baseline -> mobilenetv4 -> evaluate_register)..."
dvc repro

echo "[reproduce] Pipeline complete! Launch MLflow UI with:"
echo "  mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000 --workers 1"