#!/usr/bin/env bash
# Feature/normalisation ablation, XGBoost, LOSO only, seed 0 (cached configs are reused).
cd "$(dirname "$0")/.."
set -e
python scripts/run_classification.py --experiment ablation --seeds 0 --schemes loso > results/classification_ablation_loso.log 2>&1
echo ABLATION_DONE > results/ablation_done.flag
