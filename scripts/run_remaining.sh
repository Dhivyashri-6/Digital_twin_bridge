#!/usr/bin/env bash
# Remaining stages after the runtime cuts (3 seeds main/physics, SHAP seed 0, physics modal-only). Cached OOF scores are reused.
cd "$(dirname "$0")/.."
set -e
python scripts/run_classification.py --experiment main > results/classification_main.log 2>&1
python scripts/run_classification.py --experiment physics > results/classification_physics.log 2>&1
python scripts/run_shap.py --seeds 0 > results/shap.log 2>&1
python scripts/run_risk.py > results/risk.log 2>&1
python scripts/run_classification.py --experiment cost > results/classification_cost.log 2>&1
python scripts/run_novelty.py > results/novelty.log 2>&1
echo CHAIN_DONE > results/remaining_done.flag
