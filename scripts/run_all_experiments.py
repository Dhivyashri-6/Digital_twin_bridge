"""Run the complete, reproducible Z24 study in order (each stage writes results/runs/<run_id>/)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGES = [
    ["scripts/process_z24.py"],
    ["scripts/run_change_detection.py"],
    ["scripts/run_classification.py", "--experiment", "main"],
    ["scripts/run_classification.py", "--experiment", "physics"],
    ["scripts/run_classification.py", "--experiment", "ablation", "--seeds", "0", "--schemes", "loso"],
    ["scripts/run_classification.py", "--experiment", "cost"],
    ["scripts/run_shap.py", "--seeds", "0"],
    ["scripts/run_risk.py"],
    ["scripts/run_ems.py", "--stage", "analyse"],  # re-extracts automatically if the cache is missing
    ["scripts/run_novelty.py"],
    ["scripts/run_ssi.py"],
    ["scripts/run_mode_shapes.py"],
    ["scripts/run_relative.py"],
    ["scripts/make_figures.py"],
    ["scripts/write_results_summary.py"],
]

if __name__ == "__main__":
    skip = set(sys.argv[1:])
    for stage in STAGES:
        if Path(stage[0]).stem in skip or not (ROOT / stage[0]).exists():
            print("skip", stage)
            continue
        print(">>", " ".join(stage), flush=True)
        subprocess.run([sys.executable, *stage], cwd=ROOT, check=True)
