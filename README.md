# Digital Twin Bridge

A reproducible structural-health-monitoring digital twin for the Z24 bridge benchmark: **data → modal identification →
change detection → novelty detection → classification → SHAP explanation → gated risk score → dashboard**. It processes
all 306 progressive-damage-test recordings (17 scenarios × AVT/FVT × 9 setups, 100 Hz) and the one-year EMS campaign, and
evaluates every component with leave-one-scenario-out validation and bootstrap confidence intervals over recordings.

> The system detects statistical changes of the modal properties and supports engineering review. It does not certify
> safety or prove damage.

## Dataset setup

The raw data stay outside this repository and are never modified. Set `dataset.root_path` in `config.yaml` to the folder
that contains `pdt_01-08`, `pdt_09_17` and `Z24ems1-3`. Labels follow the documented `dataset.label_rules`
(scenarios 01, 02, 08 = REFERENCE, all others = ALTERED_CONDITION); they describe the test protocol, not measured damage.

## Installation (Windows)

```powershell
cd D:\Digital_twin_bridge
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run the complete study

```powershell
python scripts\run_all_experiments.py      # all stages in order (cached results are reused)
streamlit run app.py                        # dashboard (historical replay of the study outputs)
pytest -q                                   # unit tests
```

Stages (each can be run on its own; each writes `results/runs/<run_id>/` with config snapshot, package versions,
seeds, wall time and SHA-256 hashes of its outputs):

| Script | Output |
|---|---|
| `process_z24.py` | catalogue, duplicate checks, windows + features, FDD/EFDD modal features (`data/processed/`) |
| `run_change_detection.py` | within-record BOCPD; sequence BOCPD vs CUSUM against the 16 scenario boundaries |
| `run_classification.py --experiment main/physics/ablation/cost` | XGBoost, RF, SVM, LR, frequency thresholds; LOSO + grouped 5-fold; CIs |
| `run_shap.py` | TreeSHAP on all held-out LOSO windows; all sensors ranked, aggregated by location |
| `run_risk.py` | validation-gated risk score and digital-twin states |
| `run_ems.py` | one-year EMS tracking, temperature compensation, BOCPD/CUSUM onset detection |
| `run_novelty.py` | label-free Mahalanobis novelty score (baselines S01 and S01+S02, empirical threshold) |
| `run_ssi.py` | SSI-COV cross-check of the FDD frequencies and damping |
| `run_mode_shapes.py` | channel-level mode-shape change features, classification and SHAP localisation test |
| `make_figures.py` | 300-dpi figures in `results/figures/paper/` |
| `write_results_summary.py` | `results/RESULTS_SUMMARY.md` – every number with its source file |

## Main results (all from `results/RESULTS_SUMMARY.md`)

- Four natural frequencies (3.91, 5.13, 9.92, 12.61 Hz in scenario 01) track the interventions; SSI-COV confirms them
  (median differences 0.09–0.69%).
- Sequence BOCPD finds the scenario boundaries with precision 0.833 / recall 0.625 (CUSUM h = 5: 0.579 / 0.688).
- Supervised classifiers reach ROC-AUC 0.90–0.94 under recording-grouped CV but at most 0.759 (SVM) under LOSO, below a
  natural-frequency shift threshold (0.821) and a post-hoc frequency-decrease threshold (0.902).
- SHAP: 99.8% of the attribution falls on record-level modal features; sensor attributions are nearly uniform, and
  mode-shape change features do not change this.
- A label-free novelty score calibrated on both pre-intervention reference measurements flags 0% of the held-out
  reference and 71% of the altered recordings (balanced accuracy 0.856).
- EMS year: temperature-compensated BOCPD raises no alarm in the month before the damage tests and alarms one day after
  their start.

See `docs/methodology.md` for the methods and the list of choices made after seeing results.
