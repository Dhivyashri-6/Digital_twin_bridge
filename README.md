# Digital Twin Bridge

An explainable structural-health-monitoring workflow for the Z24 benchmark: **data → detect → classify → explain → visualize**. It loads real `.aaa`/`.mat` data, preprocesses and windows signals, extracts vibration features, runs Bayesian Online Change Point Detection (BOCPD), optionally trains leakage-aware XGBoost when reliable labels exist, explains it with SHAP, and displays a schematic monitoring twin in Streamlit.

> The system detects statistical changes and supports engineering review. It does not certify safety or prove damage.

## Dataset setup

The dataset stays outside this repository and is never modified. This installation discovered it at `D:\dataset`, with `pdt_01-08`, `pdt_09_17`, `Z24ems1`, `Z24ems2`, and `Z24ems3`. Edit `dataset.root_path` in `config.yaml`, or override it with `--dataset-root "D:\your\path"`. Extracted folders are preferred; EMS per-record ZIPs can be read in place with `--archives`.

Labels are never inferred from scenario numbers. To train a classifier, add documented regular-expression rules under `dataset.label_rules`, mapping source paths to classes. The generated `data/processed/label_mapping.json` records every rule and match count.

## Windows installation

```powershell
cd D:\Digital_twin_bridge
python -m venv venv
venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run

```powershell
python scripts\inspect_dataset.py
python scripts\preprocess_dataset.py --max-files 5
python scripts\extract_features.py --max-files 5
python scripts\run_bocpd.py
python scripts\train_model.py
python scripts\generate_shap.py
python scripts\run_pipeline.py --max-files 10
streamlit run app.py
pytest -q
```

Use `--archives` when deliberately processing ZIP members. Start small: the full dataset is several gigabytes and contains thousands of archives. All derived artifacts go to `data/processed`, `models`, and `results`.

## Outputs

- `features.csv`, `bocpd_results.csv`, `risk_results.csv`, `label_mapping.json`
- `xgboost.pkl`, `feature_columns.json` when training is valid
- inspection report/plots, metrics, and SHAP tables
- dashboard with KPI cards, change chart, sensor table, event history, and schematic Plotly bridge

The dashboard provides Start/Pause/Stop/Reset controls and 0.5×–5× historical playback. It clearly identifies replay as simulated real time.

## Feature set

Mean, standard deviation, variance, RMS, extrema, absolute peak, peak-to-peak, crest factor, skewness, kurtosis, dominant frequency, spectral energy, centroid, bandwidth, PSD peak, and configurable Nyquist-safe band energies are calculated per window.

## Research framing

The project proposes an integrated workflow combining online change detection, supervised classification, explainable AI, and monitoring-twin visualization. See `docs/methodology.md`, `docs/architecture.md`, and `docs/limitations.md` for interpretation, equations, constraints, and future work.
