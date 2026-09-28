# Digital Twin Bridge: Project and Publication Overview

## 1. Executive Summary

Digital Twin Bridge is a research prototype for explainable structural health monitoring (SHM) of the Z24 bridge benchmark. Its implemented workflow is:

```text
raw signal files -> loading and metadata preservation -> preprocessing and windowing
-> vibration feature extraction -> BOCPD change detection
-> optional XGBoost condition classification -> SHAP attribution
-> sensor prioritisation and risk aggregation -> Streamlit monitoring replay
```

The system is designed to answer four complementary questions:

1. **What data are available?** Dataset inspection and format-aware loading provide source metadata and signal statistics.
2. **When did signal behaviour change?** Bayesian Online Change Point Detection (BOCPD) analyses sequential RMS features.
3. **What configured condition does a window resemble?** A grouped, leakage-aware XGBoost classifier can be trained when explicit labels are available.
4. **Why did the model make that prediction?** SHAP feature attribution and sensor-level aggregation support engineering review.

The project should be described as an explainable monitoring and decision-support prototype. It is not a live IoT system, a safety certification tool, a damage diagnosis system, or a physics-calibrated digital twin.

## 2. Research Problem and Motivation

Bridge SHM systems must transform high-volume vibration measurements into information that an engineer can inspect. A useful prototype must combine detection, condition assessment, interpretability, and visual communication rather than reporting a single opaque score.

This project addresses that need by integrating:

- robust ingestion of heterogeneous Z24 signal files;
- configurable preprocessing and overlapping signal windows;
- time-domain and frequency-domain vibration descriptors;
- unsupervised sequential change detection with BOCPD;
- optional supervised classification with recording-grouped evaluation;
- model explanations with SHAP;
- transparent risk aggregation;
- a visual monitoring twin with historical replay controls.

The central research proposition is therefore an **integrated analytics workflow**, not a claim that any one algorithm independently solves bridge damage identification.

## 3. Main Contributions

The repository implements the following contributions that can be presented in a paper:

1. A modular end-to-end SHM workflow connecting signal ingestion, feature engineering, change detection, classification, explanation, and visualisation.
2. A format-aware loader for Z24-style `.aaa`, `.mat`, `.csv`, `.txt`, and optional ZIP-contained records, while preserving scenario, setup, sensor, recording, channel, and source-file metadata.
3. A leakage-aware supervised-learning path using recording-grouped train/test splitting so overlapping windows from one recording are not randomly distributed between both partitions.
4. A transparent risk formulation combining change probability, abnormal-condition probability, and model attribution when those components are available.
5. A monitoring interface that exposes intermediate evidence rather than only a final status: RMS trends, BOCPD probabilities, sensor rankings, classifier probabilities, SHAP rankings, event logs, and a schematic bridge view.
6. Explicit uncertainty and scope statements that separate statistical change, model attribution, prototype risk, and physical structural damage.

## 4. System Architecture

### 4.1 Processing stages

| Stage | Implementation | Main input | Main output |
|---|---|---|---|
| Configuration | `config.yaml`, `src/utils.py` | YAML settings | Resolved dataset, model, risk, and output paths |
| Inspection | `src/dataset_inspector.py` | External dataset | Inspection report and representative signal plot |
| Loading | `src/data_loader.py` | Signal files or archives | `SignalRecord` objects |
| Preprocessing | `src/preprocessing.py` | Signal records | Detrended, normalised overlapping windows |
| Feature extraction | `src/feature_extraction.py` | Signal windows | Feature table, normally 24 numeric descriptors plus metadata |
| Change detection | `src/bocpd.py` | RMS feature sequences | Change probabilities and run-length estimates |
| Label assignment | `scripts/run_pipeline.py` | Source-file paths and regex rules | Explicit configured labels and mapping report |
| Classification | `src/model_training.py` | Labelled feature table | XGBoost model, feature list, metrics |
| Prediction | `src/model_prediction.py` | Saved model and latest features | Classes, probabilities, confidence |
| Explainability | `src/shap_analysis.py` | Trained tree model and features | Global and local SHAP tables |
| Sensor prioritisation | `src/sensor_mapping.py` | SHAP features | Sensor-level importance ranking |
| Risk aggregation | `src/risk_engine.py` | Detection, ML, attribution values | Score and LOW/MEDIUM/HIGH/CRITICAL level |
| Visualisation | `src/digital_twin.py`, `app.py` | Saved analytics artifacts | Streamlit dashboard and Plotly bridge schematic |

### 4.2 End-to-end data flow

`load_dataset()` discovers files and creates one or more `SignalRecord` objects per file. Each record is validated, repaired, detrended, optionally filtered, normalised, and divided into overlapping `SignalWindow` objects. `extract_features()` converts each window into one row of `features.csv`.

The feature table is passed independently to BOCPD and, when valid labels and grouping information exist, to XGBoost. BOCPD results are stored in `bocpd_results.csv`. The model produces predictions and metrics; SHAP produces feature and local attribution tables. Sensor importance is derived by aggregating attribution by sensor identifier. The pipeline writes prototype risk rows, while the dashboard can additionally combine saved BOCPD, XGBoost, and sensor-importance values for its displayed status.

## 5. Module Inventory

### 5.1 `src/data_loader.py`

Defines the `SignalRecord` dataclass and all raw-data ingestion functions.

- Validates the external dataset root and expected folders.
- Recursively discovers supported file extensions and optional ZIP archives.
- Parses ASCII-like files using whitespace, comma, or semicolon delimiters.
- Recognises the common Z24 single-column header pattern containing sample count and sample interval.
- Loads numerical arrays from MATLAB files, including nested structures where supported by SciPy.
- Reads `.aaa`, `.csv`, and `.txt` files directly and archive members in memory.
- Infers scenario, setup, sensor, and recording identifiers from paths and file names.
- Converts one- and two-dimensional arrays into channel-level `SignalRecord` objects.
- Records loading failures while continuing through other dataset files.

Important boundary: MATLAB v7.3/HDF5 files are reported as requiring conversion or an HDF5-capable loader; the repository does not currently implement that path.

### 5.2 `src/preprocessing.py`

Defines `SignalWindow` and the signal preparation path.

- Validates minimum signal length and the presence of finite values.
- Repairs NaN and infinite samples by linear interpolation.
- Applies optional linear detrending.
- Supports optional Butterworth low-pass, high-pass, or band-pass filtering.
- Supports standard or min-max normalisation.
- Uses a duration-based window when sampling rate is known, otherwise a sample-count fallback.
- Generates overlapping windows with configurable overlap and preserves source metadata.

The default configuration uses 10-second windows, 50% overlap, standardisation, and disabled filtering.

### 5.3 `src/feature_extraction.py`

Computes window-level time, statistical, and spectral descriptors:

- mean, standard deviation, variance, RMS;
- minimum, maximum, absolute peak, peak-to-peak;
- crest factor, skewness, and kurtosis;
- dominant frequency, spectral energy, frequency centroid, spectral bandwidth, and PSD peak;
- four configurable Nyquist-bounded band-energy values.

Welch’s method is used for the power spectral density. Numeric feature selection excludes identifiers, timestamps, and window coordinates before model training.

### 5.4 `src/bocpd.py`

Implements a Student-t Bayesian Online Change Point Detector based on the Adams-MacKay run-length formulation.

For an observation $x_t$, the detector maintains a posterior distribution over run length $r_t$. With constant hazard $h$, growth and reset probabilities are updated using a conjugate Normal-Gamma predictive model. The resulting change probability is the posterior mass at run length zero:

$$
P(r_t = 0 \mid x_{1:t})
$$

The implementation:

- normalises the feature sequence per sensor and recording using its median and standard deviation;
- processes each sensor/recording group independently;
- uses RMS by default;
- bounds the run-length state by `max_run_length`;
- emits the most probable run length and a configurable threshold decision.

BOCPD identifies statistical distribution changes. A threshold crossing is not proof of damage.

### 5.5 `src/model_training.py`

Provides label validation and XGBoost training.

- Rejects missing, unknown, or single-class labels.
- Selects numeric vibration features only.
- Chooses a grouping field from recording, source file, scenario, or setup.
- Uses `GroupShuffleSplit` to reduce leakage from overlapping windows.
- Applies median imputation in a scikit-learn pipeline.
- Trains `XGBClassifier` with the configured tree count, learning rate, depth, row subsampling, and column subsampling.
- Stores the label encoder classes on the fitted pipeline.
- Writes the serialized model, feature list, accuracy, classification report, confusion matrix, and ROC-AUC where valid.

The class labels are configuration-derived path categories, not independently verified physical damage labels.

### 5.6 `src/model_prediction.py`

Loads the serialized pipeline and feature list, verifies that prediction rows contain all required columns, and returns decoded classes, class probabilities, and confidence values.

### 5.7 `src/shap_analysis.py`

Uses SHAP `TreeExplainer` on the XGBoost classifier after applying the pipeline imputer.

- Computes global mean absolute attribution per feature.
- Computes one local explanation for the final analysed row.
- Labels attribution direction as toward or away from the target class.
- Writes `global_feature_importance.csv` and `local_explanation.csv`.

SHAP explains the behaviour of the trained model; it does not establish physical causality.

### 5.8 `src/sensor_mapping.py`

Maps feature names to sensor identifiers with regular expressions and aggregates absolute SHAP importance into a ranked sensor table. This is a review-prioritisation mechanism, not geometric damage localisation.

### 5.9 `src/risk_engine.py`

Computes a bounded weighted score:

$$
R = \frac{w_b B + w_m M + w_a A}{w_b + w_m + w_a}
$$

where $B$ is BOCPD change probability, $M$ is abnormal-condition probability, and $A$ is normalised attribution. The default weights are 0.3, 0.5, and 0.2. Thresholds map the score to LOW, MEDIUM, HIGH, or CRITICAL.

The score is explicitly labelled prototype decision support. In the current pipeline script, risk export passes BOCPD only and leaves ML and attribution components unset; the dashboard can combine all three when saved model and SHAP artifacts are available.

### 5.10 `src/digital_twin.py`

Creates a Plotly schematic bridge deck, piers, and sensor markers. Sensor positions are evenly distributed along the deck using synthetic coordinates. Status colors reflect NORMAL, WARNING, or ANOMALY.

This is a visual monitoring twin, not a physics-based bridge model or verified engineering-coordinate system.

### 5.11 `src/dataset_inspector.py`

Inspects file types, archive members, representative variables, shapes, numeric ranges, missing values, infinite values, inferred metadata, and signal lengths. It writes a JSON report and a representative signal plot without modifying the source dataset.

### 5.12 `src/realtime_simulator.py`

Provides a small stateful iterator with start, pause, stop, reset, next-window, and finished operations. The current dashboard implements equivalent replay logic directly with Streamlit session state; this class is a reusable but lightly integrated simulation utility.

### 5.13 `src/utils.py` and `src/__init__.py`

`utils.py` loads YAML safely, resolves relative project paths, creates output directories, and serialises NumPy/Pandas-compatible values to JSON. `src/__init__.py` identifies the package and declares version `1.0.0`.

## 6. Command-Line Scripts

| Script | Purpose | Main outputs |
|---|---|---|
| `scripts/inspect_dataset.py` | Inspect external data before processing | `results/reports/inspection_report.json`, representative plot |
| `scripts/preprocess_dataset.py` | Load, preprocess, and window records | `data/processed/windows_manifest.json` |
| `scripts/extract_features.py` | Create window-level feature table | `data/processed/features.csv` |
| `scripts/run_bocpd.py` | Run BOCPD on saved features | `data/processed/bocpd_results.csv` |
| `scripts/train_model.py` | Train grouped XGBoost | Model, feature list, metrics |
| `scripts/generate_shap.py` | Generate SHAP and sensor importance | SHAP tables, `sensor_importance.csv` |
| `scripts/plot_vibration_changes.py` | Plot feature behaviour and change points | `results/figures/vibration_change_points.png` |
| `scripts/run_pipeline.py` | Execute the integrated workflow | Combined processed data and result artifacts |

Typical commands are:

```powershell
python scripts/inspect_dataset.py
python scripts/run_pipeline.py --max-files 17
python scripts/train_model.py
python scripts/generate_shap.py
streamlit run app.py
pytest -q
```

The raw dataset is external and must be configured through `dataset.root_path` or `--dataset-root`. ZIP members are enabled deliberately with `--archives`.

## 7. Configuration and Experimental Parameters

`config.yaml` controls the dataset, preprocessing, windowing, BOCPD, model, risk, dashboard, and output locations.

| Component | Current default |
|---|---|
| Dataset formats | `.aaa`, `.mat`, `.csv`, `.txt`; optional archives |
| Sampling rate | `null`, so fallback behaviour may apply |
| Detrending | Linear detrending enabled |
| Filtering | Disabled by default; band-pass settings 0.5-40 Hz when enabled |
| Window | 10 s when rate is known; 1,000 samples otherwise |
| Overlap | 50% |
| BOCPD feature | RMS |
| BOCPD hazard | 0.01 |
| BOCPD threshold | 0.80 |
| Maximum run length | 500 |
| XGBoost | 300 trees, learning rate 0.05, depth 6 |
| Split | 25% grouped holdout, random state 42 |
| Risk weights | BOCPD 0.3, ML 0.5, attribution 0.2 |
| Risk thresholds | 0.35 medium, 0.60 high, 0.80 critical |

The Z24 label rules currently map scenarios 01, 02, and 08 to `REFERENCE`, and scenarios 03-07 and 09-17 to `ALTERED_CONDITION`. The software applies these rules to source paths; it does not infer engineering meaning from scenario numbers.

## 8. Repository Artifacts

### Source and configuration

- `app.py`: Streamlit dashboard and historical replay.
- `config.yaml`: experiment and path configuration.
- `requirements.txt`: Python dependencies.
- `src/`: reusable implementation modules.
- `scripts/`: command-line workflow stages.
- `tests/`: unit and focused integration tests.
- `docs/`: architecture, methodology, and limitations notes.

### Generated data and results

- `data/processed/features.csv`: window-level features and metadata.
- `data/processed/bocpd_results.csv`: BOCPD probabilities and run lengths.
- `data/processed/risk_results.csv`: exported prototype risk rows.
- `data/processed/label_mapping.json`: label patterns and match counts.
- `data/processed/sensor_importance.csv`: ranked sensor attribution.
- `models/xgboost.pkl`: fitted model when training succeeds.
- `models/feature_columns.json`: model input contract.
- `results/reports/inspection_report.json`: dataset inspection report.
- `results/metrics/model_metrics.json`: classifier metrics.
- `results/metrics/model_status.json`: reason training was skipped when applicable.
- `results/shap/global_feature_importance.csv`: global SHAP rankings.
- `results/shap/local_explanation.csv`: local SHAP contributions.
- `results/figures/`: dataset and vibration plots.

Raw Z24 data is intentionally not committed to the repository.

## 9. Testing and Verification

The test suite covers:

- Z24-style ASCII header parsing and malformed-row handling;
- MATLAB array loading;
- invalid-value repair and standardisation;
- window count and metadata preservation;
- zero-safe feature extraction;
- BOCPD probability bounds and grouped execution;
- sensor-ID extraction and attribution aggregation;
- risk threshold mapping;
- rejection of missing or unusable labels;
- grouped XGBoost training;
- optional SHAP computation.

The suite does not currently provide full end-to-end validation for the Streamlit interface, archive-scale ingestion, label-rule coverage, filtering, plot CLI behaviour, artifact provenance, or run-to-run reproducibility. These are appropriate targets for a reproducibility extension or supplementary material.

## 10. Current Results and Interpretation

The checked-in generated artifacts indicate a large feature table, BOCPD results, a trained XGBoost model, SHAP tables, and inspection outputs. They also indicate that the outputs were produced by more than one run or configuration: `model_status.json` reports a skipped supervised stage while a model and SHAP outputs are present. The repository does not currently store run IDs, configuration hashes, dataset fingerprints, or timestamps linking artifacts into one immutable experiment.

The saved classifier metrics should be reported cautiously. The observed accuracy is approximately 0.50 and ROC-AUC approximately 0.49, which is close to random binary discrimination for the inspected artifact. This supports a statement that the pipeline can execute the classification and explanation stages, not a statement that it reliably identifies structural condition.

The checked-in BOCPD output contains no crossings of the configured 0.80 threshold, although probability variation remains visible below the threshold. This is evidence about the selected data, parameters, and run; it is not evidence that the bridge has no damage.

## 11. Limitations Relevant to a Paper

1. **External data dependency:** raw data, acquisition conditions, and complete provenance are outside the repository.
2. **Sampling-rate uncertainty:** the configured sampling rate is null for the inspected run. Spectral calculations may therefore use the 1 Hz fallback, with an effective 0.5 Hz Nyquist limit and potentially uninformative high-frequency bands.
3. **Label validity:** path-based `REFERENCE` and `ALTERED_CONDITION` labels are configured categories, not independently verified damage annotations.
4. **Statistical versus physical change:** BOCPD detects changes in feature distributions and cannot identify damage by itself.
5. **Weak current classification evidence:** the saved classifier performance is near random and should not be used as a reliability claim.
6. **Validation scope:** grouped window splitting reduces overlap leakage but is not independent bridge-level validation.
7. **Risk inconsistency:** the methodology defines a three-component risk score, while the pipeline export currently supplies BOCPD alone; dashboard and pipeline risk values should not be conflated.
8. **Attribution interpretation:** SHAP indicates model contribution, not causation.
9. **Sensor localisation:** sensor aggregation prioritises review but does not infer a physical damage location.
10. **Schematic geometry:** displayed sensor coordinates are synthetic unless verified coordinates are supplied.
11. **Replay only:** the dashboard simulates historical streaming and has no live acquisition, message broker, edge deployment, or IoT integration.
12. **No physics-based twin:** no finite-element model, calibrated state estimator, or mechanics-informed digital representation is implemented.

## 12. Recommended Paper Structure

### Abstract

State the SHM motivation, the integrated workflow, the Z24 benchmark, the combination of BOCPD/XGBoost/SHAP, and the prototype nature of the monitoring interface. Avoid claiming validated damage diagnosis.

### Introduction

Motivate interpretable monitoring, distinguish anomaly detection from damage diagnosis, and identify the need to combine statistical evidence with model explanations and engineering review.

### Related Work

Discuss vibration-based SHM, Bayesian online change detection, tree-based condition classification, explainable AI for SHM, and digital-twin interfaces. Position this project as an integration and transparency contribution.

### Materials and Dataset

Describe the Z24 benchmark, file formats, scenarios, sensor metadata, acquisition context, label-rule provenance, and the fact that raw data is external to this repository.

### Methods

Describe ingestion, preprocessing, windowing, the 24-feature representation, BOCPD equations and parameters, grouped XGBoost evaluation, SHAP, sensor aggregation, and the prototype risk formula.

### System Implementation

Map the methods to the modules and scripts in this document. Include the artifact paths and the Streamlit historical replay interface.

### Experimental Protocol

Report the exact dataset selection, sampling rate, file/archive options, maximum files, configuration revision, random seed, grouping field, and artifact-generation order. Add run IDs and configuration hashes before making strong reproducibility claims.

### Results

Report detection probabilities and threshold crossings, classification metrics with confusion matrix and class balance, SHAP rankings, sensor rankings, and runtime/resource usage. Keep descriptive results separate from claims about physical damage.

### Discussion

Explain how the complementary outputs support review, why attribution improves transparency, and why statistical evidence must be corroborated with engineering knowledge and independent measurements.

### Limitations and Future Work

Include live sensing, physics-based modelling, verified coordinates, better labels, bridge-level validation, sensor fusion, calibrated thresholds, and improved provenance tracking.

### Reproducibility Statement

List Python version, dependency versions, configuration, dataset location/version, selected files, random seed, output hashes, and the exact commands used to generate every table and figure.

## 13. Publication-Safe Claim Language

Prefer:

- “The system detects statistical changes in vibration-derived features.”
- “The classifier provides a configured-condition estimate under a grouped holdout protocol.”
- “SHAP identifies features that contribute to model predictions.”
- “The dashboard supports historical replay and engineering review.”
- “The prototype risk score integrates available analytical indicators.”

Avoid:

- “The system proves structural damage.”
- “The health score certifies bridge safety.”
- “SHAP identifies the physical cause of damage.”
- “The dashboard is a live digital twin.”
- “The current classifier is reliable” without new, independently validated evidence.

## 14. Recommended Next Steps Before Submission

1. Capture dataset fingerprint, configuration snapshot, dependency lockfile, run ID, timestamp, and output hashes for every experiment.
2. Resolve and document the true sampling rate before interpreting frequency features or band energies.
3. Reconcile `risk_results.csv` with the stated three-component risk method, or document the pipeline’s BOCPD-only mode explicitly.
4. Evaluate against bridge-level or recording-level independent test data and report class balance and confidence intervals.
5. Validate label semantics against the benchmark documentation and distinguish progressive test states from confirmed damage.
6. Add end-to-end tests for the pipeline and dashboard data contracts.
7. Supply verified sensor coordinates or rename the visual output consistently as a schematic.
8. Regenerate all committed artifacts in one clean, documented run before using them in paper tables or figures.
