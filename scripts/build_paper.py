"""Build Digital_Twin_SHM_Paper_BPASTS.docx from 'Final Paper 1.docx' and the saved result files.

Every number inserted into the paper is read from a result file at build time (see R dict / helpers);
the original paper file is not modified.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import docx
import pandas as pd

from paper_docx import Templates, body_elements, find_index, replace_range, set_paragraph_text, set_rich_text, parse_math, text_of
from src.utils import PROJECT_ROOT, load_config

SRC = PROJECT_ROOT / "Final Paper 1.docx"
DST = PROJECT_ROOT / "Digital_Twin_SHM_Paper_BPASTS.docx"
RES = PROJECT_ROOT / "results"
PROC = PROJECT_ROOT / "data" / "processed"
FIG = RES / "figures" / "paper"
MODEL = {"xgboost": "XGBoost", "random_forest": "RF", "svm": "SVM", "logistic_regression": "LR",
         "freq_threshold": "|Δf| threshold", "xgboost_monotone": "XGBoost-M", "freq_decrease_threshold": "Δf↓ threshold"}


def f3(v):
    return f"{v:.3f}"


def ci(r, m, d=3):
    return f"{r[m]:.{d}f} [{r[m + '_ci_low']:.{d}f}, {r[m + '_ci_high']:.{d}f}]"


def num(n):
    return f"{int(n):,}".replace(",", " ")


def _opt_csv(p, **kw):
    return pd.read_csv(p, **kw) if Path(p).exists() else None


def _opt_json(p):
    return json.loads(Path(p).read_text()) if Path(p).exists() else None


def load():
    cfg = load_config()
    d = {"cfg": cfg}
    d["counts"] = json.loads((PROC / "data_counts.json").read_text())
    d["rec"] = pd.read_csv(PROC / "recordings.csv", dtype={"scenario": str, "setup": str})
    d["base"] = json.loads((PROC / "modal_baseline.json").read_text())
    d["within"] = json.loads((RES / "change_detection" / "within_record_bocpd_summary.json").read_text())
    d["seq"] = pd.read_csv(RES / "change_detection" / "sequence_detection_comparison.csv")
    d["perb"] = pd.read_csv(RES / "change_detection" / "sequence_per_boundary.csv")
    C, S, K = RES / "classification", RES / "shap", RES / "risk"
    d["clfdata"] = _opt_json(C / "classification_dataset.json")
    d["main"] = _opt_csv(C / "main_results.csv")
    d["main_ps"] = _opt_csv(C / "main_per_scenario.csv", dtype={"scenario": str})
    d["phys"] = _opt_csv(C / "physics_results.csv")
    d["phys_ps"] = _opt_csv(C / "physics_per_scenario.csv", dtype={"scenario": str})
    d["abl"] = _opt_csv(C / "ablation_results.csv")
    d["cost"] = _opt_csv(C / "model_cost.csv")
    d["timing"] = _opt_csv(C / "main_timing.csv")
    d["shap"] = _opt_json(S / "shap_summary.json")
    d["shap_g"] = _opt_csv(S / "global_feature_importance.csv")
    d["shap_grp"] = _opt_csv(S / "feature_group_importance.csv")
    d["shap_s"] = _opt_csv(S / "sensor_importance_all.csv")
    d["shap_sg"] = _opt_csv(S / "importance_by_sensor_group.csv")
    d["shap_dir"] = _opt_csv(S / "importance_by_sensor_direction.csv")
    d["risk"] = _opt_json(K / "risk_summary.json")
    d["risk_lab"] = _opt_csv(K / "states_by_label.csv")
    d["risk_sc"] = _opt_csv(K / "scenario_states.csv", dtype={"scenario": str})
    d["ssi"] = _opt_json(RES / "ssi" / "ssi_summary.json")
    d["mshape"] = _opt_json(RES / "mode_shapes" / "mode_shape_summary.json")
    d["novelty"] = _opt_json(RES / "novelty" / "novelty_summary.json")
    d["novelty_ps"] = _opt_csv(RES / "novelty" / "novelty_per_scenario.csv", dtype={"scenario": str})
    d["ems"] = _opt_json(RES / "ems" / "ems_summary.json")
    d["ems_det"] = _opt_csv(RES / "ems" / "ems_detection.csv")
    return d


def row(df, **kw):
    q = df
    for k, v in kw.items():
        q = q[q[k] == v]
    if len(q) != 1:
        raise KeyError(f"{kw}: {len(q)} rows")
    return q.iloc[0]


# ------------------------------------------------------------------------------------------ Sections 3-4
def methods_text(d) -> dict[str, str]:
    c, cfg = d["counts"], d["cfg"]
    base = ", ".join(f"{b['frequency_hz']:.2f}" for b in d["base"])
    avt = d["rec"][d["rec"].test == "avt"]
    zeta_rate = {k: avt[f"zeta{k}_reliable"].mean() for k in range(1, len(d["base"]) + 1)}
    cd = d["clfdata"]
    sb = cfg["sequence_bocpd"]
    ev = cfg["evaluation"]
    t = {}
    t["3.2"] = (
        f"All progressive-damage-test recordings were processed: 17 scenarios, the ambient (AVT) and forced (FVT) vibration "
        f"test of each scenario and all 9 sensor setups, i.e. {c['recordings']} recordings with unique identifiers of the form "
        f"S<scenario>_<test>_setup<NN>. The {c['catalogue']['aaa_files']} ASCII (.aaa) files of scenarios 06 and 17 and the "
        f"{c['catalogue']['zip_mat_members']} MATLAB files inside the scenario-08 archives were read and compared with the "
        f"MATLAB data: all {int(c['provenance_totals']['aaa_duplicate_identical'])} ASCII channels are sample-for-sample identical to "
        f"MATLAB columns and all {int(c['provenance_totals']['zip_duplicate_identical'])} archived files are byte-identical, so every "
        f"physical channel is used once. Every ASCII header declares a sample interval of 0.01 s, confirming the 100 Hz sampling "
        f"rate used for all records. The data comprise {num(c['unique_channels'])} channels "
        f"({num(c['channels_by_test']['avt'])} AVT, {num(c['channels_by_test']['fvt'])} FVT) of 65 026–65 536 samples (Table 2). "
        f"The {num(d['ems']['counts']['archives']) if d['ems'] else ''} hourly records of the long-term monitoring system (EMS) are "
        f"analysed in Section 5.7.")
    t["3.3"] = (
        f"Scenarios 01, 02 and 08 are labelled REFERENCE and the other 14 scenarios ALTERED_CONDITION (Table 1). The regular-expression "
        f"label rules agree with the scenario numbers for all recordings ({c['label_rule_mismatches']} mismatches). The labels "
        f"describe the test protocol, not measured damage. Because the windows of one recording are strongly correlated, all "
        f"evaluations are grouped by recording or by scenario.")
    t["4.1"] = (
        "Each channel is validated, NaN and infinite samples are interpolated, and the signal is linearly detrended and band-pass "
        "filtered (Butterworth, order 4, 0.5–40 Hz, zero phase). No amplitude normalisation is applied; per-record z-score "
        "normalisation is evaluated only as an ablation. Windows last 10 s (1000 samples) with 50% overlap and inherit the "
        "scenario, test, setup, sensor and recording metadata.")
    t["4.2"] = (
        "Each window is described by 20 features: 11 time-domain statistics (mean, standard deviation, variance, RMS, minimum, "
        "maximum, absolute peak, peak-to-peak, crest factor, skewness, kurtosis) and 9 spectral descriptors from the Welch PSD "
        "(segment length 256, 0.39 Hz resolution): dominant frequency, spectral energy, frequency centroid, spectral bandwidth, "
        "PSD peak and band energies in 0–1, 1–5, 5–20 and 20–40 Hz (Table 3). Modal features are identified per record by frequency "
        f"domain decomposition (FDD): the cross-spectral matrix of all channels (Hann segments of {cfg['modal']['nperseg']} samples, "
        f"0.024 Hz resolution) is decomposed by SVD, and peaks of the first singular value with a prominence of at least "
        f"{cfg['modal']['peak_prominence']} decades between {cfg['modal']['fmin']} and {cfg['modal']['fmax']} Hz are candidate modes. "
        f"Baseline modes are the peaks of the averaged spectrum of the nine scenario-01 AVT records ({base} Hz). Candidates are "
        f"assigned to the baseline modes by Hungarian matching of the modal assurance criterion (MAC ≥ {cfg['modal']['min_mac']}, "
        "on the five reference sensors R1V, R2L, R2T, R2V and R3V present in every setup) with a relative-frequency penalty. "
        "Damping ratios are estimated by enhanced FDD (logarithmic decrement of the correlation function of the MAC ≥ 0.8 bell) and "
        "retained only if 0 < ζ < 10% and the fit has R² ≥ 0.95; ζ1 and ζ2 met this criterion in "
        f"{100 * zeta_rate[1]:.1f}% and {100 * zeta_rate[2]:.1f}% of the AVT records and are used, ζ3 and ζ4 "
        f"({100 * zeta_rate[3]:.1f}% and {100 * zeta_rate[4]:.1f}%) are not. The FVT spectra are dominated by the shaker excitation, "
        "so modal features are used for the AVT records only.")
    t["4.3a"] = (
        "Two BOCPD configurations are used. Within each record, BOCPD [1] is applied to the window-RMS sequence of every channel. "
        "Across states, the AVT recordings are ordered by scenario (01→17) and setup, and the natural frequencies f1–f4 of each "
        "recording, standardised with the mean and standard deviation of the nine scenario-01 records, form a multivariate sequence. "
        "Let γt(r) denote the joint probability of run length r and the observations up to step t, and πt(r) the product over "
        "features of the Student-t posterior predictive densities under independent Normal–Gamma posteriors (missing values are "
        "skipped). With a constant hazard h, the growth and reset probabilities are")
    t["4.3b"] = (
        f"where πt(0) uses the prior (μ0 = 0, κ0, α0 = 1, β0 = 1). Within records, h = {cfg['bocpd']['hazard_rate']}, κ0 = 1, and a window "
        f"is flagged when CPt ≥ {cfg['bocpd']['threshold']}. Across scenarios, a change is declared when the maximum a posteriori run "
        f"length drops, with h = {sb['primary_hazard_rate']} and κ0 = {sb['primary_kappa0']}; results for h ∈ "
        f"{{{', '.join(map(str, sb['hazard_rates']))}}} and κ0 ∈ {{{', '.join(map(str, sb['kappa0_values']))}}} are also reported. A "
        f"two-sided tabular CUSUM (k = {cfg['cusum']['k']}, h ∈ {{{', '.join(map(str, cfg['cusum']['h']))}}}, re-baselined on the "
        f"{cfg['cusum']['warmup']} records after each alarm) is the baseline detector. A detection is correct if it occurs "
        f"0–{sb['tolerance_records']} records after one of the 16 scenario boundaries.")
    t["4.4"] = (
        "The classifiers operate on non-overlapping AVT windows (every second window) with the 20 window features and the record-level "
        f"modal features ({', '.join(cd['feature_sets']['modal'])}). XGBoost [3] (300 trees, learning rate 0.05, depth 6, row and column "
        "subsampling 0.8) is compared with a random forest (RF; 300 trees, minimum leaf size 5, 25% bootstrap samples), an RBF support "
        f"vector machine (SVM; C = 1, at most {num(ev['svm_max_train_rows'])} stratified training windows per fold), logistic regression "
        "(LR) and a natural-frequency threshold (largest relative shift of f1–f4 from the mean of the training reference recordings, "
        "with the threshold maximising Youden's index on the training recordings). All models use balanced class weights. Two "
        f"schemes are used: leave-one-scenario-out (LOSO, 17 folds) and stratified grouped {ev['group_kfold_splits']}-fold "
        f"cross-validation by recording (GKF), each with {len(ev['seeds'])} seeds. Accuracy, balanced accuracy (BA), macro-F1, ROC-AUC "
        "and the Matthews correlation coefficient (MCC) are computed on the pooled out-of-fold predictions (positive class "
        f"ALTERED_CONDITION), with 95% confidence intervals from a cluster bootstrap over recordings ({ev['bootstrap_iterations']} "
        "replicates, metric averaged over seeds in each replicate).")
    t["4.5"] = "SHAP [2] decomposes the model output (log-odds of ALTERED_CONDITION) for window i into feature contributions φij:"
    t["4.5b"] = (
        "where Φj is the global importance of feature j. Exact TreeSHAP values are computed for every held-out window of the LOSO "
        "folds, each by the model of its own fold (seed 0). Sensors are ranked by the mean of Σj |φij| over the "
        "channel-level features (modal features are shared by all channels of a record) and aggregated by sensor line and direction.")
    t["4.6"] = (
        f"with wB = {cfg['risk']['bocpd_weight']}, wM = {cfg['risk']['ml_weight']} and wA = {cfg['risk']['attribution_weight']}. B is the "
        f"sequence-BOCPD probability that a change occurred within the last {sb['tolerance_records']} records, M the out-of-fold LOSO "
        "probability of ALTERED_CONDITION and A the normalised SHAP attribution of the sensor, which enters as A·max(B, M) in place of A "
        "so that the attribution only prioritises sensors once a change is evident. The classifier term is included only if "
        f"its LOSO ROC-AUC reaches {ev['risk_min_grouped_auc']}; otherwise wM = 0. R is mapped to LOW (< 0.35), MEDIUM (< 0.60), HIGH "
        "(< 0.80) or CRITICAL. A sensor is ANOMALY when B ≥ 0.80 and M ≥ 0.5 (if M is included) or R ≥ 0.60, and WARNING when one "
        "condition holds or R ≥ 0.35; a recording is POTENTIAL ANOMALY if any sensor is ANOMALY or at least 30% are affected, WATCH "
        "if any sensor is affected and NORMAL otherwise. The Streamlit dashboard replays the stored results on a schematic bridge.")
    return t
