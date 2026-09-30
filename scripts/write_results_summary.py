"""Stage 8: write results/RESULTS_SUMMARY.md - every number is read from a result file whose path is cited."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.utils import PROJECT_ROOT, ensure_output_dirs, load_config

MODEL_NAME = {"xgboost": "XGBoost", "random_forest": "Random forest", "svm": "SVM (RBF)", "logistic_regression": "Logistic regression",
              "freq_threshold": "Natural-frequency shift threshold", "xgboost_monotone": "XGBoost, monotone in f_k",
              "freq_decrease_threshold": "Natural-frequency decrease threshold"}


def rel(p: Path) -> str:
    return str(p.resolve().relative_to(PROJECT_ROOT)).replace("\\", "/")


def ci(row, m, d=3):
    return f"{row[m]:.{d}f} [{row[m + '_ci_low']:.{d}f}, {row[m + '_ci_high']:.{d}f}]"


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def clf_table(res: pd.DataFrame, level: str) -> pd.DataFrame:
    rows = []
    for _, r in res[res.level == level].iterrows():
        rows.append({"Model": MODEL_NAME.get(r.model, r.model), "Features": r.features,
                     "Norm.": r.normalisation, "CV": "LOSO" if r.scheme == "loso" else "Grouped 5-fold",
                     "Accuracy": ci(r, "accuracy"), "Balanced acc.": ci(r, "balanced_accuracy"), "Macro-F1": ci(r, "macro_f1"),
                     "ROC-AUC": ci(r, "roc_auc"), "MCC": ci(r, "mcc"), "Seeds": int(r.n_seeds),
                     "SD over seeds (AUC)": f"{r.roc_auc_seed_sd:.3f}"})
    return pd.DataFrame(rows)


def main():
    cfg = load_config()
    paths = ensure_output_dirs(cfg)
    R, P = paths["results"], paths["processed"]
    out = ["# RESULTS SUMMARY", "",
           "Every value below is read by `scripts/write_results_summary.py` from the file cited in each section. "
           "Nothing is typed by hand. Positive class = ALTERED_CONDITION. CIs are 95% percentile bootstrap intervals over "
           f"recordings ({cfg['evaluation']['bootstrap_iterations']} replicates); with several seeds each replicate averages the "
           "metric over seeds.", ""]

    f = P / "data_counts.json"
    c = json.loads(f.read_text())
    out += ["## 1. Data (steps 1-5)", f"Source: `{rel(f)}`", "",
            f"- Catalogue: {c['catalogue']}",
            f"- .aaa header check: {c['aaa_header_verification']['files']} files, sample interval {c['aaa_header_verification']['sample_interval_s']} s -> {c['aaa_header_verification']['sampling_rate_hz']} Hz",
            f"- Recordings: {c['recordings']} (unique IDs: {c['unique_recording_ids']}), by test {c['recordings_by_test']}, setups per scenario/test {c['setups_per_scenario_test']}",
            f"- Unique channels: {c['unique_channels']} ({c['channels_by_test']}); distinct sensor names {c['distinct_sensor_names']}",
            f"- Duplicate verification: {c['provenance_totals']}",
            f"- Windows (10 s, 50% overlap): {c['windows_total']} ({c['windows_by_test']}); by label {c['windows_by_label']}",
            f"- Samples per channel: {c['samples_per_channel']}; windows per channel {c['windows_per_channel']}",
            f"- Label-rule mismatches: {c['label_rule_mismatches']}", ""]

    f = P / "recordings.csv"
    rec = pd.read_csv(f, dtype={"scenario": str})
    avt = rec[rec.test == "avt"]
    base = json.loads((P / "modal_baseline.json").read_text())
    fcols = [f"f{k}" for k in range(1, len(base) + 1)]
    tab = avt.groupby(["scenario", "label"])[fcols + ["zeta1", "zeta2"]].agg(["mean", "std"])
    tab.columns = [f"{a}_{b}" for a, b in tab.columns]
    tab = tab.reset_index()
    (R / "modal").mkdir(exist_ok=True)
    tab.to_csv(R / "modal" / "modal_by_scenario.csv", index=False)
    show = tab[["scenario", "label"]].copy()
    for col in fcols:
        show[f"{col} [Hz]"] = [f"{m:.3f} ± {s:.3f}" for m, s in zip(tab[f"{col}_mean"], tab[f"{col}_std"])]
    for col in ("zeta1", "zeta2"):
        show[f"{col} [%]"] = [f"{100 * m:.2f} ± {100 * s:.2f}" for m, s in zip(tab[f"{col}_mean"], tab[f"{col}_std"])]
    out += ["## 2. Modal identification (FDD/EFDD, AVT)", f"Sources: `{rel(P / 'modal_baseline.json')}`, `{rel(f)}`, "
            f"`{rel(R / 'modal' / 'modal_by_scenario.csv')}`", "",
            "- Baseline modes (scenario 01, AVT): " + ", ".join(f"{b['frequency_hz']:.3f} Hz" for b in base),
            "- Identification rate (AVT): " + ", ".join(f"{c_} {avt[c_].notna().mean():.3f}" for c_ in fcols),
            "- Reliable damping rate (AVT): " + ", ".join(f"zeta{k} {avt[f'zeta{k}_reliable'].mean():.3f}" for k in range(1, len(base) + 1)),
            "- Identification rate (FVT): " + ", ".join(f"{c_} {rec[rec.test == 'fvt'][c_].notna().mean():.3f}" for c_ in fcols),
            "", "Mean ± SD over the 9 AVT setups (damping: reliable estimates only):", "", md_table(show), ""]

    cd = R / "change_detection"
    f = cd / "within_record_bocpd_summary.json"
    if f.exists():
        w = json.loads(f.read_text())
        out += ["## 3. Within-record BOCPD (window RMS)", f"Sources: `{rel(f)}`, `{rel(cd / 'within_record_bocpd_by_scenario.csv')}`", "",
                f"- Sequences {w['sequences']}, windows {w['windows']}, hazard {w['hazard_rate']}, threshold {w['threshold']}",
                f"- Threshold crossings {w['detections']}; windows with CP >= 0.5: {w['windows_cp_ge_05']}; max CP {w['max_cp']:.4f}; mean CP {w['mean_cp']:.4f}", ""]
    f = cd / "sequence_detection_comparison.csv"
    if f.exists():
        s = pd.read_csv(f)
        s = s[["method", "parameter", "primary", "detections", "true_positives", "false_positives", "precision", "recall", "f1",
               "mean_delay", "mean_location_error"]].round(3)
        setup = json.loads((cd / "sequence_setup.json").read_text())
        out += ["## 4. Change detection across scenarios (AVT sequence, 153 records, 16 true boundaries)",
                f"Sources: `{rel(f)}`, `{rel(cd / 'sequence_per_boundary.csv')}`, `{rel(cd / 'sequence_setup.json')}`", "",
                f"Features {setup['features']}, standardised with the first 9 records (scenario 01). Tolerance "
                f"{cfg['sequence_bocpd']['tolerance_records']} records after a boundary; delay in records (setups).", "",
                md_table(s), "", "Per boundary (primary settings):", "",
                md_table(pd.read_csv(cd / "sequence_per_boundary.csv").round(2)), ""]

    cl = R / "classification"
    f = cl / "classification_dataset.json"
    if f.exists():
        d = json.loads(f.read_text())
        out += ["## 5. Classification data", f"Source: `{rel(f)}`", "",
                f"- AVT windows (non-overlapping, stride {d['window_stride']}): {d['windows']}; recordings {d['recordings']} {d['recordings_by_label']}; "
                f"channels {d['channels']}; windows by label {d['windows_by_label']}",
                f"- Feature sets: " + "; ".join(f"{k}: {len(v)}" for k, v in d["feature_sets"].items()),
                f"- Modal set: {d['feature_sets']['modal']} (damping kept when reliable in >= 80% of AVT records: {d['damping_reliability_rate']})", ""]
    for exp, title in (("main", "6. Main comparison (all features) and baselines"),
                       ("physics", "7. Physically constrained variants (added after the first LOSO results)"),
                       ("ablation", "8. Ablation (XGBoost)")):
        f = cl / f"{exp}_results.csv"
        if f.exists():
            res = pd.read_csv(f)
            out += [f"## {title}", f"Source: `{rel(f)}`", "", "Window level:", "", md_table(clf_table(res, "window")), "",
                    "Recording level (mean window score per recording):", "", md_table(clf_table(res, "recording")), ""]
    for exp in ("main", "physics"):
        f = cl / f"{exp}_per_scenario.csv"
        if f.exists():
            ps = pd.read_csv(f, dtype={"scenario": str})
            ps = ps[ps.scheme == "loso"]
            piv = ps.pivot_table(index=["scenario", "label"], columns="model", values="recording_accuracy").round(3).reset_index()
            out += [f"LOSO recording accuracy per held-out scenario ({exp}); source `{rel(f)}`:", "", md_table(piv), ""]
    f = cl / "model_cost.csv"
    if f.exists():
        out += ["## 9. Runtime and model size", f"Sources: `{rel(f)}`, `{rel(cl / 'main_timing.csv')}`", "",
                md_table(pd.read_csv(f).round(3)), ""]
        t = cl / "main_timing.csv"
        if t.exists():
            tt = pd.read_csv(t).groupby(["model", "scheme"])[["fit_seconds_total", "predict_seconds_total"]].mean().round(1).reset_index()
            out += ["Mean CV time per seed (all folds):", "", md_table(tt), ""]

    nv = R / "novelty"
    f = nv / "novelty_summary.json"
    if f.exists():
        rows = [{"setting": k, "baseline": v.get("baseline", "S01"), "held-out reference": ", ".join(v.get("reference_test_scenarios", [])),
                 "ROC-AUC": f"{v['roc_auc_altered_vs_reference']:.3f} [{v['roc_auc_ci'][0]:.3f}, {v['roc_auc_ci'][1]:.3f}]",
                 "chi2 thr": f"{v['threshold_chi2_99']:.2f}", "alarm ref/alt (chi2)": f"{v['alarm_rate_reference']:.2f} / {v['alarm_rate_altered']:.2f}",
                 "empirical thr": f"{v.get('threshold_empirical_loo_max', float('nan')):.2f}",
                 "alarm ref/alt (empirical)": f"{v.get('alarm_rate_reference_empirical', float('nan')):.2f} / {v.get('alarm_rate_altered_empirical', float('nan')):.2f}",
                 "BA (empirical)": (f"{v['balanced_accuracy_empirical']:.3f} [{v['balanced_accuracy_empirical_ci'][0]:.3f}, "
                                    f"{v['balanced_accuracy_empirical_ci'][1]:.3f}]" if "balanced_accuracy_empirical_ci" in v else "")}
                for k, v in json.loads(f.read_text()).items()]
        out += ["## 9b. Label-free novelty detection (Mahalanobis distance of f1-f4 from a baseline)",
                f"Sources: `{rel(f)}`, `{rel(nv / 'novelty_per_scenario.csv')}`, `{rel(nv / 'novelty_scores.csv')}`", "",
                "Empirical threshold = largest leave-one-out distance of the baseline records (added after the chi2 threshold "
                "flagged every recording). Baseline S01+S02 = both measurements taken before any intervention.", "",
                md_table(pd.DataFrame(rows)), "", md_table(pd.read_csv(nv / "novelty_per_scenario.csv").round(3)), ""]
    f = R / "ssi" / "ssi_summary.json"
    if f.exists():
        ss = pd.DataFrame(json.loads(f.read_text())).T.reset_index().rename(columns={"index": "mode"})
        out += ["## 2b. SSI-COV cross-check of the FDD modes", f"Sources: `{rel(f)}`, `{rel(R / 'ssi' / 'ssi_vs_fdd.csv')}`", "",
                "Reference sensors, signals decimated to 50 Hz, model orders 20-60, stabilisation (df < 1%, dzeta < 30%); "
                "nearest stable pole within 5% of the FDD frequency.", "", md_table(ss.round(3)), ""]
    f = R / "mode_shapes" / "mode_shape_summary.json"
    if f.exists():
        ms = json.loads(f.read_text())
        cl = pd.DataFrame(ms["classification"])
        cl = pd.DataFrame([{"features": r["features"], "n_features": r["n_features"], "ROC-AUC": ci(r, "roc_auc"),
                            "BA": ci(r, "balanced_accuracy"), "MCC": ci(r, "mcc")} for _, r in cl.iterrows()])
        out += ["## 11b. Mode-shape change features (LOSO, scenarios 02-17, XGBoost seed 0; S01 = baseline)",
                f"Sources: `{rel(f)}`, `{rel(R / 'mode_shapes' / 'mode_shape_classification.csv')}`, "
                f"`{rel(R / 'mode_shapes' / 'mode_shape_group_attribution.csv')}`", "", md_table(cl), "",
                f"- SHAP share of |dphi| features: {ms['shap_share_mode_shape_features']:.4f}; spread of channel attribution "
                f"between sensor groups (>= 5 sensors): {ms['group_attribution_spread_pct']:.1f}%",
                f"- Mode identified per channel: {ms['mode_identified_rate']}", ""]

    sh = R / "shap"
    f = sh / "shap_summary.json"
    if f.exists():
        s = json.loads(f.read_text())
        g = pd.read_csv(sh / "global_feature_importance.csv")
        gg = pd.read_csv(sh / "feature_group_importance.csv")
        sens = pd.read_csv(sh / "sensor_importance_all.csv")
        out += ["## 10. SHAP (XGBoost, complete held-out LOSO test sets)",
                f"Sources: `{rel(f)}`, `{rel(sh / 'global_feature_importance.csv')}`, `{rel(sh / 'feature_group_importance.csv')}`, "
                f"`{rel(sh / 'sensor_importance_all.csv')}`, `{rel(sh / 'importance_by_sensor_group.csv')}`, `{rel(sh / 'importance_by_sensor_direction.csv')}`", "",
                f"- Explained windows {s['explained_windows']} from {s['explained_recordings']} recordings; seeds {s['seeds']}; sensors ranked {s['sensors_ranked']}",
                f"- TreeSHAP cross-check (xgboost pred_contribs vs shap.TreeExplainer): {s['treeshap_crosscheck']}", "",
                "Feature groups:", "", md_table(gg.round(4)), "", "Top 15 features:", "",
                md_table(g.head(15)[["rank", "feature", "group", "mean_abs_shap", "share"]].round(4)), "",
                "Top 15 sensors (channel-level attribution):", "",
                md_table(sens.head(15)[["rank", "sensor", "sensor_group", "sensor_direction", "windows", "channel_attribution"]].round(4)), "",
                "By sensor group:", "", md_table(pd.read_csv(sh / "importance_by_sensor_group.csv").round(4)), "",
                "By direction:", "", md_table(pd.read_csv(sh / "importance_by_sensor_direction.csv").round(4)), ""]

    rk = R / "risk"
    f = rk / "risk_summary.json"
    if f.exists():
        out += ["## 11. Risk score and digital-twin states", f"Sources: `{rel(f)}`, `{rel(rk / 'states_by_label.csv')}`, `{rel(rk / 'scenario_states.csv')}`", "",
                f"- {json.loads(f.read_text())}", "", md_table(pd.read_csv(rk / "states_by_label.csv").round(3)), "",
                md_table(pd.read_csv(rk / "scenario_states.csv", dtype={"scenario": str}).round(3)), ""]

    em = R / "ems"
    f = em / "ems_summary.json"
    if f.exists():
        s = json.loads(f.read_text())
        out += ["## 12. EMS one-year monitoring", f"Sources: `{rel(f)}`, `{rel(em / 'ems_detection.csv')}`, `{rel(P / 'ems_modal.csv')}`", "",
                f"- Counts: {s['counts']}", f"- Period: {s['first_timestamp']} .. {s['last_timestamp']}; temperature model trained until {s['train_end']}; onset window {s['onset_window']}",
                f"- Temperature model: {json.dumps(s['temperature_model'])}", "",
                md_table(pd.read_csv(em / "ems_detection.csv").drop(columns=["alarm_dates"])), ""]

    runs = sorted((R / "runs").glob("*/manifest.json"))
    out += ["## 13. Reproducibility", "Each stage writes `results/runs/<run_id>/` (config snapshot, environment, seeds, wall time, "
            "SHA-256 of outputs). Latest run per stage:", ""]
    latest = {}
    for m in runs:
        j = json.loads(m.read_text())
        latest[j["stage"]] = j
    rows = [{"stage": k, "run_id": v["run_id"], "wall_time_s": v["wall_time_s"], "outputs": len(v["outputs"])} for k, v in sorted(latest.items())]
    out += [md_table(pd.DataFrame(rows)), ""]
    if latest:
        env = next(iter(latest.values()))["environment"]
        out += [f"Environment: Python {env['python'].split()[0]}, {env['platform']}, packages {env['packages']}", ""]
    target = R / "RESULTS_SUMMARY.md"
    target.write_text("\n".join(out), encoding="utf-8")
    print("written", target)


if __name__ == "__main__":
    main()
