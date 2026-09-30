"""Baseline-only novelty detection (no labels): Mahalanobis distance of each AVT recording's natural
frequencies from the scenario-01 baseline records, as available for a structure in service.

Two baselines: scenario 01 alone, and scenarios 01+02 (both measured before any intervention, as for a bridge in
service). For each baseline and feature set:
  (a) ROC-AUC ALTERED vs the held-out REFERENCE scenarios, recording level, cluster-bootstrap CI;
  (b) alarm rates at a chi-square (99%) threshold and at an empirical threshold, the largest leave-one-out distance of
      the baseline recordings (calibration added after the chi-square threshold flagged every recording).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy import stats

from src.evaluation import bootstrap_over_recordings
from src.reproducibility import start_run
from src.utils import ensure_output_dirs, load_config, write_json


def mahalanobis_scores(train: np.ndarray, test: np.ndarray, shrinkage: float = 0.1) -> np.ndarray:
    mu = np.nanmean(train, axis=0)
    cov = np.cov(train, rowvar=False)
    cov = (1 - shrinkage) * cov + shrinkage * np.diag(np.diag(cov))  # shrink off-diagonals (9 baseline records)
    inv = np.linalg.pinv(cov)
    diff = test - mu
    return np.sqrt(np.einsum("ij,jk,ik->i", diff, inv, diff))


def main():
    cfg = load_config()
    paths = ensure_output_dirs(cfg)
    ev = cfg["evaluation"]
    run = start_run("novelty", cfg)
    out = paths["results"] / "novelty"
    out.mkdir(parents=True, exist_ok=True)
    rec = pd.read_csv(paths["processed"] / "recordings.csv", dtype={"scenario": str})
    avt = rec[rec.test == "avt"].copy()
    rows, summary, scores = [], {}, []
    for bname, bsc in (("S01", ["01"]), ("S01+S02", ["01", "02"])):
        for name, cols in (("f1-f3", ["f1", "f2", "f3"]), ("f1-f4", ["f1", "f2", "f3", "f4"])):
            data = avt.dropna(subset=cols)
            base = data[data.scenario.isin(bsc)]
            test = data[~data.scenario.isin(bsc)].copy()
            X = base[cols].to_numpy(float)
            loo = np.array([mahalanobis_scores(np.delete(X, i, axis=0), X[i:i + 1])[0] for i in range(len(X))])
            test["score"] = mahalanobis_scores(X, test[cols].to_numpy(float))
            y = (test["label"] == "ALTERED_CONDITION").astype(int).to_numpy()
            thr = float(np.sqrt(stats.chi2.ppf(0.99, len(cols))))
            thr_emp = float(loo.max())
            res = bootstrap_over_recordings(y, [test["score"].to_numpy()], test["recording"].to_numpy(),
                                            int(ev["bootstrap_iterations"]), int(ev["bootstrap_seed"]), threshold=thr_emp)
            test["alarm"] = test["score"] > thr
            test["alarm_empirical"] = test["score"] > thr_emp
            per = test.groupby(["scenario", "label"]).agg(recordings=("score", "size"), median_distance=("score", "median"),
                                                          alarm_rate=("alarm", "mean"),
                                                          alarm_rate_empirical=("alarm_empirical", "mean")).reset_index()
            per.insert(0, "features", name)
            per.insert(0, "baseline", bname)
            rows.append(per)
            key = name if bname == "S01" else f"{bname}:{name}"
            summary[key] = {"baseline": bname, "baseline_records": int(len(base)), "test_records": int(len(test)),
                            "reference_test_scenarios": sorted(test.loc[y == 0, "scenario"].unique().tolist()),
                            "threshold_chi2_99": thr, "threshold_empirical_loo_max": thr_emp,
                            "roc_auc_altered_vs_reference": res.point["roc_auc"],
                            "roc_auc_ci": [res.ci_low["roc_auc"], res.ci_high["roc_auc"]],
                            "alarm_rate_reference": float(test.loc[y == 0, "alarm"].mean()),
                            "alarm_rate_altered": float(test.loc[y == 1, "alarm"].mean()),
                            "alarm_rate_reference_empirical": float(test.loc[y == 0, "alarm_empirical"].mean()),
                            "alarm_rate_altered_empirical": float(test.loc[y == 1, "alarm_empirical"].mean()),
                            "balanced_accuracy_empirical": res.point["balanced_accuracy"],
                            "balanced_accuracy_empirical_ci": [res.ci_low["balanced_accuracy"], res.ci_high["balanced_accuracy"]]}
            b = base[["recording", "scenario", "label"]].copy()
            b["score"], b["role"] = loo, "baseline (leave-one-out)"
            t = test[["recording", "scenario", "label", "score"]].copy()
            t["role"] = "test"
            sc = pd.concat([b, t])
            sc["baseline"], sc["features"], sc["threshold_empirical"] = bname, name, thr_emp
            scores.append(sc)
    pd.concat(scores, ignore_index=True).to_csv(out / "novelty_scores.csv", index=False)
    table = pd.concat(rows, ignore_index=True)
    table.to_csv(out / "novelty_per_scenario.csv", index=False)
    write_json(out / "novelty_summary.json", summary)
    print(summary)
    print(table.round(3).to_string())
    run.finish([out / "novelty_per_scenario.csv", out / "novelty_summary.json", out / "novelty_scores.csv"])


if __name__ == "__main__":
    main()
