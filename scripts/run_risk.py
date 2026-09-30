"""Stage 5: fused risk score and digital-twin states for every AVT recording.

B = sequence-BOCPD recent-change probability P(r_t <= tolerance) of the recording (primary hazard)
M = seed-averaged out-of-fold LOSO XGBoost P(ALTERED) of each sensor's windows in the recording;
    the term is included only if the LOSO window-level ROC-AUC >= evaluation.risk_min_grouped_auc
A = SHAP channel attribution of the sensor divided by the maximum over sensors; it enters as A * max(B, M) so that the
    attribution prioritises sensors only when a change is evident (risk.attribution_requires_change)
States follow the dashboard rules (src.risk_engine.sensor_state / overall_state).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.evaluation import load_table
from src.reproducibility import start_run
from src.risk_engine import calculate_risk, gated_risk_config, overall_state, sensor_state
from src.utils import ensure_output_dirs, load_config, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="config.yaml")
    a = p.parse_args()
    cfg = load_config(a.config)
    ev = cfg["evaluation"]
    paths = ensure_output_dirs(cfg)
    run = start_run("risk", cfg, seeds=ev["seeds"])
    out_dir = paths["results"] / "risk"
    out_dir.mkdir(parents=True, exist_ok=True)
    n_modes = len(json.loads((paths["processed"] / "modal_baseline.json").read_text()))
    df = load_table(paths["processed"] / "features.parquet", paths["processed"] / "recordings.csv", "avt",
                    int(ev["window_stride"]), n_modes)
    scores = [np.load(paths["processed"] / "oof" / f"main__xgboost__all__none__loso__s{s}.npz")["score"] for s in ev["seeds"]]
    df["p_altered"] = np.mean(scores, axis=0)
    res = pd.read_csv(paths["results"] / "classification" / "main_results.csv")
    auc = {sch: float(res.query("model=='xgboost' and scheme==@sch and level=='window'")["roc_auc"].iloc[0])
           for sch in ("loso", "group_kfold")}
    risk_cfg, include_ml = gated_risk_config(cfg["risk"], auc["loso"], float(ev["risk_min_grouped_auc"]))
    h = cfg["sequence_bocpd"]["primary_hazard_rate"]
    series = pd.read_csv(paths["results"] / "change_detection" / "sequence_series.csv", dtype={"scenario": str})
    b_map = dict(zip(series["recording"], series[f"bocpd_recent_h{h}"]))
    imp = pd.read_csv(paths["results"] / "shap" / "sensor_importance_all.csv")
    a_map = dict(zip(imp["sensor"], imp["channel_attribution"] / imp["channel_attribution"].max()))
    thr = float(cfg["bocpd"]["threshold"])
    sens = df.groupby(["recording", "scenario", "label", "sensor"], as_index=False)["p_altered"].mean()
    rows = []
    for variant, rc, use_ml in (("gated", risk_cfg, include_ml), ("ungated", cfg["risk"], True)):
        for _, r in sens.iterrows():
            B, M, A = b_map[r.recording], r.p_altered, a_map.get(r.sensor, 0.0)
            risk = calculate_risk(B, M if use_ml else None, A, rc)
            rows.append({"variant": variant, "recording": r.recording, "scenario": r.scenario, "label": r.label,
                         "sensor": r.sensor, "B": B, "M": M, "A": A, "risk_score": risk["risk_score"],
                         "risk_level": risk["risk_level"], "state": sensor_state(B, M, risk["risk_score"], thr, use_ml)})
    sensor_df = pd.DataFrame(rows)
    sensor_df.to_csv(out_dir / "sensor_risk.csv", index=False)
    rec = sensor_df.groupby(["variant", "recording", "scenario", "label"]).agg(
        B=("B", "first"), mean_M=("M", "mean"), mean_risk=("risk_score", "mean"), max_risk=("risk_score", "max"),
        sensors=("sensor", "size"), warning=("state", lambda s: int((s == "WARNING").sum())),
        anomaly=("state", lambda s: int((s == "ANOMALY").sum())), state=("state", lambda s: overall_state(list(s)))).reset_index()
    rec["health_score"] = (100 * (1 - rec["mean_risk"])).round().astype(int)
    rec.to_csv(out_dir / "recording_states.csv", index=False)
    scen = rec.groupby(["variant", "scenario", "label"]).agg(
        recordings=("recording", "size"), mean_risk=("mean_risk", "mean"), mean_health=("health_score", "mean"),
        normal=("state", lambda s: int((s == "NORMAL").sum())), watch=("state", lambda s: int((s == "WATCH").sum())),
        potential_anomaly=("state", lambda s: int((s == "POTENTIAL ANOMALY").sum()))).reset_index()
    scen.to_csv(out_dir / "scenario_states.csv", index=False)
    by_label = rec.groupby(["variant", "label"]).agg(
        recordings=("recording", "size"), mean_risk=("mean_risk", "mean"),
        flagged=("state", lambda s: int((s != "NORMAL").sum())),
        potential_anomaly=("state", lambda s: int((s == "POTENTIAL ANOMALY").sum()))).reset_index()
    by_label["flagged_rate"] = by_label["flagged"] / by_label["recordings"]
    by_label.to_csv(out_dir / "states_by_label.csv", index=False)
    summary = {"classifier_auc": auc, "min_auc": ev["risk_min_grouped_auc"], "classifier_term_included": include_ml,
               "weights_gated": {k: risk_cfg[f"{k}_weight"] for k in ("bocpd", "ml", "attribution")},
               "bocpd_term": f"P(r_t <= {cfg['sequence_bocpd']['tolerance_records']}) at hazard {h}"}
    write_json(out_dir / "risk_summary.json", summary)
    print(summary, "\n", by_label.to_string(), "\n", scen[scen.variant == "gated"].to_string())
    run.finish([out_dir / f for f in ("sensor_risk.csv", "recording_states.csv", "scenario_states.csv",
                                      "states_by_label.csv", "risk_summary.json")])
    print("run:", run.run_id)


if __name__ == "__main__":
    main()
