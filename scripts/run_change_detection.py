"""Stage 2: change detection.

(a) Within-record BOCPD on the window RMS of every channel (all 306 recordings).
(b) Cross-scenario BOCPD on the AVT recording sequence ordered 01 -> 17 (setup 01 -> 09) using the
    per-record natural frequencies, compared with a two-sided CUSUM, against the 16 true
    scenario boundaries (precision, recall, detection delay).  Hazard / h sensitivity is saved.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.bocpd import detection_metrics, run_bocpd, run_cusum, run_sequence_bocpd, standardise_by_baseline
from src.reproducibility import start_run
from src.utils import ensure_output_dirs, load_config, write_json


def sequence_table(recordings: pd.DataFrame, test: str, features: list[str]) -> pd.DataFrame:
    seq = recordings[recordings["test"] == test].sort_values(["scenario", "setup"]).reset_index(drop=True)
    return seq[["recording", "scenario", "setup", "label"] + features]


def boundaries_of(seq: pd.DataFrame) -> list[int]:
    s = seq["scenario"].to_numpy()
    return [i for i in range(1, len(s)) if s[i] != s[i - 1]]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--skip-within", action="store_true")
    a = p.parse_args()
    cfg = load_config(a.config)
    paths = ensure_output_dirs(cfg)
    run = start_run("change_detection", cfg)
    out_dir = paths["results"] / "change_detection"
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = []

    # (a) within-record BOCPD -------------------------------------------------------------
    if not a.skip_within:
        bc = cfg["bocpd"]
        cols = ["scenario", "test", "setup", "sensor", "recording", "window_id", "start_time", "label", bc["feature"]]
        feats = pd.read_parquet(paths["processed"] / "features.parquet", columns=cols)
        within = run_bocpd(feats, bc["feature"], bc["hazard_rate"], bc["threshold"], bc.get("max_run_length", 500))
        within = within.merge(feats[["recording", "sensor", "window_id", "scenario", "test", "label"]],
                              on=["recording", "sensor", "window_id"], how="left", validate="one_to_one")
        within.to_parquet(paths["processed"] / "bocpd_within_record.parquet", index=False)
        # legacy CSV for the dashboard (AVT only to keep it manageable)
        within[within["test"] == "avt"].to_csv(paths["processed"] / "bocpd_results.csv", index=False)
        summ = within.groupby(["scenario", "label", "test"]).agg(
            windows=("change_probability", "size"), sequences=("recording", lambda r: 0),
            mean_cp=("change_probability", "mean"), max_cp=("change_probability", "max"),
            windows_cp_ge_05=("change_probability", lambda v: int((v >= 0.5).sum())),
            detections=("change_detected", "sum")).reset_index()
        seqs = within.groupby(["scenario", "test"])[["recording", "sensor"]].apply(lambda g: g.drop_duplicates().shape[0])
        summ["sequences"] = [int(seqs.loc[(s, t)]) for s, t in zip(summ["scenario"], summ["test"])]
        summ.to_csv(out_dir / "within_record_bocpd_by_scenario.csv", index=False)
        overall = {"windows": int(len(within)), "sequences": int(within.groupby(["recording", "sensor"]).ngroups),
                   "threshold": bc["threshold"], "hazard_rate": bc["hazard_rate"],
                   "detections": int(within["change_detected"].sum()),
                   "windows_cp_ge_05": int((within["change_probability"] >= 0.5).sum()),
                   "max_cp": float(within["change_probability"].max()), "mean_cp": float(within["change_probability"].mean())}
        write_json(out_dir / "within_record_bocpd_summary.json", overall)
        outputs += [paths["processed"] / "bocpd_within_record.parquet", out_dir / "within_record_bocpd_by_scenario.csv",
                    out_dir / "within_record_bocpd_summary.json"]
        print("within-record:", overall, flush=True)
        del feats, within

    # (b) cross-scenario sequence ---------------------------------------------------------
    sc = cfg["sequence_bocpd"]
    cu = cfg["cusum"]
    rec = pd.read_csv(paths["processed"] / "recordings.csv", dtype={"scenario": str, "setup": str})
    feats = [f for f in sc["features"] if f in rec]
    seq = sequence_table(rec, "avt", feats)
    b = boundaries_of(seq)
    z, mu, sd = standardise_by_baseline(seq[feats].to_numpy(float), int(sc["baseline_records"]))
    rows, series = [], {"recording": seq["recording"], "scenario": seq["scenario"], "setup": seq["setup"]}
    for f, col in zip(feats, z.T):
        series[f"z_{f}"] = col
    k_primary = float(sc.get("primary_kappa0", 1.0))
    for k0 in sc.get("kappa0_values", [1.0]):
        for h in sc["hazard_rates"]:
            r = run_sequence_bocpd(z, float(h), recent=int(sc["tolerance_records"]), kappa0=float(k0))
            m = detection_metrics(r["detections"], b, int(sc["tolerance_records"]))
            primary = float(h) == float(sc["primary_hazard_rate"]) and float(k0) == k_primary
            rows.append({"method": "BOCPD", "parameter": f"hazard={h},kappa0={k0}", "primary": primary,
                         **m, "detection_list": json.dumps(r["detections"])})
            if float(k0) == k_primary:
                series[f"bocpd_cp_h{h}"] = r["change_probability"]
                series[f"bocpd_rmap_h{h}"] = r["map_run_length"]
                series[f"bocpd_recent_h{h}"] = r["recent_change_probability"]  # P(r_t <= tolerance), risk score
    for h in cu["h"]:
        r = run_cusum(z, float(cu["k"]), float(h), int(cu["warmup"]))
        m = detection_metrics(r["detections"], b, int(sc["tolerance_records"]))
        rows.append({"method": "CUSUM", "parameter": f"k={cu['k']},h={h}", "primary": float(h) == float(cu["primary_h"]),
                     **m, "detection_list": json.dumps(r["detections"])})
        series[f"cusum_stat_h{h}"] = r["statistic"]
    # per-feature (univariate) BOCPD at the primary hazard, to show which modes carry the changes
    for j, f in enumerate(feats):
        r = run_sequence_bocpd(z[:, [j]], float(sc["primary_hazard_rate"]), kappa0=k_primary)
        m = detection_metrics(r["detections"], b, int(sc["tolerance_records"]))
        rows.append({"method": f"BOCPD[{f}]", "parameter": f"hazard={sc['primary_hazard_rate']}", "primary": False,
                     **m, "detection_list": json.dumps(r["detections"])})
    table = pd.DataFrame(rows)
    table["matched_boundaries"] = table["matched_boundaries"].apply(lambda v: json.dumps([int(x) for x in v]))
    table.to_csv(out_dir / "sequence_detection_comparison.csv", index=False)
    pd.DataFrame(series).to_csv(out_dir / "sequence_series.csv", index=False)
    per_boundary = []
    primary_b = next(r for r in rows if r["method"] == "BOCPD" and r["primary"])
    primary_c = next(r for r in rows if r["method"] == "CUSUM" and r["primary"])
    for bi in b:
        rowb = {"boundary_index": bi, "from_scenario": seq["scenario"][bi - 1], "to_scenario": seq["scenario"][bi]}
        for name, r in (("bocpd", primary_b), ("cusum", primary_c)):
            dets = json.loads(r["detection_list"])
            hit = [d for d in dets if bi <= d["t"] <= bi + int(sc["tolerance_records"])]
            rowb[f"{name}_detected"] = bool(hit)
            rowb[f"{name}_delay"] = hit[0]["t"] - bi if hit else np.nan
        for f in feats:
            prev = seq.loc[seq["scenario"] == seq["scenario"][bi - 1], f].mean()
            nxt = seq.loc[seq["scenario"] == seq["scenario"][bi], f].mean()
            rowb[f"delta_{f}_pct"] = 100 * (nxt - prev) / prev
        per_boundary.append(rowb)
    pd.DataFrame(per_boundary).to_csv(out_dir / "sequence_per_boundary.csv", index=False)
    write_json(out_dir / "sequence_setup.json", {"test": "avt", "features": feats, "n_records": len(seq),
                                                 "boundaries": b, "baseline_mean": mu, "baseline_std": sd,
                                                 "missing_values_per_feature": seq[feats].isna().sum().to_dict()})
    outputs += [out_dir / n for n in ("sequence_detection_comparison.csv", "sequence_series.csv",
                                      "sequence_per_boundary.csv", "sequence_setup.json")]
    print(table[["method", "parameter", "detections", "true_positives", "precision", "recall", "f1", "mean_delay"]].to_string())
    run.finish(outputs)
    print("run:", run.run_id)


if __name__ == "__main__":
    main()
