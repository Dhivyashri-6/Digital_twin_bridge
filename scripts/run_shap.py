"""Stage 4: exact TreeSHAP for XGBoost on the complete held-out LOSO test sets.

Each window is explained by the model of the fold in which it was held out, for every seed;
mean |SHAP| values are averaged over seeds.  Attributions are ranked for all features, all
sensors (channel-level features only, since modal features are shared by all channels of a
recording) and aggregated by sensor group, direction and measurement point.
XGBoost's ``pred_contribs`` implements TreeSHAP; it is cross-checked against shap.TreeExplainer.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import xgboost as xgb

from src.evaluation import balanced_weights, feature_sets, load_table, make_model, splits
from src.reproducibility import start_run
from src.utils import ensure_output_dirs, load_config, write_json

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_classification import reliable_modal_columns  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--scheme", default="loso")
    p.add_argument("--seeds", nargs="*", type=int)
    a = p.parse_args()
    cfg = load_config(a.config)
    ev = cfg["evaluation"]
    seeds = a.seeds if a.seeds else [int(s) for s in ev["seeds"]]
    paths = ensure_output_dirs(cfg)
    run = start_run("shap", cfg, seeds=seeds)
    out_dir = paths["results"] / "shap"
    n_modes = len(json.loads((paths["processed"] / "modal_baseline.json").read_text()))
    modal_cols, _ = reliable_modal_columns(paths["processed"] / "recordings.csv", n_modes)
    cols = feature_sets(n_modes)["time+spectral"] + modal_cols
    channel_cols = feature_sets(n_modes)["time+spectral"]
    df = load_table(paths["processed"] / "features.parquet", paths["processed"] / "recordings.csv", "avt",
                    int(ev["window_stride"]), n_modes)
    X, y = df[cols].to_numpy(float), df["y"].to_numpy()
    abs_sum = np.zeros((len(df), len(cols)))
    signed_sum = np.zeros((len(df), len(cols)))
    check = {}
    for seed in seeds:
        for k, (tr, te) in enumerate(splits(df, a.scheme, seed, int(ev["group_kfold_splits"]))):
            model = make_model("xgboost", seed, cols, cfg["model"])
            model.fit(X[tr], y[tr], sample_weight=balanced_weights(y[tr]))
            contrib = model.get_booster().predict(xgb.DMatrix(X[te], feature_names=None), pred_contribs=True)[:, :-1]
            abs_sum[te] += np.abs(contrib)
            signed_sum[te] += contrib
            if seed == seeds[0] and k == 0:
                import shap
                sub = X[te][:500]
                sv = shap.TreeExplainer(model).shap_values(sub)
                check = {"rows": len(sub), "max_abs_difference": float(np.max(np.abs(np.asarray(sv) - contrib[:500])))}
        print("seed", seed, "done", flush=True)
    abs_mean = abs_sum / len(seeds)
    signed_mean = signed_sum / len(seeds)
    glob = pd.DataFrame({"feature": cols, "mean_abs_shap": abs_mean.mean(axis=0),
                         "mean_shap_reference_windows": signed_mean[y == 0].mean(axis=0),
                         "mean_shap_altered_windows": signed_mean[y == 1].mean(axis=0),
                         "group": ["modal" if c in modal_cols else "time" if c in feature_sets(n_modes)["time"] else "spectral"
                                   for c in cols]}).sort_values("mean_abs_shap", ascending=False)
    glob["rank"] = range(1, len(glob) + 1)
    glob["share"] = glob["mean_abs_shap"] / glob["mean_abs_shap"].sum()
    glob.to_csv(out_dir / "global_feature_importance.csv", index=False)
    grp = glob.groupby("group")["mean_abs_shap"].sum().rename("sum_mean_abs_shap").reset_index()
    grp["share"] = grp["sum_mean_abs_shap"] / grp["sum_mean_abs_shap"].sum()
    grp.to_csv(out_dir / "feature_group_importance.csv", index=False)

    ch_idx = [cols.index(c) for c in channel_cols]
    per_row = pd.DataFrame({"sensor": df["sensor"], "sensor_group": df["sensor_group"], "sensor_direction": df["sensor_direction"],
                            "sensor_point": df["sensor_point"], "scenario": df["scenario"],
                            "channel_attribution": abs_mean[:, ch_idx].sum(axis=1),
                            "total_attribution": abs_mean.sum(axis=1)})
    sens = per_row.groupby(["sensor", "sensor_group", "sensor_direction", "sensor_point"]).agg(
        windows=("channel_attribution", "size"), channel_attribution=("channel_attribution", "mean"),
        total_attribution=("total_attribution", "mean")).reset_index().sort_values("channel_attribution", ascending=False)
    sens["rank"] = range(1, len(sens) + 1)
    sens.to_csv(out_dir / "sensor_importance_all.csv", index=False)
    sens.rename(columns={"channel_attribution": "importance_score"})[["sensor", "importance_score", "rank"]].to_csv(
        paths["processed"] / "sensor_importance.csv", index=False)
    for key in ("sensor_group", "sensor_direction", "sensor_point"):
        agg = per_row.groupby(key).agg(windows=("channel_attribution", "size"), sensors=("sensor", "nunique"),
                                       channel_attribution=("channel_attribution", "mean")).reset_index()
        agg.sort_values("channel_attribution", ascending=False).to_csv(out_dir / f"importance_by_{key}.csv", index=False)
    per_row.groupby("scenario")[["channel_attribution", "total_attribution"]].mean().to_csv(out_dir / "importance_by_scenario.csv")
    write_json(out_dir / "shap_summary.json", {"scheme": a.scheme, "seeds": seeds, "explained_windows": len(df),
                                               "explained_recordings": int(df["recording"].nunique()),
                                               "sensors_ranked": int(len(sens)), "features": cols,
                                               "treeshap_crosscheck": check})
    outs = [out_dir / f for f in ("global_feature_importance.csv", "feature_group_importance.csv", "sensor_importance_all.csv",
                                  "importance_by_sensor_group.csv", "importance_by_sensor_direction.csv",
                                  "importance_by_sensor_point.csv", "importance_by_scenario.csv", "shap_summary.json")]
    np.save(paths["processed"] / "shap_abs_mean_loso.npy", abs_mean.astype(np.float32))
    outs.append(paths["processed"] / "shap_abs_mean_loso.npy")
    print(glob.head(15).to_string(), "\n", sens.head(15).to_string(), "\n", check)
    run.finish(outs)
    print("run:", run.run_id)


if __name__ == "__main__":
    main()
