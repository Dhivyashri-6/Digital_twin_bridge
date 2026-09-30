"""Mode-shape change features and their effect on classification and SHAP localisation (AVT records).

For every AVT record the FDD singular vector at each matched mode gives one mode-shape component per channel. Shapes are
phase-aligned and scaled on the reference sensors present in every setup, and compared with the scenario-01 record of
the same setup (which contains the same sensors): dphi_k = phi_k - phi_k(S01), a channel-level feature that, unlike
the natural frequencies, differs between sensors. Scenario 01 is therefore the baseline and is excluded from evaluation.

Evaluation (LOSO over scenarios 02-17, XGBoost, seed 0): all features vs all features + |dphi_1..4|, cluster-bootstrap
CIs over recordings, and TreeSHAP on every held-out window (share of dphi features, channel-level attribution spread).
Added after the first SHAP results showed near-uniform sensor attributions.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import xgboost as xgb
from joblib import Parallel, delayed

from src.evaluation import balanced_weights, bootstrap_over_recordings, feature_sets, load_table, make_model, splits
from src.modal_analysis import fdd, match_modes, pick_peaks
from src.preprocessing import preprocess_signal
from src.reproducibility import start_run
from src.utils import ensure_output_dirs, load_config, write_json
from src.z24_catalog import build_catalog, load_recording, sensor_location


def load_baseline(path: Path) -> list[dict]:
    modes = json.loads(path.read_text())
    return [{"frequency": m["frequency_hz"], "prominence": m["prominence_decades"],
             "shape": np.array([complex(*v) for v in m["shape_reference_sensors"].values()])} for m in modes]


def aligned_shape(u: np.ndarray, sensors: list[str], subset: list[str], ref_shape: np.ndarray) -> np.ndarray:
    """Rotate/scale a complex shape so that its reference-sensor part best matches ``ref_shape`` with unit norm."""
    idx = [sensors.index(s) for s in subset]
    alpha = np.vdot(u[idx], ref_shape)  # conj(u_ref) . ref, so u * alpha/|alpha| is in phase with ref_shape
    u = u * (alpha / abs(alpha) if abs(alpha) > 0 else 1.0)
    return np.real(u) / max(np.linalg.norm(np.real(u[idx])), 1e-12)


def shapes_one(rec, cfg: dict, baseline: list[dict]) -> list[dict]:
    fs = float(cfg["dataset"]["sampling_rate"])
    mc = cfg["modal"]
    subset = list(mc["reference_sensors"])
    loaded = load_recording(rec, fs)
    pre = dict(cfg["preprocessing"], normalization="none")
    x = np.column_stack([preprocess_signal(loaded.data[:, i], fs, pre) for i in range(loaded.data.shape[1])])
    res = fdd(x, fs, loaded.sensors, nperseg=mc["nperseg"])
    matched = match_modes(pick_peaks(res, mc["fmin"], mc["fmax"], mc["peak_prominence"]), loaded.sensors, baseline, subset,
                          min_mac=mc["min_mac"])
    rows = [{"recording": rec.recording_id, "scenario": f"{rec.scenario:02d}", "setup": f"{rec.setup:02d}", "sensor": s}
            for s in loaded.sensors]
    for k, m in enumerate(matched, start=1):
        phi = (aligned_shape(res.u[m["index"]], loaded.sensors, subset, baseline[k - 1]["shape"]) if m is not None
               else np.full(len(loaded.sensors), np.nan))
        for r, v in zip(rows, phi):
            r[f"phi{k}"] = float(v)
    return rows


def main():
    cfg = load_config()
    paths = ensure_output_dirs(cfg)
    ev = cfg["evaluation"]
    run = start_run("mode_shapes", cfg, seeds=[0])
    out = paths["results"] / "mode_shapes"
    out.mkdir(parents=True, exist_ok=True)
    baseline = load_baseline(paths["processed"] / "modal_baseline.json")
    K = len(baseline)
    shape_path = paths["processed"] / "mode_shapes.parquet"
    if not shape_path.exists():
        recs = [r for r in build_catalog(cfg["dataset"]["root_path"])[0] if r.test == "avt"]
        rows = Parallel(n_jobs=6, verbose=5)(delayed(shapes_one)(r, cfg, baseline) for r in recs)
        pd.DataFrame([x for rr in rows for x in rr]).to_parquet(shape_path, index=False)
    sh = pd.read_parquet(shape_path)
    base_sc = f"{cfg['modal']['baseline_scenario']:02d}"
    base = sh[sh.scenario == base_sc].set_index(["setup", "sensor"])[[f"phi{k}" for k in range(1, K + 1)]]
    sh = sh.join(base, on=["setup", "sensor"], rsuffix="_base")
    dcols = []
    for k in range(1, K + 1):
        sh[f"dphi{k}"] = (sh[f"phi{k}"] - sh[f"phi{k}_base"]).abs()
        dcols.append(f"dphi{k}")
    sh.to_csv(out / "mode_shape_changes.csv", index=False)

    n_modes = K
    sets = feature_sets(n_modes)
    rec = pd.read_csv(paths["processed"] / "recordings.csv")
    rates = {f"zeta{k}": float(rec.loc[rec.test == "avt", f"zeta{k}_reliable"].mean()) for k in range(1, n_modes + 1)}
    modal = [f"f{k}" for k in range(1, n_modes + 1)] + [z for z, r in rates.items() if r >= 0.8]
    all_cols = sets["time+spectral"] + modal
    df = load_table(paths["processed"] / "features.parquet", paths["processed"] / "recordings.csv",
                    ev.get("classification_test", "avt"), int(ev["window_stride"]), n_modes)
    df = df[df.scenario != base_sc].merge(sh[["recording", "sensor"] + dcols], on=["recording", "sensor"], how="left")
    df = df.reset_index(drop=True)
    y = df["y"].to_numpy()
    results, shap_rows = [], {}
    for name, cols in (("all", all_cols), ("all+mode_shape", all_cols + dcols)):
        X = df[cols].to_numpy(float)
        score = np.full(len(df), np.nan)
        contrib = np.zeros((len(df), len(cols)))
        for tr, te in splits(df, "loso", 0):
            m = make_model("xgboost", 0, cols, cfg["model"])
            m.fit(X[tr], y[tr], sample_weight=balanced_weights(y[tr]))
            score[te] = m.predict_proba(X[te])[:, 1]
            contrib[te] = m.get_booster().predict(xgb.DMatrix(X[te]), pred_contribs=True)[:, :-1]
        res = bootstrap_over_recordings(y, [score], df["recording"].to_numpy(), int(ev["bootstrap_iterations"]),
                                        int(ev["bootstrap_seed"]))
        results.append({"features": name, "n_features": len(cols), "scheme": "loso", "scenarios": "02-17", "seed": 0,
                        **{m_: res.point[m_] for m_ in res.point},
                        **{f"{m_}_ci_low": res.ci_low[m_] for m_ in res.point},
                        **{f"{m_}_ci_high": res.ci_high[m_] for m_ in res.point}})
        print(name, {k: round(v, 3) for k, v in res.point.items()}, flush=True)
        shap_rows[name] = (cols, np.abs(contrib))
    pd.DataFrame(results).to_csv(out / "mode_shape_classification.csv", index=False)

    cols, a = shap_rows["all+mode_shape"]
    imp = pd.DataFrame({"feature": cols, "mean_abs_shap": a.mean(axis=0)}).sort_values("mean_abs_shap", ascending=False)
    imp["share"] = imp.mean_abs_shap / imp.mean_abs_shap.sum()
    imp.to_csv(out / "mode_shape_shap_features.csv", index=False)
    ch_idx = [i for i, c in enumerate(cols) if c not in modal]  # channel-varying features
    df["channel_attr"] = a[:, ch_idx].sum(axis=1)
    loc = pd.DataFrame([{"sensor": s, **sensor_location(s)} for s in df.sensor.unique()])
    sens = df.groupby("sensor")["channel_attr"].mean().rename("channel_attribution").reset_index().merge(loc, on="sensor")
    sens = sens.sort_values("channel_attribution", ascending=False)
    sens.to_csv(out / "mode_shape_sensor_attribution.csv", index=False)
    grp = df.merge(loc, on="sensor").groupby("group").agg(sensors=("sensor", "nunique"), channel_attribution=("channel_attr", "mean"))
    grp.to_csv(out / "mode_shape_group_attribution.csv")
    big = grp[grp.sensors >= 5]
    per_sc = (df.merge(loc, on="sensor").groupby(["scenario", "group"])["channel_attr"].mean().reset_index()
              .sort_values(["scenario", "channel_attr"], ascending=[True, False]).groupby("scenario").head(1))
    per_sc.to_csv(out / "mode_shape_top_group_per_scenario.csv", index=False)
    share_shape = float(imp.loc[imp.feature.isin(dcols), "share"].sum())
    summary = {"records_with_shapes": int(sh.recording.nunique()), "channels": int(len(sh)),
               "mode_identified_rate": {c: float(sh[c].notna().mean()) for c in [f"phi{k}" for k in range(1, K + 1)]},
               "classification": results, "shap_share_mode_shape_features": share_shape,
               "group_attribution_spread_pct": float(100 * (big.channel_attribution.max() - big.channel_attribution.min())
                                                     / big.channel_attribution.mean()),
               "top_sensors": sens.head(5)[["sensor", "group", "channel_attribution"]].to_dict("records")}
    write_json(out / "mode_shape_summary.json", summary)
    print(json.dumps(summary, indent=1, default=str))
    run.finish([shape_path, out / "mode_shape_changes.csv", out / "mode_shape_classification.csv",
                out / "mode_shape_shap_features.csv", out / "mode_shape_sensor_attribution.csv",
                out / "mode_shape_group_attribution.csv", out / "mode_shape_top_group_per_scenario.csv",
                out / "mode_shape_summary.json"])


if __name__ == "__main__":
    main()
