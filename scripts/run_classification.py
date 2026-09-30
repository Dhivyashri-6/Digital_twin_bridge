"""Stage 3: reference-vs-altered classification with LOSO and grouped k-fold, several seeds,
class weights, bootstrap CIs over recordings, baselines and ablations.

  --experiment main      XGBoost + RF + SVM + LR on all features, and the natural-frequency threshold
  --experiment ablation  XGBoost on time / spectral / modal / time+spectral / all, with and without
                         per-record z-score normalisation (time+spectral, all)
  --experiment cost      runtime and model size of the final XGBoost model

Out-of-fold scores are cached in data/processed/oof/ so that reruns only compute missing configs.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd

from src.evaluation import (METRICS, bootstrap_over_recordings, cross_validate, feature_sets, frequency_threshold_cv,
                            load_table, make_model, balanced_weights, point_metrics)
from src.reproducibility import start_run
from src.utils import ensure_output_dirs, load_config, write_json

MODELS = ["xgboost", "random_forest", "svm", "logistic_regression"]


def reliable_modal_columns(rec_path: Path, n_modes: int, min_rate: float = 0.8) -> tuple[list[str], dict]:
    rec = pd.read_csv(rec_path)
    avt = rec[rec["test"] == "avt"]
    freq = [f"f{k}" for k in range(1, n_modes + 1) if f"f{k}" in avt]
    rates = {f"zeta{k}": float(avt[f"zeta{k}_reliable"].mean()) for k in range(1, n_modes + 1) if f"zeta{k}_reliable" in avt}
    damp = [c for c, r in rates.items() if r >= min_rate]
    return freq + damp, rates


def cached_cv(cache: Path, key: str, fn):
    path = cache / f"{key}.npz"
    if path.exists():
        d = np.load(path, allow_pickle=True)
        return {"score": d["score"], "fit_seconds": list(d["fit_seconds"]), "predict_seconds": list(d["predict_seconds"])}
    t0 = time.perf_counter()
    out = fn()
    out.setdefault("fit_seconds", [time.perf_counter() - t0])
    out.setdefault("predict_seconds", [0.0])
    np.savez_compressed(path, score=out["score"].astype(np.float64), fit_seconds=np.array(out["fit_seconds"]),
                        predict_seconds=np.array(out["predict_seconds"]))
    return out


def summarise(name: dict, df: pd.DataFrame, scores: list[np.ndarray], ev: dict, extra: dict | None = None) -> list[dict]:
    rows = []
    for level in ("window", "recording"):
        res = bootstrap_over_recordings(df["y"].to_numpy(), scores, df["recording"].to_numpy(),
                                        int(ev["bootstrap_iterations"]), int(ev["bootstrap_seed"]), level=level)
        row = {**name, "level": level, "n_seeds": len(scores)}
        for m in METRICS:
            row[m] = res.point[m]
            row[f"{m}_ci_low"] = res.ci_low[m]
            row[f"{m}_ci_high"] = res.ci_high[m]
            row[f"{m}_seed_sd"] = res.seed_sd[m]
        row.update(extra or {})
        rows.append(row)
    return rows


def per_scenario(df: pd.DataFrame, scores: list[np.ndarray], name: dict) -> pd.DataFrame:
    s = np.mean(scores, axis=0)
    g = pd.DataFrame({"scenario": df["scenario"], "y": df["y"], "correct": (s >= 0.5) == (df["y"] == 1), "score": s,
                      "recording": df["recording"]})
    rec = g.groupby("recording").agg(scenario=("scenario", "first"), y=("y", "first"), score=("score", "mean"))
    out = g.groupby("scenario").agg(label=("y", lambda v: "ALTERED" if v.iloc[0] else "REFERENCE"),
                                    windows=("y", "size"), window_accuracy=("correct", "mean"),
                                    mean_p_altered=("score", "mean")).reset_index()
    out["recording_accuracy"] = out["scenario"].map(rec.assign(c=(rec["score"] >= 0.5) == (rec["y"] == 1))
                                                    .groupby("scenario")["c"].mean())
    for k, v in name.items():
        out.insert(0, k, v)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--experiment", choices=["main", "ablation", "cost", "physics"], default="main")
    p.add_argument("--models", nargs="*", default=MODELS)
    p.add_argument("--schemes", nargs="*", default=["loso", "group_kfold"])
    p.add_argument("--seeds", nargs="*", type=int, help="override evaluation.seeds (e.g. fewer seeds for ablations)")
    a = p.parse_args()
    cfg = load_config(a.config)
    ev = cfg["evaluation"]
    seeds = a.seeds if a.seeds else [int(s) for s in ev["seeds"]]
    paths = ensure_output_dirs(cfg)
    run = start_run(f"classification_{a.experiment}", cfg, seeds=seeds)
    out_dir = paths["results"] / "classification"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = paths["processed"] / "oof"
    cache.mkdir(parents=True, exist_ok=True)
    n_modes = len(json.loads((paths["processed"] / "modal_baseline.json").read_text()))
    modal_cols, rates = reliable_modal_columns(paths["processed"] / "recordings.csv", n_modes)
    sets = feature_sets(n_modes)
    sets["modal"] = modal_cols
    sets["all"] = sets["time+spectral"] + modal_cols
    df = load_table(paths["processed"] / "features.parquet", paths["processed"] / "recordings.csv",
                    ev.get("classification_test", "avt"), int(ev["window_stride"]), n_modes)
    info = {"windows": len(df), "recordings": int(df["recording"].nunique()), "scenarios": int(df["scenario"].nunique()),
            "windows_by_label": df["label"].value_counts().to_dict(),
            "recordings_by_label": df.groupby("label")["recording"].nunique().to_dict(),
            "channels": int(df.groupby(["recording", "sensor"]).ngroups), "feature_sets": sets,
            "damping_reliability_rate": rates, "window_stride": ev["window_stride"]}
    write_json(out_dir / "classification_dataset.json", info)
    print(json.dumps({k: v for k, v in info.items() if k != "feature_sets"}, default=str), flush=True)
    outputs = [out_dir / "classification_dataset.json"]
    rows, scen_rows, timing = [], [], []

    if a.experiment == "main":
        cols = sets["all"]
        for scheme in a.schemes:
            for model in a.models:
                scores = []
                for seed in seeds:
                    key = f"main__{model}__all__none__{scheme}__s{seed}"
                    out = cached_cv(cache, key, lambda: cross_validate(
                        df, cols, model, scheme, seed, cfg["model"], int(ev["group_kfold_splits"]),
                        int(ev["svm_max_train_rows"]), n_jobs=6 if model in ("svm", "logistic_regression") else 1))
                    scores.append(out["score"])
                    timing.append({"model": model, "scheme": scheme, "seed": seed,
                                   "fit_seconds_total": float(np.sum(out["fit_seconds"])),
                                   "predict_seconds_total": float(np.sum(out["predict_seconds"]))})
                    print(key, {k: round(v, 4) for k, v in point_metrics(df["y"].to_numpy(), out["score"]).items()
                                if k in ("balanced_accuracy", "roc_auc", "mcc")}, flush=True)
                name = {"experiment": "main", "model": model, "features": "all", "normalisation": "none", "scheme": scheme}
                rows += summarise(name, df, scores, ev)
                scen_rows.append(per_scenario(df, scores, name))
            fcols = [c for c in modal_cols if c.startswith("f")]
            scores = []
            for seed in seeds:
                key = f"main__freq_threshold__f__none__{scheme}__s{seed}"
                out = cached_cv(cache, key, lambda: frequency_threshold_cv(df, fcols, scheme, seed, int(ev["group_kfold_splits"])))
                scores.append(out["score"])
            name = {"experiment": "main", "model": "freq_threshold", "features": "+".join(fcols), "normalisation": "none",
                    "scheme": scheme}
            rows += summarise(name, df, scores, ev)
            scen_rows.append(per_scenario(df, scores, name))
        pd.DataFrame(timing).to_csv(out_dir / "main_timing.csv", index=False)
        outputs.append(out_dir / "main_timing.csv")

    elif a.experiment == "ablation":
        norm_df = None
        configs = [(fs, "none") for fs in ("time", "spectral", "modal", "time+spectral", "all")]
        configs += [("time+spectral", "zscore"), ("all", "zscore")]
        for scheme in a.schemes:
            for fs, norm in configs:
                data = df
                if norm == "zscore":
                    if norm_df is None:
                        norm_df = load_table(paths["processed"] / "features_norm.parquet", paths["processed"] / "recordings.csv",
                                             ev.get("classification_test", "avt"), int(ev["window_stride"]), n_modes)
                        assert (norm_df["recording"].to_numpy() == df["recording"].to_numpy()).all()
                    data = norm_df
                scores = []
                for seed in seeds:
                    key = ("main__xgboost__all__none__" if (fs, norm) == ("all", "none") else f"abl__xgboost__{fs}__{norm}__") + f"{scheme}__s{seed}"
                    out = cached_cv(cache, key, lambda: cross_validate(data, sets[fs], "xgboost", scheme, seed, cfg["model"],
                                                                       int(ev["group_kfold_splits"])))
                    scores.append(out["score"])
                    print(key, round(point_metrics(df["y"].to_numpy(), out["score"])["roc_auc"], 4), flush=True)
                name = {"experiment": "ablation", "model": "xgboost", "features": fs, "normalisation": norm, "scheme": scheme}
                rows += summarise(name, df, scores, ev, {"n_features": len(sets[fs])})
                scen_rows.append(per_scenario(df, scores, name))

    elif a.experiment == "physics":
        # Added after the first LOSO results: physically constrained variants (see docs/methodology.md)
        for scheme in a.schemes:
            for fs in ("modal",):  # all-feature monotone variant dropped for runtime (modal is the physics case)
                scores = []
                for seed in seeds:
                    key = f"phys__xgboost_monotone__{fs}__none__{scheme}__s{seed}"
                    out = cached_cv(cache, key, lambda: cross_validate(df, sets[fs], "xgboost_monotone", scheme, seed,
                                                                       cfg["model"], int(ev["group_kfold_splits"])))
                    scores.append(out["score"])
                    print(key, round(point_metrics(df["y"].to_numpy(), out["score"])["roc_auc"], 4), flush=True)
                name = {"experiment": "physics", "model": "xgboost_monotone", "features": fs, "normalisation": "none",
                        "scheme": scheme}
                rows += summarise(name, df, scores, ev, {"n_features": len(sets[fs])})
                scen_rows.append(per_scenario(df, scores, name))
            fcols = [c for c in modal_cols if c.startswith("f")]
            scores = []
            for seed in seeds:
                key = f"phys__freq_decrease_threshold__f__none__{scheme}__s{seed}"
                out = cached_cv(cache, key, lambda: frequency_threshold_cv(df, fcols, scheme, seed,
                                                                           int(ev["group_kfold_splits"]), signed=True))
                scores.append(out["score"])
            name = {"experiment": "physics", "model": "freq_decrease_threshold", "features": "+".join(fcols),
                    "normalisation": "none", "scheme": scheme}
            rows += summarise(name, df, scores, ev)
            scen_rows.append(per_scenario(df, scores, name))

    elif a.experiment == "cost":
        cols = sets["all"]
        X, y = df[cols].to_numpy(float), df["y"].to_numpy()
        cost = []
        for model in MODELS:
            m = make_model(model, 0, cols, cfg["model"])
            Xt, yt = X, y
            if model == "svm":
                rng = np.random.default_rng(0)
                idx = rng.choice(len(y), int(ev["svm_max_train_rows"]), replace=False)
                Xt, yt = X[idx], y[idx]
            t0 = time.perf_counter()
            m.fit(Xt, yt, sample_weight=balanced_weights(yt)) if model == "xgboost" else m.fit(Xt, yt)
            fit_s = time.perf_counter() - t0
            sample = X[:10000]
            t0 = time.perf_counter()
            (m.predict_proba(sample) if model != "svm" else m.decision_function(sample))
            pred_s = time.perf_counter() - t0
            buf = io.BytesIO()
            joblib.dump(m, buf)
            cost.append({"model": model, "train_rows": len(yt), "features": len(cols), "fit_seconds": fit_s,
                         "inference_ms_per_1000_windows": 1000 * pred_s / len(sample) * 1000,
                         "serialized_size_bytes": buf.getbuffer().nbytes})
            if model == "xgboost":
                joblib.dump(m, paths["models"] / "xgboost.pkl")
                write_json(paths["models"] / "feature_columns.json", cols)
                outputs += [paths["models"] / "xgboost.pkl", paths["models"] / "feature_columns.json"]
            print(cost[-1], flush=True)
        pd.DataFrame(cost).to_csv(out_dir / "model_cost.csv", index=False)
        outputs.append(out_dir / "model_cost.csv")

    if rows:
        table = pd.DataFrame(rows)
        f = out_dir / f"{a.experiment}_results.csv"
        table.to_csv(f, index=False)
        pd.concat(scen_rows, ignore_index=True).to_csv(out_dir / f"{a.experiment}_per_scenario.csv", index=False)
        outputs += [f, out_dir / f"{a.experiment}_per_scenario.csv"]
        show = ["model", "features", "normalisation", "scheme", "level", "balanced_accuracy", "roc_auc", "mcc", "macro_f1"]
        print(table[show].round(3).to_string())
    run.finish(outputs)
    print("run:", run.run_id)


if __name__ == "__main__":
    main()
