"""XGBoost with baseline-relative modal features (added after the first LOSO results).

The absolute natural frequencies differ between measurement days, which the LOSO results showed to limit XGBoost. Here
each modal feature is expressed relative to the REFERENCE recordings of the training fold only:
    r_fk = (f_k - mean_ref(f_k)) / mean_ref(f_k),   r_zk = zeta_k - mean_ref(zeta_k),
recomputed inside every fold, so the held-out scenario never contributes to its own reference. Variants (XGBoost,
window level, balanced weights, same hyper-parameters as the main study):
    rel_modal            relative modal features only
    rel_modal_monotone   same, P(ALTERED) non-increasing in each r_fk (stiffness loss lowers frequencies)
    all_rel              time + spectral window features + relative modal features
LOSO (3 seeds) and grouped 5-fold; cluster-bootstrap CIs over recordings; TreeSHAP on the held-out LOSO windows of the
best LOSO variant.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import xgboost as xgb

from src.evaluation import balanced_weights, bootstrap_over_recordings, feature_sets, load_table, splits
from src.reproducibility import start_run
from src.utils import ensure_output_dirs, load_config, write_json


def relative_features(df: pd.DataFrame, train_idx: np.ndarray, fcols: list[str], zcols: list[str]) -> pd.DataFrame:
    tr = df.iloc[train_idx]
    ref = tr[tr["y"] == 0].groupby("recording")[fcols + zcols].first().mean()  # recording-level reference mean
    out = pd.DataFrame(index=df.index)
    for c in fcols:
        out["r_" + c] = (df[c] - ref[c]) / ref[c]
    for c in zcols:
        out["r_" + c] = df[c] - ref[c]
    return out


def model(seed: int, cols: list[str], cfg: dict, monotone: bool):
    extra = {}
    if monotone:
        extra["monotone_constraints"] = "(" + ",".join("-1" if c.startswith("r_f") else "0" for c in cols) + ")"
    return xgb.XGBClassifier(n_estimators=int(cfg.get("n_estimators", 300)), learning_rate=float(cfg.get("learning_rate", .05)),
                             max_depth=int(cfg.get("max_depth", 6)), subsample=float(cfg.get("subsample", .8)),
                             colsample_bytree=float(cfg.get("colsample_bytree", .8)), tree_method="hist",
                             eval_metric="logloss", n_jobs=8, random_state=seed, **extra)


def main():
    cfg = load_config()
    paths = ensure_output_dirs(cfg)
    ev = cfg["evaluation"]
    seeds = [int(s) for s in ev["seeds"]]
    run = start_run("relative_features", cfg, seeds=seeds)
    out = paths["results"] / "relative"
    out.mkdir(parents=True, exist_ok=True)
    rec = pd.read_csv(paths["processed"] / "recordings.csv")
    n_modes = sum(1 for c in rec.columns if c.startswith("f") and c[1:].isdigit())
    fcols = [f"f{k}" for k in range(1, n_modes + 1)]
    zcols = [f"zeta{k}" for k in range(1, n_modes + 1) if rec.loc[rec.test == "avt", f"zeta{k}_reliable"].mean() >= 0.8]
    ts = feature_sets(n_modes)["time+spectral"]
    df = load_table(paths["processed"] / "features.parquet", paths["processed"] / "recordings.csv",
                    ev.get("classification_test", "avt"), int(ev["window_stride"]), n_modes)
    y = df["y"].to_numpy()
    rel_names = ["r_" + c for c in fcols + zcols]
    variants = {"rel_modal": (rel_names, False), "rel_modal_monotone": (rel_names, True), "all_rel": (ts + rel_names, False)}
    rows, scores_store = [], {}
    for name, (cols, mono) in variants.items():
        for scheme in ("loso", "group_kfold"):
            if name == "all_rel" and scheme != "loso":
                continue  # full-feature variant: LOSO only (runtime)
            scores, contribs = [], []
            for seed in (seeds if name != "all_rel" else seeds[:1]):  # full-feature variant: one seed (runtime)
                score = np.full(len(df), np.nan)
                contrib = np.zeros((len(df), len(cols))) if scheme == "loso" and seed == seeds[0] else None
                for tr, te in splits(df, scheme, seed, int(ev["group_kfold_splits"])):
                    if name == "all_rel":
                        feats = pd.concat([df[ts], relative_features(df, tr, fcols, zcols)], axis=1)[cols].to_numpy(float)
                        m = model(seed, cols, cfg["model"], mono)
                        m.fit(feats[tr], y[tr], sample_weight=balanced_weights(y[tr]))
                        score[te] = m.predict_proba(feats[te])[:, 1]
                        if contrib is not None:
                            contrib[te] = m.get_booster().predict(xgb.DMatrix(feats[te]), pred_contribs=True)[:, :-1]
                        continue
                    # modal-only variants: features are constant within a recording, so the window-level fit equals a
                    # fit on the unique recordings weighted by their window counts (balanced class weights preserved)
                    rel = relative_features(df, tr, fcols, zcols)[cols]
                    w = pd.Series(balanced_weights(y[tr]), index=df.index[tr])
                    g_tr = pd.DataFrame({"recording": df.recording.iloc[tr], "w": w, "y": y[tr]}).groupby("recording")
                    rtr = g_tr.agg(w=("w", "sum"), y=("y", "first"))
                    first = rel.groupby(df.recording).first()
                    m = model(seed, cols, cfg["model"], mono)
                    m.fit(first.loc[rtr.index].to_numpy(float), rtr.y.to_numpy(), sample_weight=rtr.w.to_numpy())
                    te_rec = df.recording.iloc[te]
                    ute = te_rec.unique()
                    p_ = pd.Series(m.predict_proba(first.loc[ute].to_numpy(float))[:, 1], index=ute)
                    score[te] = te_rec.map(p_).to_numpy()
                    if contrib is not None:
                        cc = m.get_booster().predict(xgb.DMatrix(first.loc[ute].to_numpy(float)), pred_contribs=True)[:, :-1]
                        contrib[te] = pd.DataFrame(cc, index=ute).loc[te_rec].to_numpy()
                scores.append(score)
                if contrib is not None:
                    contribs.append(contrib)
                print(scheme, name, seed, flush=True)
            for level in ("window", "recording"):
                res = bootstrap_over_recordings(y, scores, df["recording"].to_numpy(), int(ev["bootstrap_iterations"]),
                                                int(ev["bootstrap_seed"]), level=level)
                row = {"variant": name, "scheme": scheme, "level": level, "n_seeds": len(scores), "n_features": len(cols)}
                for m_ in res.point:
                    row.update({m_: res.point[m_], f"{m_}_ci_low": res.ci_low[m_], f"{m_}_ci_high": res.ci_high[m_],
                                f"{m_}_seed_sd": res.seed_sd[m_]})
                rows.append(row)
            r = rows[-2]
            print(f"  {scheme} {name}: AUC {r['roc_auc']:.3f} [{r['roc_auc_ci_low']:.3f}, {r['roc_auc_ci_high']:.3f}] "
                  f"BA {r['balanced_accuracy']:.3f} MCC {r['mcc']:.3f}", flush=True)
            np.savez_compressed(paths["processed"] / "oof" / f"rel__{name}__{scheme}.npz", scores=np.array(scores))
            if contribs:
                scores_store[name] = (cols, contribs[0])
    res_df = pd.DataFrame(rows)
    res_df.to_csv(out / "relative_results.csv", index=False)
    lo = res_df[(res_df.scheme == "loso") & (res_df.level == "window")].set_index("variant")
    best = lo["roc_auc"].idxmax()
    cols, contrib = scores_store[best]
    a = np.abs(contrib)
    imp = pd.DataFrame({"feature": cols, "mean_abs_shap": a.mean(axis=0),
                        "mean_shap_reference": contrib[y == 0].mean(axis=0),
                        "mean_shap_altered": contrib[y == 1].mean(axis=0)}).sort_values("mean_abs_shap", ascending=False)
    imp["share"] = imp.mean_abs_shap / imp.mean_abs_shap.sum()
    imp.to_csv(out / "relative_shap.csv", index=False)
    best_scores = np.load(paths["processed"] / "oof" / f"rel__{best}__loso.npz")["scores"].mean(axis=0)
    per = df.assign(score=best_scores).groupby(["scenario", "label", "recording"]).score.mean().reset_index()
    per["correct"] = (per.score >= 0.5) == (per.label != "REFERENCE")
    per_sc = per.groupby(["scenario", "label"]).agg(recordings=("score", "size"), mean_score=("score", "mean"),
                                                    recording_accuracy=("correct", "mean")).reset_index()
    per_sc.to_csv(out / "relative_per_scenario.csv", index=False)
    write_json(out / "relative_summary.json", {"best_loso_variant": best, "reference_features": fcols + zcols,
                                               "results_window_loso": lo[["roc_auc", "roc_auc_ci_low", "roc_auc_ci_high",
                                                                          "balanced_accuracy", "mcc"]].to_dict("index"),
                                               "shap_top": imp.head(8)[["feature", "share"]].to_dict("records")})
    print(res_df[res_df.level == "window"][["variant", "scheme", "roc_auc", "roc_auc_ci_low", "roc_auc_ci_high",
                                            "balanced_accuracy", "mcc", "macro_f1"]].round(3).to_string())
    print(imp.head(8).round(4).to_string())
    print(per_sc.round(3).to_string())
    run.finish([out / "relative_results.csv", out / "relative_shap.csv", out / "relative_per_scenario.csv",
                out / "relative_summary.json"])


if __name__ == "__main__":
    main()
