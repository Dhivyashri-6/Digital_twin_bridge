"""Leakage-aware evaluation of reference-vs-altered classifiers on the Z24 PDT.

* Cross-validation: leave-one-scenario-out (LOSO) and stratified grouped k-fold by recording.
* Class imbalance: balanced class weights for every model.
* Metrics: accuracy, balanced accuracy, macro-F1, F1 (ALTERED), ROC-AUC, MCC; positive class =
  ALTERED_CONDITION.  Window-level and recording-level (mean window score per recording).
* Uncertainty: percentile bootstrap over recordings (cluster bootstrap); for several seeds each
  replicate averages the metric over seeds, so the interval covers the seed-mean.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC

from .feature_extraction import TIME_FEATURES, spectral_feature_names

POSITIVE = "ALTERED_CONDITION"
MAGNITUDE_FEATURES = {"std", "variance", "rms", "absolute_peak", "peak_to_peak", "spectral_energy", "psd_peak"}


def modal_feature_names(n_modes: int) -> list[str]:
    return [f"f{k}" for k in range(1, n_modes + 1)] + [f"zeta{k}" for k in range(1, n_modes + 1)]


def feature_sets(n_modes: int, fs: float = 100.0) -> dict[str, list[str]]:
    time_f, spec_f, modal_f = list(TIME_FEATURES), spectral_feature_names(fs), modal_feature_names(n_modes)
    return {"time": time_f, "spectral": spec_f, "modal": modal_f, "time+spectral": time_f + spec_f,
            "all": time_f + spec_f + modal_f}


def load_table(features_path, recordings_path, test: str = "avt", stride: int = 2, n_modes: int = 4) -> pd.DataFrame:
    """Window table (non-overlapping windows via ``stride``) joined with record-level modal features."""
    df = pd.read_parquet(features_path)
    df = df[(df["test"] == test) & (df["window_id"] % stride == 0)].reset_index(drop=True)
    rec = pd.read_csv(recordings_path, dtype={"scenario": str, "setup": str})
    cols = ["recording"] + [c for c in modal_feature_names(n_modes) if c in rec]
    df = df.merge(rec[cols], on="recording", how="left", validate="many_to_one")
    df["y"] = (df["label"] == POSITIVE).astype(int)
    return df


def _signed_log(x):
    return np.sign(x) * np.log10(1 + np.abs(x) * 1e6)


def _log_magnitudes(x, idx=()):
    """Signed log of the magnitude-type columns (module level so fitted pipelines can be pickled)."""
    x = np.array(x, dtype=float, copy=True)
    if idx:
        x[:, list(idx)] = _signed_log(x[:, list(idx)])
    return x


def make_model(name: str, seed: int, columns: list[str], cfg: dict | None = None):
    cfg = cfg or {}
    if name in ("xgboost", "xgboost_monotone"):
        from xgboost import XGBClassifier
        extra = {}
        if name == "xgboost_monotone":
            # physical prior: loss of stiffness lowers natural frequencies -> P(altered) non-increasing in f_k
            extra["monotone_constraints"] = "(" + ",".join("-1" if c.startswith("f") and c[1:].isdigit() else "0"
                                                          for c in columns) + ")"
        return XGBClassifier(n_estimators=int(cfg.get("n_estimators", 300)), learning_rate=float(cfg.get("learning_rate", .05)),
                             max_depth=int(cfg.get("max_depth", 6)), subsample=float(cfg.get("subsample", .8)),
                             colsample_bytree=float(cfg.get("colsample_bytree", .8)), tree_method="hist",
                             eval_metric="logloss", n_jobs=8, random_state=seed, **extra)
    log_idx = [i for i, c in enumerate(columns) if c in MAGNITUDE_FEATURES or c.startswith("band_energy")]
    prep = [("imputer", SimpleImputer(strategy="median")),
            ("logmag", FunctionTransformer(_log_magnitudes, kw_args={"idx": tuple(log_idx)}))]
    if name == "random_forest":
        return Pipeline(prep[:1] + [("classifier", RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, max_samples=0.25, class_weight="balanced", n_jobs=8, random_state=seed))])
    if name == "svm":
        return Pipeline(prep + [("scaler", StandardScaler()), ("classifier", SVC(
            C=1.0, gamma="scale", kernel="rbf", class_weight="balanced", random_state=seed))])
    if name == "logistic_regression":
        return Pipeline(prep + [("scaler", StandardScaler()), ("classifier", LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=5000))])
    raise ValueError(name)


def balanced_weights(y: np.ndarray) -> np.ndarray:
    counts = np.bincount(y, minlength=2).astype(float)
    return (len(y) / (2 * counts))[y]


def _fit_predict(name, seed, columns, cfg, X_tr, y_tr, X_te, svm_max_rows):
    model = make_model(name, seed, columns, cfg)
    t0 = time.perf_counter()
    if name in ("xgboost", "xgboost_monotone"):
        model.fit(X_tr, y_tr, sample_weight=balanced_weights(y_tr))
        fit_s = time.perf_counter() - t0
        t1 = time.perf_counter()
        score = model.predict_proba(X_te)[:, 1]
    elif name == "svm":
        rng = np.random.default_rng(seed)
        if len(y_tr) > svm_max_rows:  # stratified random subsample for the kernel SVM
            idx = np.concatenate([rng.choice(np.flatnonzero(y_tr == c), size=max(1, int(round(svm_max_rows * np.mean(y_tr == c)))),
                                             replace=False) for c in (0, 1)])
            X_tr, y_tr = X_tr[idx], y_tr[idx]
        model.fit(X_tr, y_tr)
        fit_s = time.perf_counter() - t0
        t1 = time.perf_counter()
        score = 1 / (1 + np.exp(-model.decision_function(X_te)))
    else:
        model.fit(X_tr, y_tr)
        fit_s = time.perf_counter() - t0
        t1 = time.perf_counter()
        score = model.predict_proba(X_te)[:, 1]
    return score, fit_s, time.perf_counter() - t1, model


def splits(df: pd.DataFrame, scheme: str, seed: int, n_splits: int = 5):
    if scheme == "loso":
        return list(LeaveOneGroupOut().split(df, df["y"], df["scenario"]))
    if scheme == "group_kfold":
        return list(StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(df, df["y"], df["recording"]))
    raise ValueError(scheme)


def cross_validate(df: pd.DataFrame, columns: list[str], model: str, scheme: str, seed: int, cfg: dict | None = None,
                   n_splits: int = 5, svm_max_rows: int = 20000, n_jobs: int = 1, keep_models: bool = False) -> dict:
    X = df[columns].to_numpy(dtype=float)
    y = df["y"].to_numpy()
    folds = splits(df, scheme, seed, n_splits)
    jobs = (delayed(_fit_predict)(model, seed, columns, cfg, X[tr], y[tr], X[te], svm_max_rows) for tr, te in folds)
    results = Parallel(n_jobs=n_jobs)(jobs) if n_jobs != 1 else [
        _fit_predict(model, seed, columns, cfg, X[tr], y[tr], X[te], svm_max_rows) for tr, te in folds]
    score = np.full(len(df), np.nan)
    fold_id = np.full(len(df), -1)
    for k, ((_, te), (s, _, _, _)) in enumerate(zip(folds, results)):
        score[te] = s
        fold_id[te] = k
    out = {"score": score, "fold": fold_id, "fit_seconds": [r[1] for r in results],
           "predict_seconds": [r[2] for r in results], "test_sizes": [len(te) for _, te in folds]}
    if keep_models:
        out["models"] = [r[3] for r in results]
        out["folds"] = folds
    return out


# ----------------------------------------------------------------------------- natural-frequency baseline
def frequency_shift_scores(rec: pd.DataFrame, train_mask: np.ndarray, freq_cols: list[str], signed: bool = False) -> np.ndarray:
    """max_k |f_k - f_ref_k| / f_ref_k (or, if ``signed``, the largest relative frequency *decrease*),
    with f_ref the mean over training REFERENCE recordings."""
    ref = rec.loc[train_mask & (rec["y"].to_numpy() == 0), freq_cols].mean()
    rel = (ref - rec[freq_cols]) / ref if signed else (rec[freq_cols] - ref).abs() / ref
    return rel.max(axis=1, skipna=True).fillna(0).to_numpy()


def youden_threshold(score: np.ndarray, y: np.ndarray) -> float:
    cands = np.unique(score)
    best, thr = -1, cands[0]
    for c in cands:
        pred = score >= c
        tpr = pred[y == 1].mean() if (y == 1).any() else 0
        tnr = (~pred[y == 0]).mean() if (y == 0).any() else 0
        if tpr + tnr - 1 > best:
            best, thr = tpr + tnr - 1, c
    return float(thr)


def frequency_threshold_cv(df: pd.DataFrame, freq_cols: list[str], scheme: str, seed: int, n_splits: int = 5,
                           signed: bool = False) -> dict:
    """Recording-level threshold baseline evaluated with the same folds; scores broadcast to windows.

    The threshold is chosen on the training recordings of each fold (Youden J).  The returned
    ``score`` is the relative shift mapped monotonically to (0, 1) with 0.5 at that threshold."""
    rec = df.groupby("recording", sort=False).agg(scenario=("scenario", "first"), y=("y", "first"),
                                                  **{c: (c, "first") for c in freq_cols}).reset_index()
    folds = splits(df, scheme, seed, n_splits)
    rec_index = {r: i for i, r in enumerate(rec["recording"])}
    win_rec = df["recording"].map(rec_index).to_numpy()
    score = np.full(len(df), np.nan)
    thresholds = []
    raw_all = np.full(len(rec), np.nan)
    for tr, te in folds:
        tr_rec = np.zeros(len(rec), bool)
        tr_rec[np.unique(win_rec[tr])] = True
        te_rec = np.unique(win_rec[te])
        raw = frequency_shift_scores(rec, tr_rec, freq_cols, signed)
        thr = youden_threshold(raw[tr_rec], rec["y"].to_numpy()[tr_rec])
        thresholds.append(thr)
        scale = 0.1 * (np.std(raw[tr_rec]) or 1.0)
        mapped = 1 / (1 + np.exp(-np.clip((raw - thr) / scale, -700, 700)))  # monotone, 0.5 at the threshold
        score[te] = mapped[win_rec[te]]
        raw_all[te_rec] = raw[te_rec]
    return {"score": score, "thresholds": thresholds, "recording_raw_shift": dict(zip(rec["recording"], raw_all))}


# ----------------------------------------------------------------------------- metrics
def _metrics_from_counts(tp, fp, tn, fn) -> dict:
    n = tp + fp + tn + fn
    tpr = tp / (tp + fn) if tp + fn else 0.0
    tnr = tn / (tn + fp) if tn + fp else 0.0
    prec1 = tp / (tp + fp) if tp + fp else 0.0
    prec0 = tn / (tn + fn) if tn + fn else 0.0
    f1_1 = 2 * prec1 * tpr / (prec1 + tpr) if prec1 + tpr else 0.0
    f1_0 = 2 * prec0 * tnr / (prec0 + tnr) if prec0 + tnr else 0.0
    den = np.sqrt(float(tp + fp) * float(tp + fn) * float(tn + fp) * float(tn + fn))
    mcc = (tp * tn - fp * fn) / den if den > 0 else 0.0
    return {"accuracy": (tp + tn) / n if n else 0.0, "balanced_accuracy": (tpr + tnr) / 2, "macro_f1": (f1_0 + f1_1) / 2,
            "f1_altered": f1_1, "recall_reference": tnr, "recall_altered": tpr, "mcc": mcc}


def _weighted_auc(uniq_idx: np.ndarray, y: np.ndarray, w: np.ndarray, n_unique: int) -> float:
    pos = np.bincount(uniq_idx, weights=w * (y == 1), minlength=n_unique)
    neg = np.bincount(uniq_idx, weights=w * (y == 0), minlength=n_unique)
    P, N = pos.sum(), neg.sum()
    if P == 0 or N == 0:
        return float("nan")
    below = np.cumsum(neg) - neg
    return float(np.sum(pos * (below + 0.5 * neg)) / (P * N))


def point_metrics(y: np.ndarray, score: np.ndarray, threshold: float = 0.5) -> dict:
    pred = score >= threshold
    tp, fp = int(np.sum(pred & (y == 1))), int(np.sum(pred & (y == 0)))
    tn, fn = int(np.sum(~pred & (y == 0))), int(np.sum(~pred & (y == 1)))
    out = _metrics_from_counts(tp, fp, tn, fn)
    u, idx = np.unique(score, return_inverse=True)
    out["roc_auc"] = _weighted_auc(idx, y, np.ones(len(y)), len(u))
    out.update({"tp": tp, "fp": fp, "tn": tn, "fn": fn, "n": len(y)})
    return out


METRICS = ["accuracy", "balanced_accuracy", "macro_f1", "f1_altered", "roc_auc", "mcc", "recall_reference", "recall_altered"]


@dataclass
class BootstrapResult:
    point: dict
    ci_low: dict
    ci_high: dict
    seed_sd: dict


def bootstrap_over_recordings(y: np.ndarray, scores: list[np.ndarray], recordings: np.ndarray, n_boot: int = 2000,
                              seed: int = 12345, threshold: float = 0.5, level: str = "window") -> BootstrapResult:
    """Cluster bootstrap over recordings; ``scores`` holds one score vector per model seed."""
    rec_codes, rec_idx = np.unique(recordings, return_inverse=True)
    R = len(rec_codes)
    if level == "recording":
        agg = []
        for s in scores:
            agg.append(np.bincount(rec_idx, weights=s, minlength=R) / np.bincount(rec_idx, minlength=R))
        yr = np.bincount(rec_idx, weights=y, minlength=R) / np.bincount(rec_idx, minlength=R)
        y, scores, rec_idx = (yr > 0.5).astype(int), agg, np.arange(R)
    per_seed = [point_metrics(y, s, threshold) for s in scores]
    point = {m: float(np.mean([p[m] for p in per_seed])) for m in METRICS}
    seed_sd = {m: float(np.std([p[m] for p in per_seed], ddof=1)) if len(per_seed) > 1 else 0.0 for m in METRICS}
    # per-recording confusion counts per seed
    counts, uniq = [], []
    for s in scores:
        pred = s >= threshold
        c = np.stack([np.bincount(rec_idx, weights=(pred & (y == 1)).astype(float), minlength=R),
                      np.bincount(rec_idx, weights=(pred & (y == 0)).astype(float), minlength=R),
                      np.bincount(rec_idx, weights=(~pred & (y == 0)).astype(float), minlength=R),
                      np.bincount(rec_idx, weights=(~pred & (y == 1)).astype(float), minlength=R)], axis=1)
        counts.append(c)
        u, idx = np.unique(s, return_inverse=True)
        uniq.append((idx, len(u)))
    rng = np.random.default_rng(seed)
    boot = {m: np.empty(n_boot) for m in METRICS}
    for b in range(n_boot):
        mult = np.bincount(rng.integers(0, R, R), minlength=R).astype(float)
        w = mult[rec_idx]
        vals = {m: [] for m in METRICS}
        for c, (idx, nu) in zip(counts, uniq):
            tp, fp, tn, fn = (mult @ c)
            mm = _metrics_from_counts(tp, fp, tn, fn)
            mm["roc_auc"] = _weighted_auc(idx, y, w, nu)
            for m in METRICS:
                vals[m].append(mm[m])
        for m in METRICS:
            boot[m][b] = np.nanmean(vals[m])
    lo = {m: float(np.nanpercentile(boot[m], 2.5)) for m in METRICS}
    hi = {m: float(np.nanpercentile(boot[m], 97.5)) for m in METRICS}
    return BootstrapResult(point, lo, hi, seed_sd)
