import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, matthews_corrcoef, roc_auc_score

from src.evaluation import (bootstrap_over_recordings, cross_validate, frequency_threshold_cv, point_metrics,
                            youden_threshold)


def _toy(seed=0, n_rec=24, per=20):
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(n_rec):
        y = int(r % 4 != 0)
        sc = f"{r // 2:02d}"
        f1 = 3.9 - 0.2 * y + rng.normal(0, 0.02)
        for w in range(per):
            rows.append({"recording": f"R{r}", "scenario": sc, "y": y, "a": rng.normal(y, 1.0), "b": rng.normal(0, 1), "f1": f1})
    return pd.DataFrame(rows)


def test_point_metrics_match_sklearn():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    s = np.clip(rng.normal(0.3 + 0.4 * y, 0.3), 0, 1).round(2)  # rounding creates ties
    m = point_metrics(y, s)
    pred = s >= 0.5
    assert np.isclose(m["roc_auc"], roc_auc_score(y, s))
    assert np.isclose(m["balanced_accuracy"], balanced_accuracy_score(y, pred))
    assert np.isclose(m["macro_f1"], f1_score(y, pred, average="macro"))
    assert np.isclose(m["mcc"], matthews_corrcoef(y, pred))


def test_bootstrap_contains_point_and_recording_level():
    df = _toy()
    s = 1 / (1 + np.exp(-df["a"].to_numpy()))
    res = bootstrap_over_recordings(df["y"].to_numpy(), [s, s * 0.99], df["recording"].to_numpy(), n_boot=200)
    for m in ("roc_auc", "balanced_accuracy"):
        assert res.ci_low[m] <= res.point[m] <= res.ci_high[m]
    rec = bootstrap_over_recordings(df["y"].to_numpy(), [s], df["recording"].to_numpy(), n_boot=100, level="recording")
    assert 0 <= rec.point["roc_auc"] <= 1


def test_grouped_cv_keeps_groups_apart_and_threshold_baseline():
    df = _toy()
    out = cross_validate(df, ["a", "b"], "logistic_regression", "loso", seed=0)
    assert np.isfinite(out["score"]).all()
    fold_of_scenario = pd.Series(out["fold"]).groupby(df["scenario"]).nunique()
    assert (fold_of_scenario == 1).all()
    g = cross_validate(df, ["a", "b"], "xgboost", "group_kfold", seed=1, n_splits=3)
    assert (pd.Series(g["fold"]).groupby(df["recording"]).nunique() == 1).all()
    ft = frequency_threshold_cv(df, ["f1"], "loso", seed=0)
    assert point_metrics(df["y"].to_numpy(), ft["score"])["roc_auc"] > 0.9
    assert youden_threshold(np.array([0.1, 0.2, 0.8, 0.9]), np.array([0, 0, 1, 1])) == 0.8


def test_mahalanobis_novelty_orders_distance():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from run_novelty import mahalanobis_scores
    rng = np.random.default_rng(0)
    base = rng.normal(0, 1, (30, 3))
    test = np.array([[0, 0, 0], [3, 0, 0], [10, 0, 0]], dtype=float)
    s = mahalanobis_scores(base, test)
    assert s[0] < s[1] < s[2]
