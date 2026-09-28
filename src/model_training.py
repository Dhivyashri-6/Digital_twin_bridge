from __future__ import annotations

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline

from .feature_extraction import numeric_feature_columns
from .utils import write_json


METADATA_COLUMNS = {"scenario", "setup", "sensor", "recording", "source_file", "window_id", "start_sample",
                    "end_sample", "start_time", "end_time", "label", "sampling_rate"}


def prepare_supervised_data(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    if "label" not in frame:
        raise ValueError("Feature table has no label column")
    valid = frame["label"].notna() & ~frame["label"].astype(str).str.lower().isin({"", "unknown", "none", "nan"})
    labeled = frame.loc[valid].copy()
    if labeled.empty or labeled["label"].nunique() < 2:
        raise ValueError("Supervised training requires at least two reliable label classes; no labels are fabricated")
    columns = numeric_feature_columns(labeled)
    if not columns:
        raise ValueError("No numeric feature columns are available")
    return labeled[columns], labeled["label"].astype(str), columns


def train_xgboost(frame: pd.DataFrame, config: dict, model_dir: str | Path, metrics_dir: str | Path) -> dict:
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise RuntimeError("XGBoost is not installed; run pip install -r requirements.txt") from exc
    X, y, feature_columns = prepare_supervised_data(frame)
    if len(X) < int(config.get("min_samples_for_training", 20)):
        raise ValueError(f"Only {len(X)} labeled windows are available; more samples are required")
    labeled = frame.loc[X.index]
    for candidate in ("recording", "source_file", "scenario", "setup"):
        if candidate in labeled and labeled[candidate].nunique() >= 2:
            groups, group_field = labeled[candidate].astype(str), candidate
            break
    else:
        raise ValueError("No grouping field has at least two values; leakage-safe train/test splitting is impossible")
    splitter = GroupShuffleSplit(n_splits=1, test_size=float(config.get("test_size", 0.25)), random_state=int(config.get("random_state", 42)))
    train_idx, test_idx = next(splitter.split(X, y, groups))
    if y.iloc[train_idx].nunique() < 2:
        raise ValueError("Grouped training split contains fewer than two classes; revise label rules or grouping")
    encoder = LabelEncoder().fit(y)
    y_encoded = encoder.transform(y)
    if set(np.unique(y_encoded[train_idx])) != set(range(len(encoder.classes_))):
        raise ValueError("Grouped training split does not contain every label class; add recordings or revise grouping")
    classifier = XGBClassifier(
        n_estimators=int(config.get("n_estimators", 300)), learning_rate=float(config.get("learning_rate", .05)),
        max_depth=int(config.get("max_depth", 6)), subsample=float(config.get("subsample", .8)),
        colsample_bytree=float(config.get("colsample_bytree", .8)), random_state=int(config.get("random_state", 42)),
        eval_metric=config.get("eval_metric", "logloss"), n_jobs=-1, tree_method="hist")
    pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")), ("classifier", classifier)])
    pipeline.fit(X.iloc[train_idx], y_encoded[train_idx])
    pipeline.z24_label_classes_ = encoder.classes_.tolist()
    prediction_encoded = pipeline.predict(X.iloc[test_idx]).astype(int)
    prediction = encoder.inverse_transform(prediction_encoded)
    probabilities = pipeline.predict_proba(X.iloc[test_idx])
    labels = encoder.classes_.tolist()
    metrics = {"accuracy": accuracy_score(y.iloc[test_idx], prediction),
               "classification_report": classification_report(y.iloc[test_idx], prediction, output_dict=True, zero_division=0),
               "confusion_matrix": confusion_matrix(y.iloc[test_idx], prediction, labels=labels).tolist(),
               "classes": labels, "algorithm": "XGBClassifier", "split_strategy": "GroupShuffleSplit", "group_field": group_field,
               "train_samples": len(train_idx), "test_samples": len(test_idx), "feature_count": len(feature_columns)}
    if len(labels) == 2 and y.iloc[test_idx].nunique() == 2:
        metrics["roc_auc"] = roc_auc_score(y.iloc[test_idx], probabilities[:, 1])
    elif len(labels) > 2 and y.iloc[test_idx].nunique() == len(labels):
        metrics["roc_auc_ovr_weighted"] = roc_auc_score(y.iloc[test_idx], probabilities, labels=labels, multi_class="ovr", average="weighted")
    model_dir, metrics_dir = Path(model_dir), Path(metrics_dir)
    model_dir.mkdir(parents=True, exist_ok=True); metrics_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_dir / "xgboost.pkl")
    write_json(model_dir / "feature_columns.json", feature_columns)
    write_json(metrics_dir / "model_metrics.json", metrics)
    return {"model": pipeline, "metrics": metrics, "features": feature_columns, "test_index": X.index[test_idx].tolist()}
