from __future__ import annotations

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd


def load_model(model_path: str | Path, features_path: str | Path):
    model_path, features_path = Path(model_path), Path(features_path)
    if not model_path.exists() or not features_path.exists():
        raise FileNotFoundError("Trained model or feature list is missing; run scripts/train_model.py first")
    return joblib.load(model_path), json.loads(features_path.read_text(encoding="utf-8"))


def predict_condition(model, feature_columns: list[str], rows: pd.DataFrame) -> list[dict]:
    missing = [c for c in feature_columns if c not in rows]
    if missing:
        raise ValueError(f"Prediction data is missing features: {missing}")
    probs = model.predict_proba(rows[feature_columns])
    predictions = model.predict(rows[feature_columns]).astype(int)
    classes = list(getattr(model, "z24_label_classes_", model.classes_))
    decoded = [classes[index] for index in predictions]
    return [{"predicted_class": str(pred), "class_probabilities": {str(c): float(p) for c, p in zip(classes, prob)},
             "confidence": float(np.max(prob))} for pred, prob in zip(decoded, probs)]
