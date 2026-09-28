from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


def compute_shap(model_pipeline, X: pd.DataFrame, output_dir: str | Path | None = None, max_samples: int = 500) -> dict:
    try:
        import shap
    except ImportError as exc:
        raise RuntimeError("SHAP is not installed; install requirements.txt") from exc
    classifier = model_pipeline.named_steps["classifier"]
    transformed = model_pipeline.named_steps["imputer"].transform(X.iloc[:max_samples])
    explainer = shap.TreeExplainer(classifier)
    raw_values = explainer.shap_values(transformed)
    if isinstance(raw_values, list):
        class_values = np.asarray(raw_values[-1])
    else:
        values = np.asarray(raw_values)
        class_values = values[:, :, -1] if values.ndim == 3 else values
    importance = np.mean(np.abs(class_values), axis=0)
    table = pd.DataFrame({"feature": X.columns, "mean_abs_shap": importance}).sort_values("mean_abs_shap", ascending=False)
    local = pd.DataFrame({"feature": X.columns, "feature_value": transformed[-1], "shap_value": class_values[-1]})
    local["direction"] = np.where(local["shap_value"] >= 0, "toward target class", "away from target class")
    local["rank"] = local["shap_value"].abs().rank(ascending=False, method="dense").astype(int)
    local = local.sort_values("rank")
    if output_dir:
        output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
        table.to_csv(output / "global_feature_importance.csv", index=False)
        local.to_csv(output / "local_explanation.csv", index=False)
    return {"values": class_values, "global_importance": table, "local_explanation": local, "data": transformed}
