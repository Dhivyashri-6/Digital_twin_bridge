from __future__ import annotations

import re
import pandas as pd


def extract_sensor_id(feature_name: str) -> str:
    patterns = [r"sensor[_-]?(\d+[a-z]?)", r"(?:^|_)(?:ch(?:annel)?[_-]?)?(\d+[a-z]?)(?:_|$)"]
    for pattern in patterns:
        match = re.search(pattern, str(feature_name), re.I)
        if match:
            return match.group(1)
    return "unmapped"


def aggregate_sensor_importance(attributions: pd.DataFrame, sensor_column: str | None = None) -> pd.DataFrame:
    frame = attributions.copy()
    value_col = "mean_abs_shap" if "mean_abs_shap" in frame else "shap_value"
    frame["importance"] = frame[value_col].abs()
    frame["sensor"] = frame[sensor_column].astype(str) if sensor_column and sensor_column in frame else frame["feature"].map(extract_sensor_id)
    result = frame.groupby("sensor", as_index=False)["importance"].sum().rename(columns={"importance": "importance_score"})
    result = result.sort_values("importance_score", ascending=False).reset_index(drop=True)
    result["rank"] = result.index + 1
    return result
