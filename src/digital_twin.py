from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def schematic_sensor_layout(sensor_ids: list[str]) -> pd.DataFrame:
    ordered = sorted(set(map(str, sensor_ids))) or ["unknown"]
    x = np.linspace(5, 95, len(ordered))
    return pd.DataFrame({"sensor": ordered, "x": x, "y": 3 + 0.6 * np.sin(x / 100 * np.pi)})


def build_digital_twin(sensor_status: pd.DataFrame | None = None) -> go.Figure:
    status = sensor_status.copy() if sensor_status is not None and not sensor_status.empty else pd.DataFrame({"sensor": ["unknown"], "status": ["NORMAL"]})
    layout = schematic_sensor_layout(status["sensor"].astype(str).tolist()).merge(status, on="sensor", how="left")
    colors = {"NORMAL": "#2ecc71", "WARNING": "#f1c40f", "ANOMALY": "#e74c3c"}
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 100], y=[3, 3], mode="lines", line={"width": 14, "color": "#617181"}, name="Bridge deck"))
    for px in (15, 50, 85):
        fig.add_trace(go.Scatter(x=[px, px], y=[0, 3], mode="lines", line={"width": 9, "color": "#86979c"}, showlegend=False))
    fig.add_trace(go.Scatter(x=layout.x, y=layout.y, mode="markers+text", text=layout.sensor, textposition="top center",
                             marker={"size": 16, "color": [colors.get(str(s), "#95a5a6") for s in layout.get("status", "NORMAL")], "line": {"width": 2, "color": "white"}},
                             customdata=layout[[c for c in ("status", "change_probability", "importance_score", "risk_level") if c in layout]],
                             hovertemplate="Sensor %{text}<extra></extra>", name="Sensors"))
    fig.update_layout(title="Schematic sensor visualization (not verified engineering coordinates)", template="plotly_dark", height=430,
                      xaxis={"visible": False, "range": [-5, 105]}, yaxis={"visible": False, "range": [-.5, 6]}, margin=dict(l=20,r=20,t=60,b=20))
    return fig
