# Architecture

The workflow is **data → detect → classify → explain → visualize**. Unified readers preserve source metadata and feed preprocessing/windowing. Features feed BOCPD (when behavior changed) and, only when traceable labels exist, XGBoost (what condition). SHAP describes model contributions, sensor aggregation supports prioritization, and Plotly renders a monitoring twin. Outputs are immutable derivatives under `data/processed`, `models`, and `results`.

The dashboard's replay mode processes historical windows in order. It is a simulation of streaming, not a live sensor connection.
