from __future__ import annotations


def calculate_risk(change_probability: float, abnormal_probability: float | None, attribution: float | None, config: dict) -> dict:
    values = {"bocpd": max(0.0, min(1.0, float(change_probability))),
              "ml": max(0.0, min(1.0, float(abnormal_probability or 0))),
              "attribution": max(0.0, min(1.0, float(attribution or 0)))}
    weights = {k: float(config.get(f"{k}_weight", default)) for k, default in (("bocpd", .3), ("ml", .5), ("attribution", .2))}
    total_weight = sum(weights.values())
    if total_weight <= 0:
        raise ValueError("Risk weights must have a positive sum")
    score = sum(values[k] * weights[k] for k in values) / total_weight
    thresholds = config.get("thresholds", {"medium": .35, "high": .6, "critical": .8})
    level = "CRITICAL" if score >= thresholds["critical"] else "HIGH" if score >= thresholds["high"] else "MEDIUM" if score >= thresholds["medium"] else "LOW"
    return {"risk_score": score, "risk_level": level, "components": values, "weights": weights,
            "disclaimer": "Prototype decision-support score; not a structural safety certification."}
