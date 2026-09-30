from __future__ import annotations


def gated_risk_config(config: dict, classifier_auc: float | None, min_auc: float = 0.7) -> tuple[dict, bool]:
    """Drop the classifier term (weight 0) unless its grouped ROC-AUC reaches ``min_auc``."""
    include = classifier_auc is not None and classifier_auc >= min_auc
    cfg = dict(config)
    if not include:
        cfg["ml_weight"] = 0.0
    return cfg, include


def sensor_state(change_probability: float, altered_probability: float, risk_score: float, change_threshold: float,
                 use_classifier: bool = True) -> str:
    """Dashboard rule (app.py): two-vote ANOMALY / one-vote WARNING, or risk bands."""
    votes = int(change_probability >= change_threshold) + int(use_classifier and altered_probability >= .5)
    if votes == 2 or risk_score >= .6:
        return "ANOMALY"
    if votes == 1 or risk_score >= .35:
        return "WARNING"
    return "NORMAL"


def overall_state(states: list[str]) -> str:
    affected = sum(s in ("WARNING", "ANOMALY") for s in states)
    if "ANOMALY" in states or (states and affected / len(states) >= .3):
        return "POTENTIAL ANOMALY"
    return "WATCH" if affected else "NORMAL"


def calculate_risk(change_probability: float, abnormal_probability: float | None, attribution: float | None, config: dict) -> dict:
    values = {"bocpd": max(0.0, min(1.0, float(change_probability))),
              "ml": max(0.0, min(1.0, float(abnormal_probability or 0))),
              "attribution": max(0.0, min(1.0, float(attribution or 0)))}
    weights = {k: float(config.get(f"{k}_weight", default)) for k, default in (("bocpd", .3), ("ml", .5), ("attribution", .2))}
    total_weight = sum(weights.values())
    if total_weight <= 0:
        raise ValueError("Risk weights must have a positive sum")
    contrib = dict(values)
    if config.get("attribution_requires_change", True):
        # attribution prioritises sensors only in proportion to the evidence of a change (BOCPD, or the classifier when
        # it is included); without evidence of a change the SHAP ranking alone cannot raise the risk
        evidence = max(values["bocpd"], values["ml"] if weights["ml"] > 0 else 0.0)
        contrib["attribution"] = values["attribution"] * evidence
    score = sum(contrib[k] * weights[k] for k in contrib) / total_weight
    thresholds = config.get("thresholds", {"medium": .35, "high": .6, "critical": .8})
    level = "CRITICAL" if score >= thresholds["critical"] else "HIGH" if score >= thresholds["high"] else "MEDIUM" if score >= thresholds["medium"] else "LOW"
    return {"risk_score": score, "risk_level": level, "components": values, "weights": weights,
            "disclaimer": "Prototype decision-support score; not a structural safety certification."}
