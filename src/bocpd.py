from __future__ import annotations

import math
import numpy as np
import pandas as pd
from scipy.special import gammaln


class StudentTBOCPD:
    """Adams-MacKay BOCPD with a conjugate Normal-Gamma/Student-t model."""

    def __init__(self, hazard_rate: float = 0.01, threshold: float = 0.8, max_run_length: int = 500,
                 mu0: float = 0.0, kappa0: float = 1.0, alpha0: float = 1.0, beta0: float = 1.0):
        if not 0 < hazard_rate < 1:
            raise ValueError("hazard_rate must be between 0 and 1")
        self.hazard, self.threshold, self.max_run_length = hazard_rate, threshold, max_run_length
        self.prior = (mu0, kappa0, alpha0, beta0)
        self.reset()

    def reset(self):
        mu, kappa, alpha, beta = self.prior
        self.run_probs = np.array([1.0])
        self.mu, self.kappa, self.alpha, self.beta = map(lambda v: np.array([v], dtype=float), (mu, kappa, alpha, beta))

    def _predictive(self, x: float) -> np.ndarray:
        df = 2 * self.alpha
        scale2 = self.beta * (self.kappa + 1) / (self.alpha * self.kappa)
        z = (x - self.mu) ** 2 / (df * scale2)
        logp = gammaln((df + 1) / 2) - gammaln(df / 2) - 0.5 * np.log(df * math.pi * scale2) - ((df + 1) / 2) * np.log1p(z)
        return np.exp(np.clip(logp, -745, 100))

    @staticmethod
    def _student_t_density(x: float, mu: float, kappa: float, alpha: float, beta: float) -> float:
        df = 2 * alpha
        scale2 = beta * (kappa + 1) / (alpha * kappa)
        z = (x - mu) ** 2 / (df * scale2)
        logp = gammaln((df + 1) / 2) - gammaln(df / 2) - 0.5 * math.log(df * math.pi * scale2) - ((df + 1) / 2) * math.log1p(z)
        return math.exp(max(-745.0, min(100.0, float(logp))))

    def update(self, x: float) -> tuple[float, int]:
        pred = self._predictive(float(x))
        growth = self.run_probs * pred * (1 - self.hazard)
        prior_pred = self._student_t_density(float(x), *self.prior)
        cp = float(np.sum(self.run_probs * self.hazard) * prior_pred)
        new_probs = np.concatenate(([cp], growth))[: self.max_run_length + 1]
        total = new_probs.sum()
        new_probs = new_probs / total if total > 0 else np.r_[1.0, np.zeros(len(new_probs) - 1)]
        mu0, kappa0, alpha0, beta0 = self.prior
        kappa = self.kappa + 1
        mu = (self.kappa * self.mu + x) / kappa
        alpha = self.alpha + 0.5
        beta = self.beta + self.kappa * (x - self.mu) ** 2 / (2 * kappa)
        self.mu = np.concatenate(([mu0], mu))[: len(new_probs)]
        self.kappa = np.concatenate(([kappa0], kappa))[: len(new_probs)]
        self.alpha = np.concatenate(([alpha0], alpha))[: len(new_probs)]
        self.beta = np.concatenate(([beta0], beta))[: len(new_probs)]
        self.run_probs = new_probs
        return float(new_probs[0]), int(np.argmax(new_probs))


def run_bocpd(frame: pd.DataFrame, feature: str = "rms", hazard_rate: float = 0.01, threshold: float = 0.8,
              max_run_length: int = 500) -> pd.DataFrame:
    if feature not in frame:
        raise KeyError(f"BOCPD feature '{feature}' is absent; available columns: {list(frame.columns)}")
    rows = []
    group_columns = [c for c in ("sensor", "recording") if c in frame]
    groups = frame.groupby(group_columns, dropna=False, sort=False) if group_columns else [((), frame)]
    for _, group in groups:
        values = group[feature].astype(float).to_numpy()
        finite = values[np.isfinite(values)]
        center, scale = (float(np.median(finite)), float(np.std(finite))) if finite.size else (0.0, 1.0)
        scale = scale if scale > np.finfo(float).eps else 1.0
        detector = StudentTBOCPD(hazard_rate, threshold, max_run_length)
        for (_, original), value in zip(group.iterrows(), values):
            cp, run_length = detector.update((value - center) / scale if np.isfinite(value) else 0.0)
            rows.append({"timestamp": original.get("start_time"), "sensor": original.get("sensor", "unknown"),
                         "recording": original.get("recording", "unknown"), "window_id": original.get("window_id"),
                         "feature": feature, "feature_value": value, "change_probability": cp,
                         "most_probable_run_length": run_length, "change_detected": bool(cp >= threshold)})
    return pd.DataFrame(rows)
