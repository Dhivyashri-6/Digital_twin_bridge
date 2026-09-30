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
        out = np.array([detector.update((v - center) / scale if np.isfinite(v) else 0.0) for v in values])
        n = len(values)
        rows.append(pd.DataFrame({
            "timestamp": group["start_time"].to_numpy() if "start_time" in group else [None] * n,
            "sensor": group["sensor"].to_numpy() if "sensor" in group else ["unknown"] * n,
            "recording": group["recording"].to_numpy() if "recording" in group else ["unknown"] * n,
            "window_id": group["window_id"].to_numpy() if "window_id" in group else [None] * n,
            "feature": feature, "feature_value": values, "change_probability": out[:, 0],
            "most_probable_run_length": out[:, 1].astype(int), "change_detected": out[:, 0] >= threshold}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


class MultivariateBOCPD:
    """BOCPD for a vector observation with independent Normal-Gamma priors per dimension.

    The joint predictive is the product of per-dimension Student-t predictives (diagonal
    model).  Missing values (NaN) are skipped in both prediction and update.
    """

    def __init__(self, n_dims: int, hazard_rate: float = 0.05, max_run_length: int = 500,
                 mu0: float = 0.0, kappa0: float = 1.0, alpha0: float = 1.0, beta0: float = 1.0):
        if not 0 < hazard_rate < 1:
            raise ValueError("hazard_rate must be between 0 and 1")
        self.d, self.hazard, self.max_run_length = n_dims, hazard_rate, max_run_length
        self.prior = np.array([mu0, kappa0, alpha0, beta0], dtype=float)
        self.log_r = np.array([0.0])
        self.params = np.tile(self.prior, (1, n_dims, 1))  # (R, D, 4)

    @staticmethod
    def _logt(x, p):
        mu, kappa, alpha, beta = p[..., 0], p[..., 1], p[..., 2], p[..., 3]
        df = 2 * alpha
        scale2 = beta * (kappa + 1) / (alpha * kappa)
        return (gammaln((df + 1) / 2) - gammaln(df / 2) - 0.5 * np.log(df * math.pi * scale2)
                - ((df + 1) / 2) * np.log1p((x - mu) ** 2 / (df * scale2)))

    def update(self, x) -> tuple[float, int, np.ndarray]:
        x = np.asarray(x, dtype=float)
        ok = np.isfinite(x)
        xx = np.where(ok, x, 0.0)
        logpred = np.where(ok, self._logt(xx, self.params), 0.0).sum(axis=1)
        logprior = np.where(ok, self._logt(xx, self.prior), 0.0).sum()
        log_growth = self.log_r + logpred + math.log(1 - self.hazard)
        log_cp = np.logaddexp.reduce(self.log_r) + math.log(self.hazard) + logprior
        new = np.concatenate(([log_cp], log_growth))[: self.max_run_length + 1]
        new -= np.logaddexp.reduce(new)
        mu, kappa, alpha, beta = (self.params[..., i] for i in range(4))
        k1 = kappa + 1
        upd = np.stack([(kappa * mu + xx) / k1, k1, alpha + 0.5, beta + kappa * (xx - mu) ** 2 / (2 * k1)], axis=-1)
        upd = np.where(ok[None, :, None], upd, self.params)
        self.params = np.concatenate((np.tile(self.prior, (1, self.d, 1)), upd), axis=0)[: len(new)]
        self.log_r = new
        probs = np.exp(new)
        return float(probs[0]), int(np.argmax(probs)), probs


def standardise_by_baseline(values: np.ndarray, n_baseline: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardise columns with mean/std of the first ``n_baseline`` rows (causal baseline)."""
    base = values[:n_baseline]
    mu = np.nanmean(base, axis=0)
    sd = np.nanstd(base, axis=0, ddof=1)
    sd = np.where(np.isfinite(sd) & (sd > 0), sd, 1.0)
    return (values - mu) / sd, mu, sd


def run_sequence_bocpd(values: np.ndarray, hazard_rate: float = 0.05, max_run_length: int = 500,
                       recent: int = 4, kappa0: float = 1.0) -> dict:
    """Run multivariate BOCPD on an ordered (T, D) array; detections are MAP run-length resets.

    A detection is declared at step t when the MAP run length drops (r*_t < r*_{t-1}); the
    estimated change location is t - r*_t.  Re-confirmations of an already reported location
    (+-1 step) are not counted again.
    """
    values = np.atleast_2d(np.asarray(values, dtype=float))
    det = MultivariateBOCPD(values.shape[1], hazard_rate, max_run_length, kappa0=kappa0)
    cp, rmap = np.zeros(len(values)), np.zeros(len(values), dtype=int)
    p_recent = np.zeros(len(values))
    detections: list[dict] = []
    reported: list[int] = []
    for t, x in enumerate(values):
        cp[t], rmap[t], probs = det.update(x)
        p_recent[t] = float(probs[: recent + 1].sum())
        if t > 0 and rmap[t] < rmap[t - 1]:
            loc = t - rmap[t]
            if not any(abs(loc - r) <= 1 for r in reported):
                detections.append({"t": t, "location": int(loc), "run_length": int(rmap[t])})
                reported.append(loc)
    return {"change_probability": cp, "map_run_length": rmap, "recent_change_probability": p_recent,
            "detections": detections}


def run_cusum(values: np.ndarray, k: float = 0.5, h: float = 5.0, warmup: int = 3) -> dict:
    """Two-sided tabular CUSUM on baseline-standardised columns with re-baselining after alarms.

    The reference mean is 0 (the baseline mean) until the first alarm.  After an alarm at t,
    monitoring pauses for ``warmup`` steps and the reference mean is re-estimated from them.
    An alarm is raised when any column's upper or lower sum exceeds ``h``; the estimated change
    location is the step after the last zero of the alarming sum.
    """
    values = np.atleast_2d(np.asarray(values, dtype=float))
    n_steps, n_dims = values.shape
    ref = np.zeros(n_dims)
    sp, sn = np.zeros(n_dims), np.zeros(n_dims)
    zero_p, zero_n = np.full(n_dims, -1), np.full(n_dims, -1)
    detections, stat = [], np.zeros(n_steps)
    t = 0
    while t < n_steps:
        z = values[t] - ref
        ok = np.isfinite(z)
        zz = np.where(ok, z, 0.0)
        sp = np.where(ok, np.maximum(0, sp + zz - k), sp)
        sn = np.where(ok, np.maximum(0, sn - zz - k), sn)
        zero_p = np.where(sp == 0, t, zero_p)
        zero_n = np.where(sn == 0, t, zero_n)
        stat[t] = max(sp.max(), sn.max())
        if stat[t] > h:
            if sp.max() >= sn.max():
                loc = int(zero_p[int(np.argmax(sp))]) + 1
            else:
                loc = int(zero_n[int(np.argmax(sn))]) + 1
            detections.append({"t": t, "location": loc})
            w = values[t + 1: t + 1 + warmup]
            if len(w):
                with np.errstate(all="ignore"):
                    m = np.nanmean(w, axis=0)
                ref = np.where(np.isfinite(m), m, ref)
            sp, sn = np.zeros(n_dims), np.zeros(n_dims)
            zero_p[:] = zero_n[:] = t + warmup
            t += 1 + warmup
            continue
        t += 1
    return {"statistic": stat, "detections": detections}


def detection_metrics(detections: list[dict], boundaries: list[int], tolerance: int) -> dict:
    """Match detections (by detection time) to true change points.

    A detection at time t is a true positive for boundary b if b <= t <= b + tolerance and b is
    not yet matched (greedy in time order).  Delay = t - b in steps.
    """
    boundaries = sorted(boundaries)
    matched: dict[int, dict] = {}
    n_tp = 0
    for d in sorted(detections, key=lambda x: x["t"]):
        hit = next((b for b in boundaries if b not in matched and b <= d["t"] <= b + tolerance), None)
        if hit is not None:
            matched[hit] = d
            n_tp += 1
    n_det = len(detections)
    precision = n_tp / n_det if n_det else 0.0
    recall = n_tp / len(boundaries) if boundaries else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    delays = [matched[b]["t"] - b for b in boundaries if b in matched]
    loc_err = [abs(matched[b].get("location", matched[b]["t"]) - b) for b in boundaries if b in matched]
    return {"detections": n_det, "true_positives": n_tp, "false_positives": n_det - n_tp,
            "boundaries": len(boundaries), "missed": len(boundaries) - n_tp, "precision": precision,
            "recall": recall, "f1": f1, "mean_delay": float(np.mean(delays)) if delays else float("nan"),
            "median_delay": float(np.median(delays)) if delays else float("nan"),
            "mean_location_error": float(np.mean(loc_err)) if loc_err else float("nan"),
            "matched_boundaries": sorted(matched), "tolerance": tolerance}
