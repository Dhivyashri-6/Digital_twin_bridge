from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import signal, stats

from .preprocessing import SignalWindow


def extract_signal_features(values: np.ndarray, sampling_rate: float | None) -> dict[str, float]:
    x = np.asarray(values, dtype=float)
    eps = np.finfo(float).eps
    rms = float(np.sqrt(np.mean(x ** 2)))
    abs_peak = float(np.max(np.abs(x)))
    rate = float(sampling_rate or 1.0)
    freqs, psd = signal.welch(x, fs=rate, nperseg=min(256, x.size))
    psd_sum = float(psd.sum())
    centroid = float(np.sum(freqs * psd) / psd_sum) if psd_sum > eps else 0.0
    bandwidth = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * psd) / psd_sum)) if psd_sum > eps else 0.0
    dominant = float(freqs[int(np.argmax(psd))]) if psd.size else 0.0
    df = freqs[1] - freqs[0] if freqs.size > 1 else 1.0
    spectral_energy = float(np.sum(psd) * df)
    nyquist = rate / 2
    bands = [(0, min(1, nyquist)), (1, min(5, nyquist)), (5, min(20, nyquist)), (20, min(40, nyquist))]
    features = {
        "mean": float(np.mean(x)), "std": float(np.std(x)), "variance": float(np.var(x)), "rms": rms,
        "minimum": float(np.min(x)), "maximum": float(np.max(x)), "absolute_peak": abs_peak,
        "peak_to_peak": float(np.ptp(x)), "crest_factor": abs_peak / rms if rms > eps else 0.0,
        "skewness": float(np.nan_to_num(stats.skew(x, bias=False))),
        "kurtosis": float(np.nan_to_num(stats.kurtosis(x, fisher=True, bias=False))),
        "dominant_frequency": dominant, "spectral_energy": spectral_energy,
        "frequency_centroid": centroid, "spectral_bandwidth": bandwidth,
        "psd_peak": float(psd.max()) if psd.size else 0.0,
    }
    for low, high in bands:
        key = f"band_energy_{low:g}_{high:g}hz"
        mask = (freqs >= low) & (freqs < high if high < nyquist else freqs <= high)
        features[key] = float(np.sum(psd[mask]) * df) if mask.any() and high > low else 0.0
    return features


def extract_features(windows: list[SignalWindow]) -> pd.DataFrame:
    rows = []
    for w in windows:
        row = {"scenario": w.scenario, "setup": w.setup, "sensor": w.sensor, "recording": w.recording,
               "source_file": w.source_file, "window_id": w.window_id, "start_sample": w.start_sample,
               "end_sample": w.end_sample, "start_time": w.start_time, "end_time": w.end_time, "label": w.label,
               "sampling_rate": w.sampling_rate}
        row.update(extract_signal_features(w.signal, w.sampling_rate))
        rows.append(row)
    return pd.DataFrame(rows)


def numeric_feature_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {"window_id", "start_sample", "end_sample", "start_time", "end_time", "sampling_rate"}
    return [c for c in frame.select_dtypes(include=np.number).columns if c not in excluded]
