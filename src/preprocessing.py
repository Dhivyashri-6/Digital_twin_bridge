from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy import signal as scipy_signal

from .data_loader import SignalRecord


@dataclass
class SignalWindow:
    scenario: str
    setup: str
    sensor: str
    window_id: int
    start_sample: int
    end_sample: int
    start_time: float | None
    end_time: float | None
    label: str
    signal: np.ndarray
    sampling_rate: float | None
    source_file: str
    recording: str


def validate_signal(values: np.ndarray, min_samples: int = 8) -> np.ndarray:
    x = np.asarray(values, dtype=float).reshape(-1)
    if x.size < min_samples:
        raise ValueError(f"Signal has {x.size} samples; at least {min_samples} are required")
    if not np.isfinite(x).any():
        raise ValueError("Signal contains no finite values")
    return x


def repair_invalid(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=float).copy()
    valid = np.isfinite(x)
    if valid.all():
        return x
    indices = np.arange(x.size)
    if not valid.any():
        raise ValueError("Cannot repair a signal containing only NaN/Inf values")
    x[~valid] = np.interp(indices[~valid], indices[valid], x[valid])
    return x


def preprocess_signal(values: np.ndarray, sampling_rate: float | None, config: dict) -> np.ndarray:
    x = repair_invalid(validate_signal(values))
    if config.get("detrend", True):
        x = scipy_signal.detrend(x, type="linear")
    if config.get("filter_enabled", False):
        if not sampling_rate or sampling_rate <= 0:
            raise ValueError("Filtering requires a valid sampling rate")
        nyquist = sampling_rate / 2
        kind = config.get("filter_type", "bandpass")
        low, high = float(config.get("low_cutoff", 0.5)), float(config.get("high_cutoff", 40.0))
        if kind == "bandpass":
            if not 0 < low < high < nyquist:
                raise ValueError(f"Bandpass cutoffs must satisfy 0 < low < high < Nyquist ({nyquist:g} Hz)")
            wn = [low / nyquist, high / nyquist]
        elif kind == "lowpass":
            if not 0 < high < nyquist:
                raise ValueError("Invalid lowpass cutoff")
            wn = high / nyquist
        elif kind == "highpass":
            if not 0 < low < nyquist:
                raise ValueError("Invalid highpass cutoff")
            wn = low / nyquist
        else:
            raise ValueError(f"Unsupported filter type: {kind}")
        sos = scipy_signal.butter(int(config.get("filter_order", 4)), wn, btype=kind.replace("pass", ""), output="sos")
        x = scipy_signal.sosfiltfilt(sos, x)
    normalization = str(config.get("normalization", "none")).lower()
    if normalization == "standard":
        std = x.std()
        x = (x - x.mean()) / std if std > np.finfo(float).eps else np.zeros_like(x)
    elif normalization == "minmax":
        span = x.max() - x.min()
        x = (x - x.min()) / span if span > np.finfo(float).eps else np.zeros_like(x)
    elif normalization not in {"none", "false", "null"}:
        raise ValueError(f"Unsupported normalization: {normalization}")
    return x


def window_record(record: SignalRecord, config: dict, preprocess_config: dict | None = None) -> list[SignalWindow]:
    x = preprocess_signal(record.signal, record.sampling_rate, preprocess_config or {})
    if record.sampling_rate:
        size = int(round(float(config.get("size_seconds", 10)) * record.sampling_rate))
    else:
        size = int(config.get("fallback_size_samples", 1000))
    overlap = float(config.get("overlap", 0.5))
    if size < 8 or not 0 <= overlap < 1:
        raise ValueError("Window size must be >= 8 and overlap must be in [0, 1)")
    step = max(1, int(round(size * (1 - overlap))))
    windows = []
    for window_id, start in enumerate(range(0, x.size - size + 1, step)):
        end = start + size
        rate = record.sampling_rate
        windows.append(SignalWindow(record.scenario, record.setup, record.sensor, window_id, start, end,
                                    start / rate if rate else None, end / rate if rate else None, record.label,
                                    x[start:end], rate, record.source_file, record.recording))
    return windows


def preprocess_records(records: list[SignalRecord], preprocessing: dict, window: dict) -> list[SignalWindow]:
    windows = []
    for record in records:
        windows.extend(window_record(record, window, preprocessing))
    if not windows:
        raise ValueError("Preprocessing produced no complete windows; reduce the configured window size")
    return windows
