"""Output-only modal identification: frequency domain decomposition (FDD) and EFDD damping.

For a multichannel record the cross-spectral density matrix G(f) is estimated with
Welch averaging and decomposed by SVD at every frequency line.  Natural frequencies are
peaks of the first singular value s1(f); the first singular vector u1(f) at a peak is the
operational mode shape.  Modes are tracked against a baseline (reference) set using
frequency proximity and the modal assurance criterion (MAC) on channels that exist in
every setup.  Damping is estimated with the enhanced FDD (EFDD): the SDOF bell around a
peak (MAC >= threshold) is transformed back to a correlation function and fitted by
logarithmic decrement.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal
from scipy.optimize import linear_sum_assignment


@dataclass
class FDDResult:
    freqs: np.ndarray        # (n_freq,)
    s: np.ndarray            # singular values (n_freq, n_ch)
    u: np.ndarray            # first singular vectors (n_freq, n_ch)
    n_segments: int
    sensors: list[str]


def fdd(data: np.ndarray, fs: float, sensors: list[str] | None = None, nperseg: int = 4096,
        fmax: float = 20.0) -> FDDResult:
    """FDD of an already preprocessed record ``data`` with shape (n_samples, n_channels)."""
    x = np.asarray(data, dtype=float)
    if x.ndim != 2 or x.shape[0] < nperseg:
        raise ValueError(f"FDD needs (n_samples >= {nperseg}, n_channels); got {x.shape}")
    step = nperseg // 2
    window = signal.get_window("hann", nperseg)
    starts = range(0, x.shape[0] - nperseg + 1, step)
    freqs = np.fft.rfftfreq(nperseg, 1.0 / fs)
    keep = freqs <= fmax
    g = np.zeros((int(keep.sum()), x.shape[1], x.shape[1]), dtype=complex)
    n = 0
    for s0 in starts:
        y = np.fft.rfft(x[s0:s0 + nperseg] * window[:, None], axis=0)[keep]
        g += np.conj(y)[:, :, None] * y[:, None, :]
        n += 1
    g /= n * fs * np.sum(window ** 2)
    u, s, _ = np.linalg.svd(g)
    return FDDResult(freqs[keep], s, u[:, :, 0], n, list(sensors or [str(i) for i in range(x.shape[1])]))


def mac(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    den = np.real(np.vdot(a, a) * np.vdot(b, b))
    return float(np.abs(np.vdot(a, b)) ** 2 / den) if den > 0 else 0.0


def _refine_peak(freqs: np.ndarray, logs: np.ndarray, i: int) -> float:
    if 0 < i < len(freqs) - 1:
        a, b, c = logs[i - 1], logs[i], logs[i + 1]
        den = a - 2 * b + c
        if den != 0:
            delta = 0.5 * (a - c) / den
            if abs(delta) <= 1:
                return float(freqs[i] + delta * (freqs[1] - freqs[0]))
    return float(freqs[i])


def pick_peaks(res: FDDResult, fmin: float = 1.5, fmax: float = 15.0, prominence: float = 0.3,
               max_peaks: int = 15) -> list[dict]:
    """Candidate modes: peaks of log10 s1 with at least ``prominence`` decades of prominence."""
    logs = np.log10(res.s[:, 0] / res.s[:, 0].max())
    idx, props = signal.find_peaks(logs, prominence=prominence)
    out = []
    for i, prom in zip(idx, props["prominences"]):
        if fmin <= res.freqs[i] <= fmax:
            out.append({"index": int(i), "frequency": _refine_peak(res.freqs, logs, i), "prominence": float(prom),
                        "shape": res.u[i]})
    out.sort(key=lambda d: d["prominence"], reverse=True)
    return out[:max_peaks]


def subset_shape(shape: np.ndarray, sensors: list[str], subset: list[str]) -> np.ndarray:
    lookup = {s: i for i, s in enumerate(sensors)}
    v = np.array([shape[lookup[s]] for s in subset], dtype=complex)
    k = np.argmax(np.abs(v))
    return v * np.exp(-1j * np.angle(v[k])) if np.abs(v[k]) > 0 else v


def baseline_modes(results: list[FDDResult], subset: list[str], n_modes: int = 5, fmin: float = 1.5,
                   fmax: float = 15.0, prominence: float = 0.3) -> list[dict]:
    """Identify baseline modes from the averaged normalised s1 of several baseline records."""
    grid = results[0].freqs
    mean_log = np.mean([np.log10(r.s[:, 0] / r.s[:, 0].max()) for r in results], axis=0)
    idx, props = signal.find_peaks(mean_log, prominence=prominence)
    cand = [(p, i) for i, p in zip(idx, props["prominences"]) if fmin <= grid[i] <= fmax]
    cand = sorted(cand, reverse=True)[:n_modes]
    modes = []
    for prom, i in sorted(cand, key=lambda t: t[1]):
        shapes = [subset_shape(r.u[i], r.sensors, subset) for r in results]
        modes.append({"frequency": _refine_peak(grid, mean_log, i), "prominence": float(prom),
                      "shape": np.mean(shapes, axis=0)})
    return modes


def match_modes(peaks: list[dict], sensors: list[str], baseline: list[dict], subset: list[str],
                rel_band: tuple[float, float] = (0.85, 1.10), min_mac: float = 0.8,
                freq_weight: float = 2.0) -> list[dict | None]:
    """Assign candidate peaks to baseline modes (Hungarian on MAC - w*|df/f|)."""
    if not peaks:
        return [None] * len(baseline)
    score = np.full((len(baseline), len(peaks)), -np.inf)
    macs = np.zeros_like(score)
    for k, mode in enumerate(baseline):
        for j, pk in enumerate(peaks):
            ratio = pk["frequency"] / mode["frequency"]
            if rel_band[0] <= ratio <= rel_band[1]:
                m = mac(subset_shape(pk["shape"], sensors, subset), mode["shape"])
                macs[k, j] = m
                if m >= min_mac:
                    score[k, j] = m - freq_weight * abs(ratio - 1.0)
    finite = np.where(np.isfinite(score), score, -1e6)
    rows, cols = linear_sum_assignment(-finite)
    out: list[dict | None] = [None] * len(baseline)
    for k, j in zip(rows, cols):
        if np.isfinite(score[k, j]):
            out[k] = {**peaks[j], "mac": float(macs[k, j])}
    return out


def efdd_damping(res: FDDResult, peak_index: int, mac_threshold: float = 0.8, fs: float | None = None,
                 min_cycles: int = 3) -> dict:
    """EFDD damping estimate around one s1 peak; returns ``reliable`` flag and fit quality."""
    ref = res.u[peak_index]
    n = len(res.freqs)
    lo = hi = peak_index
    while lo > 0 and mac(res.u[lo - 1], ref) >= mac_threshold:
        lo -= 1
    while hi < n - 1 and mac(res.u[hi + 1], ref) >= mac_threshold:
        hi += 1
    bell = np.zeros(n)
    bell[lo:hi + 1] = res.s[lo:hi + 1, 0]
    df = res.freqs[1] - res.freqs[0]
    fs = fs or 2 * res.freqs[-1]
    nfft = 2 * (n - 1)
    corr = np.fft.irfft(bell, n=nfft)
    corr = corr / corr[0] if corr[0] != 0 else corr
    dt = 1.0 / (nfft * df)
    half = corr[: nfft // 2]
    ext, _ = signal.find_peaks(np.abs(half))
    ext = np.r_[0, ext]
    amp = np.abs(half[ext])
    sel = (amp <= 0.9) & (amp >= 0.2)
    result = {"bell_points": int(hi - lo + 1), "damping_ratio": np.nan, "fit_r2": np.nan, "reliable": False,
              "damped_frequency": np.nan}
    if sel.sum() < 2 * min_cycles:
        return result
    t = ext[sel] * dt
    y = np.log(amp[sel])
    slope, intercept = np.polyfit(t, y, 1)
    pred = slope * t + intercept
    ss = np.sum((y - y.mean()) ** 2)
    r2 = 1 - np.sum((y - pred) ** 2) / ss if ss > 0 else np.nan
    # extrema of |corr| are half a period apart
    fd = 1.0 / (2 * np.mean(np.diff(t))) if len(t) > 1 else np.nan
    wn_guess = 2 * np.pi * fd
    zeta = -slope / np.sqrt(wn_guess ** 2 + slope ** 2) if np.isfinite(wn_guess) and wn_guess > 0 else np.nan
    result.update({"damping_ratio": float(zeta), "fit_r2": float(r2), "damped_frequency": float(fd),
                   "reliable": bool(np.isfinite(zeta) and 0 < zeta < 0.10 and r2 >= 0.95 and hi - lo + 1 >= 3)})
    return result


def identify_record(data: np.ndarray, fs: float, sensors: list[str], baseline: list[dict], subset: list[str],
                    nperseg: int = 4096, fmin: float = 1.5, fmax: float = 15.0, prominence: float = 0.3,
                    min_mac: float = 0.8) -> dict:
    """Modal features (f_k, zeta_k, MAC_k) for one preprocessed record."""
    res = fdd(data, fs, sensors, nperseg=nperseg)
    peaks = pick_peaks(res, fmin, fmax, prominence)
    matched = match_modes(peaks, sensors, baseline, subset, min_mac=min_mac)
    row: dict = {"fdd_segments": res.n_segments, "candidate_peaks": len(peaks)}
    for k, m in enumerate(matched, start=1):
        if m is None:
            row.update({f"f{k}": np.nan, f"zeta{k}": np.nan, f"mac{k}": np.nan, f"zeta{k}_reliable": False})
            continue
        d = efdd_damping(res, m["index"], fs=fs)
        row.update({f"f{k}": m["frequency"], f"mac{k}": m["mac"], f"zeta{k}": d["damping_ratio"] if d["reliable"] else np.nan,
                    f"zeta{k}_raw": d["damping_ratio"], f"zeta{k}_r2": d["fit_r2"], f"zeta{k}_reliable": d["reliable"],
                    f"prom{k}": m["prominence"]})
    return row


def ssi_cov(y: np.ndarray, fs: float, lags: int = 40, orders: tuple[int, ...] = (20, 30, 40, 50, 60)) -> dict[int, np.ndarray]:
    """Covariance-driven stochastic subspace identification.

    ``y`` has shape (n_samples, n_channels). Returns, per model order, an array of poles (frequency_hz, damping_ratio)
    with positive frequency.
    """
    y = np.asarray(y, dtype=float)
    y = y - y.mean(axis=0)
    n, m = y.shape
    R = [y[k:].T @ y[: n - k] / (n - k) for k in range(1, 2 * lags + 1)]  # output covariances R_1..R_2i
    T = np.block([[R[lags + r - c - 1] for c in range(lags)] for r in range(lags)])  # block Toeplitz (mi x mi)
    U, S, _ = np.linalg.svd(T)
    out = {}
    for order in orders:
        O = U[:, :order] * np.sqrt(S[:order])  # observability matrix
        A = np.linalg.pinv(O[:-m]) @ O[m:]
        lam = np.log(np.linalg.eigvals(A).astype(complex)) * fs
        f = np.abs(lam) / (2 * np.pi)
        z = -np.real(lam) / np.abs(lam)
        keep = np.imag(lam) > 0
        out[order] = np.column_stack([f[keep], z[keep]])
    return out


def stable_poles(poles: dict[int, np.ndarray], df: float = 0.01, dz: float = 0.3, zmax: float = 0.10) -> np.ndarray:
    """Poles (order, f, zeta) that recur at the next lower order (relative change of f < df and of zeta < dz)."""
    orders = sorted(poles)
    stable = []
    for lo, hi in zip(orders[:-1], orders[1:]):
        for f, z in poles[hi]:
            if not (0 < z < zmax):
                continue
            prev = poles[lo]
            ok = (np.abs(prev[:, 0] - f) / f < df) & (np.abs(prev[:, 1] - z) / z < dz)
            if ok.any():
                stable.append((hi, f, z))
    return np.array(stable) if stable else np.empty((0, 3))
