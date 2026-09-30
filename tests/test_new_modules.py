import json

import numpy as np
import pandas as pd
import pytest
from scipy.io import savemat

from src.bocpd import MultivariateBOCPD, detection_metrics, run_cusum, run_sequence_bocpd, standardise_by_baseline
from src.feature_extraction import extract_signal_features, extract_window_features_batch
from src.modal_analysis import baseline_modes, efdd_damping, fdd, identify_record, mac, pick_peaks
from src.reproducibility import start_run
from src.z24_catalog import build_catalog, load_recording, scenario_label, sensor_location, verify_aaa_headers


def _write_aaa(path, values, dt=0.01):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{path.name}|8Averages\n{len(values)}\n{dt:.6f}\n" + "\n".join(f"{v:.6e}" for v in values) + "\n")


def _fake_z24(tmp_path):
    rng = np.random.default_rng(0)
    data = rng.normal(size=(512, 3))
    labels = np.array(["100V", "R1V ", "R2L "])
    for sc in ("06", "17"):
        d = tmp_path / "pdt_x" / sc / "avt"
        d.mkdir(parents=True)
        savemat(d / f"{sc}setup01.mat", {"data": data, "labelshulp": labels})
        for i, lab in enumerate(labels):
            _write_aaa(d / "01" / f"{lab.strip()}.aaa", data[:, i])
        _write_aaa(d / "02" / "100V.aaa", data[:, 0])  # setup 02 exists only as ASCII
    return data


def test_catalog_unique_ids_and_duplicate_verification(tmp_path):
    data = _fake_z24(tmp_path)
    recs, counts = build_catalog(tmp_path)
    ids = [r.recording_id for r in recs]
    assert len(ids) == len(set(ids)) == 4  # 06/17 x setup 01/02; both scenarios have "100V"
    assert counts["mat_files"] == 2 and counts["aaa_files"] == 8
    rec = next(r for r in recs if r.recording_id == "S06_avt_setup01")
    loaded = load_recording(rec, 100)
    assert loaded.provenance["aaa_duplicate_identical"] == 3
    assert loaded.data.shape == (512, 3)
    assert np.allclose(loaded.data[:, 0], data[:, 0])
    ascii_only = load_recording(next(r for r in recs if r.recording_id == "S17_avt_setup02"), 100)
    assert ascii_only.provenance["channels_from_aaa"] == 1
    headers = verify_aaa_headers(recs)
    assert headers["sampling_rate_hz"] == {"100.0": 8}


def test_catalog_rejects_wrong_sampling_rate(tmp_path):
    _fake_z24(tmp_path)
    recs, _ = build_catalog(tmp_path)
    with pytest.raises(ValueError):
        load_recording(recs[0], 50)


def test_labels_and_sensor_locations():
    assert scenario_label(8) == "REFERENCE" and scenario_label(9) == "ALTERED_CONDITION"
    assert sensor_location("241T") == {"point": "241", "direction": "transversal", "group": "line_2xx"}
    assert sensor_location("R2V")["group"] == "reference"


def test_batch_features_match_single_window():
    rng = np.random.default_rng(1)
    w = rng.normal(size=(4, 1000))
    batch = extract_window_features_batch(w, 100)
    for i in range(4):
        single = extract_signal_features(w[i], 100)
        for key, value in single.items():
            assert np.isclose(value, batch[key][i], rtol=1e-9, atol=1e-12)


def _two_mode_response(fs=100.0, n=2 ** 15, freqs=(3.9, 9.9), zetas=(0.01, 0.015), seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fs
    shapes = np.array([[1.0, 0.8, 0.5], [0.6, -0.4, 1.0]])
    out = np.zeros((n, 3))
    for (f, z), phi in zip(zip(freqs, zetas), shapes):
        wn = 2 * np.pi * f
        wd = wn * np.sqrt(1 - z ** 2)
        h = np.exp(-z * wn * t[:2000]) * np.sin(wd * t[:2000])
        q = np.convolve(rng.normal(size=n), h)[:n]
        out += q[:, None] * phi[None, :]
    return out + 0.01 * rng.normal(size=out.shape) * out.std()


def test_fdd_recovers_frequencies_and_damping():
    x = _two_mode_response()
    sensors = ["A", "B", "C"]
    res = fdd(x, 100.0, sensors, nperseg=4096)
    peaks = pick_peaks(res, 1.5, 15, 0.3)
    found = sorted(p["frequency"] for p in peaks[:2])
    assert abs(found[0] - 3.9) < 0.05 and abs(found[1] - 9.9) < 0.05
    top = max(peaks, key=lambda p: p["prominence"])
    d = efdd_damping(res, top["index"], fs=100.0)
    assert np.isfinite(d["damping_ratio"]) and 0.002 < d["damping_ratio"] < 0.03
    base = baseline_modes([res], sensors, 5, 1.5, 15, 0.3)
    row = identify_record(_two_mode_response(seed=1), 100.0, sensors, base, sensors)
    assert abs(row["f1"] - 3.9) < 0.05 and row["mac1"] > 0.9


def test_mac_properties():
    a = np.array([1, 2, 3], dtype=complex)
    assert mac(a, 2 * a) == pytest.approx(1.0)
    assert mac(np.array([1, 0]), np.array([0, 1])) == 0


def test_multivariate_bocpd_detects_mean_shift_with_missing_values():
    rng = np.random.default_rng(0)
    x = np.r_[rng.normal(0, 1, (30, 2)), rng.normal(6, 1, (30, 2))]
    x[5, 1] = np.nan
    z, _, _ = standardise_by_baseline(x, 10)
    out = run_sequence_bocpd(z, 0.05)
    m = detection_metrics(out["detections"], [30], tolerance=3)
    assert m["recall"] == 1.0 and m["mean_delay"] <= 3
    det = MultivariateBOCPD(2, 0.1)
    cp, r, probs = det.update([0.0, np.nan])
    assert np.isclose(probs.sum(), 1.0)


def test_cusum_and_metrics():
    x = np.r_[np.zeros(20), np.full(20, 3.0)][:, None] + np.random.default_rng(0).normal(0, 0.3, (40, 1))
    z, _, _ = standardise_by_baseline(x, 10)
    out = run_cusum(z, k=0.5, h=5.0, warmup=3)
    assert any(20 <= d["t"] <= 23 for d in out["detections"])
    m = detection_metrics([{"t": 21}, {"t": 5}], [20, 30], tolerance=2)
    assert m["true_positives"] == 1 and m["precision"] == 0.5 and m["recall"] == 0.5


def test_run_manifest(tmp_path, monkeypatch):
    run = start_run("unittest", {"a": 1}, seeds=[0])
    f = tmp_path / "out.txt"
    f.write_text("x")
    manifest = run.finish([f])
    assert manifest["seeds"] == [0] and list(manifest["outputs"].values())[0]["bytes"] == 1
    assert json.loads((run.dir / "manifest.json").read_text())["run_id"] == run.run_id
    import shutil
    shutil.rmtree(run.dir)
    (run.dir.parent / "LATEST_unittest.txt").unlink()


def test_ssi_cov_recovers_synthetic_modes():
    from src.modal_analysis import ssi_cov, stable_poles
    y = _two_mode_response()
    st = stable_poles(ssi_cov(y, 100.0))
    for f0, z0 in ((3.9, 0.01), (9.9, 0.015)):
        c = st[np.abs(st[:, 1] - f0) < 0.3]
        assert len(c) and abs(np.median(c[:, 1]) - f0) / f0 < 0.01
        assert 0.5 * z0 < np.median(c[:, 2]) < 2 * z0
