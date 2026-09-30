"""Stage 1: load every Z24 PDT recording (all scenarios, AVT+FVT, all setups), verify duplicates,
preprocess, window, extract window features (main + normalisation ablation) and modal features.

Outputs (data/processed):
  features.parquet        one row per (channel, 10 s window), no normalisation (main)
  features_norm.parquet   same with per-record z-score normalisation (ablation only)
  recordings.csv          one row per recording: provenance, counts, modal features
  channels.csv            one row per unique channel
  modal_baseline.json     baseline modes identified from scenario 01 AVT
  data_counts.json        file / channel / window counts
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numpy.lib.stride_tricks import sliding_window_view

from src.feature_extraction import extract_window_features_batch
from src.modal_analysis import baseline_modes, fdd, identify_record
from src.preprocessing import preprocess_signal
from src.reproducibility import start_run
from src.utils import ensure_output_dirs, load_config, write_json
from src.z24_catalog import (build_catalog, load_recording, scenario_label, sensor_location,
                             verify_aaa_headers)


def preprocess_matrix(data: np.ndarray, fs: float, pre: dict) -> np.ndarray:
    return np.column_stack([preprocess_signal(data[:, i], fs, pre) for i in range(data.shape[1])])


def window_frame(x: np.ndarray, fs: float, win: dict, meta: dict) -> pd.DataFrame:
    size = int(round(float(win["size_seconds"]) * fs))
    step = max(1, int(round(size * (1 - float(win["overlap"])))))
    w = sliding_window_view(x, size)[::step]
    feats = extract_window_features_batch(w, fs)
    n = len(w)
    starts = np.arange(n) * step
    frame = pd.DataFrame({"window_id": np.arange(n), "start_sample": starts, "end_sample": starts + size,
                          "start_time": starts / fs, "end_time": (starts + size) / fs, **feats})
    for key, value in meta.items():
        frame.insert(0, key, value)
    return frame


def process_one(rec, cfg: dict, baseline: list[dict] | None) -> dict:
    fs = float(cfg["dataset"]["sampling_rate"])
    ref = tuple(cfg["dataset"].get("reference_scenarios", [1, 2, 8]))
    loaded = load_recording(rec, fs)
    pre_main = dict(cfg["preprocessing"], normalization=cfg["preprocessing"].get("normalization", "none"))
    pre_abl = dict(cfg["preprocessing"], normalization=cfg["preprocessing"].get("ablation_normalization", "standard"))
    x_main = preprocess_matrix(loaded.data, fs, pre_main)
    x_abl = preprocess_matrix(loaded.data, fs, pre_abl)
    label = scenario_label(rec.scenario, ref)
    main_frames, abl_frames, channels = [], [], []
    source = str(rec.mat_path or (rec.zip_member[0] if rec.zip_member else ""))
    for i, sensor in enumerate(loaded.sensors):
        loc = sensor_location(sensor)
        meta = {"label": label, "sampling_rate": fs, "sensor_group": loc["group"], "sensor_direction": loc["direction"],
                "sensor_point": loc["point"], "source_file": source, "file_stem": rec.file_stem,
                "channel_id": f"{rec.recording_id}_{sensor}", "recording": rec.recording_id, "sensor": sensor,
                "setup": f"{rec.setup:02d}", "test": rec.test, "scenario": f"{rec.scenario:02d}"}
        main_frames.append(window_frame(x_main[:, i], fs, cfg["window"], meta))
        abl_frames.append(window_frame(x_abl[:, i], fs, cfg["window"], meta))
        channels.append({"channel_id": meta["channel_id"], "recording": rec.recording_id, "scenario": meta["scenario"],
                         "test": rec.test, "setup": meta["setup"], "sensor": sensor, **loc,
                         "n_samples": int(loaded.data.shape[0]), "n_windows": len(main_frames[-1])})
    row = {"recording": rec.recording_id, "scenario": f"{rec.scenario:02d}", "test": rec.test, "setup": f"{rec.setup:02d}",
           "file_stem": rec.file_stem, "label": label, "source_file": source, "n_channels": len(loaded.sensors),
           "n_samples": int(loaded.data.shape[0]), "duration_s": loaded.data.shape[0] / fs,
           "n_windows": int(sum(len(f) for f in main_frames)), "n_aaa_files": len(rec.aaa_paths),
           "has_mat": rec.mat_path is not None, "has_zip_copy": rec.zip_member is not None, **loaded.provenance}
    mc = cfg["modal"]
    if baseline is not None:
        try:
            modal = identify_record(x_main, fs, loaded.sensors, baseline, list(mc["reference_sensors"]), mc["nperseg"],
                                    mc["fmin"], mc["fmax"], mc["peak_prominence"], mc["min_mac"])
            modal["modal_valid_test_type"] = rec.test in mc.get("test_types", ["avt"])
            row.update(modal)
        except Exception as exc:  # recorded, never silently dropped
            row["modal_error"] = str(exc)
    return {"main": pd.concat(main_frames, ignore_index=True), "abl": pd.concat(abl_frames, ignore_index=True),
            "channels": channels, "row": row}


def compute_baseline(recs, cfg: dict) -> list[dict]:
    fs = float(cfg["dataset"]["sampling_rate"])
    mc = cfg["modal"]
    pre = dict(cfg["preprocessing"], normalization="none")
    results = []
    for rec in recs:
        if rec.scenario == int(mc["baseline_scenario"]) and rec.test in mc.get("test_types", ["avt"]):
            loaded = load_recording(rec, fs)
            results.append(fdd(preprocess_matrix(loaded.data, fs, pre), fs, loaded.sensors, nperseg=mc["nperseg"]))
    modes = baseline_modes(results, list(mc["reference_sensors"]), int(mc["max_modes"]), mc["fmin"], mc["fmax"],
                           mc["peak_prominence"])
    return modes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--dataset-root")
    p.add_argument("--jobs", type=int, default=6)
    p.add_argument("--limit", type=int, help="process only the first N recordings (smoke test)")
    a = p.parse_args()
    cfg = load_config(a.config)
    paths = ensure_output_dirs(cfg)
    run = start_run("process", cfg)
    root = Path(a.dataset_root or cfg["dataset"]["root_path"])
    recs, cat_counts = build_catalog(root)
    if a.limit:
        recs = recs[: a.limit]
    headers = verify_aaa_headers(recs)
    print("Catalogue:", cat_counts, "\nAAA headers:", headers, flush=True)

    baseline = compute_baseline(recs, cfg)
    write_json(paths["processed"] / "modal_baseline.json",
               [{"mode": k + 1, "frequency_hz": m["frequency"], "prominence_decades": m["prominence"],
                 "shape_reference_sensors": {s: [float(np.real(v)), float(np.imag(v))] for s, v in
                                             zip(cfg["modal"]["reference_sensors"], m["shape"])}}
                for k, m in enumerate(baseline)])
    print("Baseline modes (Hz):", [round(m["frequency"], 3) for m in baseline], flush=True)

    results = Parallel(n_jobs=a.jobs, verbose=5)(delayed(process_one)(r, cfg, baseline) for r in recs)
    main_df = pd.concat([r["main"] for r in results], ignore_index=True)
    abl_df = pd.concat([r["abl"] for r in results], ignore_index=True)
    channels = pd.DataFrame([c for r in results for c in r["channels"]])
    recordings = pd.DataFrame([r["row"] for r in results])

    # Consistency check of the label rules against scenario-derived labels.
    rule_labels = pd.Series("unknown", index=recordings.index)
    for pattern, lab in cfg["dataset"].get("label_rules", {}).items():
        rule_labels[recordings["source_file"].str.contains(pattern, regex=True)] = lab
    mismatches = int((rule_labels != recordings["label"]).sum())

    out = {k: paths["processed"] / v for k, v in {"features": "features.parquet", "features_norm": "features_norm.parquet",
           "recordings": "recordings.csv", "channels": "channels.csv"}.items()}
    main_df.to_parquet(out["features"], index=False)
    abl_df.to_parquet(out["features_norm"], index=False)
    recordings.to_csv(out["recordings"], index=False)
    channels.to_csv(out["channels"], index=False)

    prov_cols = [c for c in recordings.columns if c.startswith(("aaa_", "zip_", "channels_from"))]
    counts = {
        "catalogue": cat_counts,
        "aaa_header_verification": headers,
        "recordings": int(len(recordings)), "unique_recording_ids": int(recordings["recording"].nunique()),
        "recordings_by_test": recordings["test"].value_counts().to_dict(),
        "recordings_by_scenario": recordings.groupby("scenario").size().to_dict(),
        "setups_per_scenario_test": {"min": int(recordings.groupby(["scenario", "test"]).size().min()),
                                     "max": int(recordings.groupby(["scenario", "test"]).size().max())},
        "unique_channels": int(len(channels)), "channels_by_test": channels["test"].value_counts().to_dict(),
        "distinct_sensor_names": int(channels["sensor"].nunique()),
        "windows_total": int(len(main_df)), "windows_by_test": main_df["test"].value_counts().to_dict(),
        "windows_by_label": main_df["label"].value_counts().to_dict(),
        "windows_per_channel": channels["n_windows"].value_counts().sort_index().to_dict(),
        "samples_per_channel": channels["n_samples"].value_counts().sort_index().to_dict(),
        "provenance_totals": {c: float(recordings[c].fillna(0).sum()) for c in prov_cols},
        "label_rule_mismatches": mismatches,
        "window": cfg["window"], "sampling_rate_hz": cfg["dataset"]["sampling_rate"],
        "baseline_modes_hz": [m["frequency"] for m in baseline],
    }
    write_json(paths["processed"] / "data_counts.json", counts)
    print(json.dumps(counts, indent=1, default=str))
    run.finish(list(out.values()) + [paths["processed"] / "data_counts.json", paths["processed"] / "modal_baseline.json"])
    print("run:", run.run_id)


if __name__ == "__main__":
    main()
