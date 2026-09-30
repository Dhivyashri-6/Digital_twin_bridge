"""Stage 7 (optional): Z24 one-year EMS monitoring.

Pass 1 (extract): FDD of every hourly archive (8 accelerometers), candidate peaks + temperatures.
Pass 2 (analyse): baseline modes from the first two weeks, MAC tracking, temperature compensation
(piecewise-linear regression with a 0 degC hinge, and PCA), then BOCPD / CUSUM on the daily mean
novelty index to detect the onset of the progressive damage tests.

Ground truth used for evaluation (from the PDT reports in the dataset): first reference measurement
on 04-05.08.1998 (scenario 01); the 40 mm pier lowering (scenario 04) was measured on 13-14.08.1998.
The onset window for damage is therefore 1998-08-05 .. 1998-08-14.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src.bocpd import detection_metrics, run_cusum, run_sequence_bocpd, standardise_by_baseline
from src.ems import list_archives, read_archive
from src.modal_analysis import FDDResult, baseline_modes, fdd, match_modes, pick_peaks
from src.preprocessing import preprocess_signal
from src.reproducibility import start_run
from src.utils import ensure_output_dirs, load_config, write_json

CHANNELS = ["03", "05", "06", "07", "10", "12", "14", "16"]
TRAIN_END = datetime(1998, 7, 1)          # temperature model fitted on 10.11.1997 - 30.06.1998
ONSET_START, ONSET_END = datetime(1998, 8, 5), datetime(1998, 8, 14)


def extract_one(path: Path, pre: dict, nperseg: int) -> dict:
    try:
        rec = read_archive(path)
    except Exception as exc:
        return {"archive": path.stem, "error": str(exc)}
    out = {"archive": rec["archive"], "timestamp": rec["timestamp"], "temperatures": rec["temperatures"],
           "channels": sorted(rec["channels"])}
    if any(c not in rec["channels"] for c in CHANNELS):
        out["error"] = "missing channels"
        return out
    dts = {round(rec["channels"][c][1], 6) for c in CHANNELS}
    if dts != {0.01}:
        out["error"] = f"sample interval {dts}"
        return out
    n = min(rec["channels"][c][0].size for c in CHANNELS)
    x = np.column_stack([preprocess_signal(rec["channels"][c][0][:n], 100.0, pre) for c in CHANNELS])
    res = fdd(x, 100.0, CHANNELS, nperseg=nperseg)
    out["peaks"] = [{k: v for k, v in p.items()} for p in pick_peaks(res, 1.5, 15.0, 0.3)]
    out["s1_log"] = np.log10(res.s[:, 0] / res.s[:, 0].max()).astype(np.float32)
    out["freqs"] = res.freqs
    out["u1_rows"] = res.u.astype(np.complex64)
    out["rms"] = float(np.sqrt(np.mean(x ** 2)))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--stage", choices=["extract", "analyse", "all"], default="all")
    p.add_argument("--jobs", type=int, default=6)
    a = p.parse_args()
    cfg = load_config(a.config)
    paths = ensure_output_dirs(cfg)
    run = start_run("ems", cfg)
    out_dir = paths["results"] / "ems"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = paths["processed"] / "ems_fdd.pkl"
    pre = dict(cfg["preprocessing"], normalization="none")
    outputs = []
    if a.stage in ("extract", "all") or not cache.exists():
        archives = list_archives(cfg["dataset"]["root_path"])
        res = Parallel(n_jobs=a.jobs, verbose=2, batch_size=8)(
            delayed(extract_one)(pth, pre, cfg["modal"]["nperseg"]) for pth in archives)
        with cache.open("wb") as fh:
            pickle.dump(res, fh)
    if a.stage == "extract":
        run.finish([cache])
        return
    with cache.open("rb") as fh:
        res = pickle.load(fh)
    counts = {"archives": len(res), "read_errors": sum("error" in r and "peaks" not in r and r.get("error") != "missing channels"
                                                       and not str(r.get("error", "")).startswith("sample") for r in res),
              "missing_channels": sum(r.get("error") == "missing channels" for r in res),
              "bad_interval": sum(str(r.get("error", "")).startswith("sample") for r in res),
              "no_timestamp": sum(r.get("timestamp") is None for r in res)}
    ok = [r for r in res if "peaks" in r and r.get("timestamp") is not None]
    ok.sort(key=lambda r: r["timestamp"])
    counts["usable"] = len(ok)
    t0 = ok[0]["timestamp"]
    base_rs = [r for r in ok if (r["timestamp"] - t0).days < 14]
    base_fdd = [FDDResult(r["freqs"], np.exp(np.log(10) * r["s1_log"])[:, None], r["u1_rows"], 0, CHANNELS) for r in base_rs]
    modes = baseline_modes(base_fdd, CHANNELS, int(cfg["modal"]["max_modes"]), 1.5, 15.0, cfg["modal"]["peak_prominence"])
    counts["baseline_archives"] = len(base_rs)
    counts["baseline_modes_hz"] = [m["frequency"] for m in modes]
    # Frequency-only tracking: the MAC between the November baseline and Jan-May records collapses for
    # all modes (see ems_summary.json -> mac_diagnostic), so shapes are not comparable over the year.
    # Each mode takes the most prominent peak inside a band bounded by the midpoints between baseline
    # modes (outer limits 0.85 f_1 and 1.15 f_K), which admits the winter stiffening of the deck.
    fb = [m["frequency"] for m in modes]
    edges = [0.85 * fb[0]] + [(a + b) / 2 for a, b in zip(fb[:-1], fb[1:])] + [1.15 * fb[-1]]
    counts["tracking_bands_hz"] = [[edges[k], edges[k + 1]] for k in range(len(fb))]
    diag = {}
    rows = []
    for r in ok:
        row = {"archive": r["archive"], "timestamp": r["timestamp"], "rms": r["rms"]}
        temps = [v for k, v in r["temperatures"].items() if k.startswith("T") and k != "TE" and np.isfinite(v)]
        row["temperature_median"] = float(np.median(temps)) if temps else np.nan
        row["temperature_channels"] = len(temps)
        for k in range(len(fb)):
            cand = [p_ for p_ in r["peaks"] if edges[k] <= p_["frequency"] < edges[k + 1]]
            best = max(cand, key=lambda p_: p_["prominence"]) if cand else None
            row[f"f{k + 1}"] = best["frequency"] if best else np.nan
            row[f"prom{k + 1}"] = best["prominence"] if best else np.nan
            row[f"mac{k + 1}"] = float(np.abs(np.vdot(best["shape"], modes[k]["shape"])) ** 2 /
                                       np.real(np.vdot(best["shape"], best["shape"]) * np.vdot(modes[k]["shape"], modes[k]["shape"]))) if best else np.nan
        rows.append(row)
    ems = pd.DataFrame(rows)
    ems.to_csv(paths["processed"] / "ems_modal.csv", index=False)
    outputs.append(paths["processed"] / "ems_modal.csv")
    fcols = [f"f{k}" for k in range(1, len(modes) + 1)]
    ems["month"] = pd.to_datetime(ems["timestamp"]).dt.to_period("M").astype(str)
    counts["mac_diagnostic_median_by_month"] = ems.groupby("month")[[c.replace("f", "mac") for c in fcols]].median().round(3).to_dict()
    counts["identification_rate"] = {c: float(ems[c].notna().mean()) for c in fcols}

    # temperature compensation --------------------------------------------------------------------
    train = (ems["timestamp"] < TRAIN_END) & ems["temperature_median"].notna()
    T = ems["temperature_median"].to_numpy()
    design = np.column_stack([np.ones(len(ems)), T, np.minimum(T, 0.0)])  # hinge at 0 degC (asphalt stiffening)
    resid = pd.DataFrame(index=ems.index)
    fit = {}
    for c in fcols:
        m = train & ems[c].notna()
        coef, *_ = np.linalg.lstsq(design[m], ems.loc[m, c].to_numpy(), rcond=None)
        pred = design @ coef
        r2 = 1 - np.sum((ems.loc[m, c] - pred[m]) ** 2) / np.sum((ems.loc[m, c] - ems.loc[m, c].mean()) ** 2)
        resid[c] = ems[c] - pred
        fit[c] = {"coef_intercept_slope_hinge": coef.tolist(), "train_r2": float(r2),
                  "corr_with_temperature_train": float(np.corrcoef(T[m], ems.loc[m, c])[0, 1])}
    # PCA alternative (no temperature input): remove the first principal component of the training data
    fm = train & ems[fcols].notna().all(axis=1)
    mu, sd = ems.loc[fm, fcols].mean(), ems.loc[fm, fcols].std()
    Z = ((ems[fcols] - mu) / sd).to_numpy()
    _, s, vt = np.linalg.svd(Z[fm.to_numpy()], full_matrices=False)
    pc1 = vt[0]
    Zf = np.nan_to_num(Z)
    pca_resid = Zf - np.outer(Zf @ pc1, pc1)
    fit["pca"] = {"explained_variance_ratio": (s ** 2 / np.sum(s ** 2)).tolist(), "pc1_loadings": pc1.tolist()}

    # daily novelty series -------------------------------------------------------------------------
    ems["date"] = pd.to_datetime(ems["timestamp"]).dt.floor("D")
    variants = {}
    for name, frame in (("raw", ems[fcols]), ("regression", resid[fcols]),
                        ("pca", pd.DataFrame(pca_resid, columns=fcols, index=ems.index).where(ems[fcols].notna()))):
        daily = frame.groupby(ems["date"]).mean()
        daily = daily[daily.notna().sum(axis=1) >= 2]
        variants[name] = daily
    det_rows, series = [], {}
    for name, daily in variants.items():
        dates = daily.index.to_pydatetime()
        n_base = int(np.sum(dates < TRAIN_END))
        z, _, _ = standardise_by_baseline(daily.to_numpy(float), n_base)
        series[name] = pd.DataFrame(z, index=daily.index, columns=[f"z_{c}" for c in fcols])
        for method in ("BOCPD", "CUSUM"):
            if method == "BOCPD":
                r = run_sequence_bocpd(z, float(cfg["sequence_bocpd"]["primary_hazard_rate"]) / 9,
                                       kappa0=float(cfg["sequence_bocpd"]["primary_kappa0"]))
                series[name][f"{method}_rmap"] = r["map_run_length"]
            else:
                r = run_cusum(z, float(cfg["cusum"]["k"]), float(cfg["cusum"]["primary_h"]), int(cfg["cusum"]["warmup"]))
                series[name][f"{method}_stat"] = r["statistic"]
            dets = [d for d in r["detections"]]
            det_dates = [dates[d["t"]] for d in dets]
            after_train = [d for d in det_dates if d >= TRAIN_END]
            first_after_onset = next((d for d in det_dates if d >= ONSET_START), None)
            det_rows.append({"compensation": name, "method": method, "days": len(daily), "training_days": n_base,
                             "alarms_total": len(dets), "alarms_in_training_period": sum(d < TRAIN_END for d in det_dates),
                             "alarms_validation_jul01_aug04": sum(TRAIN_END <= d < ONSET_START for d in det_dates),
                             "first_alarm_after_training": str(after_train[0].date()) if after_train else None,
                             "first_alarm_on_or_after_aug05": str(first_after_onset.date()) if first_after_onset else None,
                             "delay_days_from_aug05": (first_after_onset - ONSET_START).days if first_after_onset else None,
                             "alarm_dates": json.dumps([str(d.date()) for d in det_dates])})
    det = pd.DataFrame(det_rows)
    det.to_csv(out_dir / "ems_detection.csv", index=False)
    pd.concat(series, axis=1).to_csv(out_dir / "ems_daily_series.csv")
    write_json(out_dir / "ems_summary.json", {"counts": counts, "temperature_model": fit,
                                              "train_end": str(TRAIN_END.date()), "onset_window": [str(ONSET_START.date()), str(ONSET_END.date())],
                                              "first_timestamp": str(t0), "last_timestamp": str(ok[-1]["timestamp"])})
    outputs += [out_dir / "ems_detection.csv", out_dir / "ems_daily_series.csv", out_dir / "ems_summary.json"]

    # figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 6.5, "xtick.labelsize": 5.8, "ytick.labelsize": 5.8,
                         "axes.linewidth": .6})
    fig, axes = plt.subplots(3, 1, figsize=(3.15, 3.2), sharex=True)
    ts = pd.to_datetime(ems["timestamp"])
    axes[0].scatter(ts, ems["f1"], s=.3, lw=0, color="#2a78d6")
    axes[0].set_ylabel("f1 [Hz]\n(hourly)")
    axes[1].scatter(ts, ems["temperature_median"], s=.3, lw=0, color="#eb6834")
    axes[1].set_ylabel("Median temp.\n[°C]")
    d = variants["regression"]
    axes[2].plot(d.index, (d["f1"] - d.loc[d.index < TRAIN_END, "f1"].mean()) / d.loc[d.index < TRAIN_END, "f1"].std(),
                 color="#2a78d6", lw=1)
    axes[2].set_ylabel("Compensated f1\n(daily, σ units)")
    for ax in axes:
        ax.axvspan(ONSET_START, ONSET_END, color="#e34948", alpha=.15, lw=0)
        ax.axvline(TRAIN_END, color="#52514e", ls="--", lw=.8)
        ax.grid(True, color="#e4e3df", lw=.5)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
    fig.align_ylabels(axes)
    fig.autofmt_xdate()
    fig_path = paths["results"] / "figures" / "paper" / "fig_ems.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    outputs.append(fig_path)
    print(json.dumps(counts, default=str, indent=1))
    print(json.dumps(fit, indent=1))
    print(det.drop(columns=["alarm_dates"]).to_string())
    run.finish(outputs)
    print("run:", run.run_id)


if __name__ == "__main__":
    main()
