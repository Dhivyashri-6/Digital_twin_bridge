"""Cross-check of the FDD natural frequencies and EFDD damping with covariance-driven SSI (AVT records).

SSI-COV is applied to the reference sensors present in every setup (signals decimated to 50 Hz), with model orders
20-60 and a stabilisation check. For each FDD mode of a record, the stable SSI pole closest in frequency (within 5%) is
taken; the script reports how often a pole was found and the frequency and damping differences to FDD/EFDD.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import signal

from src.modal_analysis import ssi_cov, stable_poles
from src.preprocessing import preprocess_signal
from src.reproducibility import start_run
from src.utils import ensure_output_dirs, load_config, write_json
from src.z24_catalog import build_catalog, load_recording


def ssi_one(rec, cfg: dict) -> dict:
    fs = float(cfg["dataset"]["sampling_rate"])
    subset = list(cfg["modal"]["reference_sensors"])
    loaded = load_recording(rec, fs)
    pre = dict(cfg["preprocessing"], normalization="none")
    idx = [loaded.sensors.index(s) for s in subset if s in loaded.sensors]
    x = np.column_stack([preprocess_signal(loaded.data[:, i], fs, pre) for i in idx])
    x = signal.decimate(x, 2, axis=0, zero_phase=True)
    st = stable_poles(ssi_cov(x, fs / 2))
    return {"recording": rec.recording_id, "stable": st}


def main():
    cfg = load_config()
    paths = ensure_output_dirs(cfg)
    run = start_run("ssi", cfg)
    out = paths["results"] / "ssi"
    out.mkdir(parents=True, exist_ok=True)
    recs = [r for r in build_catalog(cfg["dataset"]["root_path"])[0] if r.test == "avt"]
    res = Parallel(n_jobs=4, verbose=5)(delayed(ssi_one)(r, cfg) for r in recs)
    rec = pd.read_csv(paths["processed"] / "recordings.csv", dtype={"scenario": str}).set_index("recording")
    K = sum(1 for c in rec.columns if c.startswith("f") and c[1:].isdigit())
    rows = []
    for r in res:
        st = r["stable"]
        for k in range(1, K + 1):
            f_fdd = rec.loc[r["recording"], f"f{k}"]
            row = {"recording": r["recording"], "scenario": rec.loc[r["recording"], "scenario"], "mode": k, "f_fdd": f_fdd,
                   "zeta_efdd": rec.loc[r["recording"], f"zeta{k}"], "f_ssi": np.nan, "zeta_ssi": np.nan, "n_stable": 0}
            if np.isfinite(f_fdd) and len(st):
                c = st[np.abs(st[:, 1] - f_fdd) / f_fdd < 0.05]
                if len(c):
                    row.update({"f_ssi": float(np.median(c[:, 1])), "zeta_ssi": float(np.median(c[:, 2])), "n_stable": int(len(c))})
            rows.append(row)
    df = pd.DataFrame(rows)
    df["rel_diff_pct"] = 100 * (df.f_ssi - df.f_fdd) / df.f_fdd
    df.to_csv(out / "ssi_vs_fdd.csv", index=False)
    summary = {}
    for k, g in df.groupby("mode"):
        has = g.dropna(subset=["f_fdd"])
        both = has.dropna(subset=["f_ssi"])
        zz = both.dropna(subset=["zeta_efdd"])
        summary[f"mode{k}"] = {"records_with_fdd": int(len(has)), "ssi_found_rate": float(len(both) / max(len(has), 1)),
                               "median_abs_rel_diff_pct": float(both.rel_diff_pct.abs().median()) if len(both) else None,
                               "max_abs_rel_diff_pct": float(both.rel_diff_pct.abs().max()) if len(both) else None,
                               "median_zeta_ssi_pct": float(100 * both.zeta_ssi.median()) if len(both) else None,
                               "median_zeta_efdd_pct": float(100 * zz.zeta_efdd.median()) if len(zz) else None,
                               "frequency_correlation": float(np.corrcoef(both.f_fdd, both.f_ssi)[0, 1]) if len(both) > 2 else None}
    write_json(out / "ssi_summary.json", summary)
    print(pd.DataFrame(summary).T.to_string())
    run.finish([out / "ssi_vs_fdd.csv", out / "ssi_summary.json"])


if __name__ == "__main__":
    main()
