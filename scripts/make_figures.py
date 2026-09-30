"""Stage 6: paper figures (300 dpi, single-column width 8 cm) generated only from the saved result files."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, Patch

from src.reproducibility import start_run
from src.utils import ensure_output_dirs, load_config

# reference palette (light mode), categorical slots in fixed order
S1, S2, S3, S4, S5, S6, S7, S8 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
GOOD, WARN, SERIOUS, CRIT = "#0ca30c", "#fab219", "#ec835a", "#d03b3b"
MODEL_COLOR = {"xgboost": S1, "random_forest": S2, "svm": S3, "logistic_regression": S4, "freq_threshold": S7}
MODEL_NAME = {"xgboost": "XGBoost", "random_forest": "Random forest", "svm": "SVM (RBF)",
              "logistic_regression": "Logistic regr.", "freq_threshold": "Freq.-shift threshold"}
ORDER = ["xgboost", "random_forest", "svm", "logistic_regression", "freq_threshold"]
COL_W = 3.15  # inches (8.0 cm column)

plt.rcParams.update({"font.family": "serif", "font.size": 6.5, "axes.titlesize": 6.5, "legend.fontsize": 5.8,
                     "axes.labelsize": 6.5, "xtick.labelsize": 5.8, "ytick.labelsize": 5.8, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": .5, "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "savefig.dpi": 300, "savefig.bbox": "tight", "lines.linewidth": 1.0,
                     "axes.linewidth": .6, "xtick.major.width": .6, "ytick.major.width": .6})


def save(fig, path, outs):
    fig.savefig(path)
    plt.close(fig)
    outs.append(path)


def scenario_ticks(ax, seq):
    s = seq["scenario"].to_numpy()
    bounds = [i for i in range(1, len(s)) if s[i] != s[i - 1]]
    for b in bounds:
        ax.axvline(b - .5, color=GRID, lw=.8, zorder=0)
    starts, ends = [0] + bounds, bounds + [len(s)]
    ax.set_xticks([(a + b - 1) / 2 for a, b in zip(starts, ends)])
    ax.set_xticklabels([s[a] for a in starts])
    ax.grid(axis="x", visible=False)


def fig_natural_frequencies(R, rec, ref_sc, out, outs):
    seq = pd.read_csv(R / "change_detection" / "sequence_series.csv", dtype={"scenario": str, "setup": str})
    seq = seq.merge(rec[["recording", "f1", "f2", "f3", "f4"]], on="recording", how="left")
    fig, axes = plt.subplots(4, 1, figsize=(COL_W, 4.2), sharex=True)
    isref = seq["scenario"].isin(ref_sc)
    for ax, f in zip(axes, ["f1", "f2", "f3", "f4"]):
        ax.plot(np.arange(len(seq)), seq[f], color=S1, lw=.8, zorder=2)
        ax.scatter(np.flatnonzero(~isref), seq.loc[~isref, f], s=4, color=S1, zorder=3, label="Altered", edgecolor=SURF, lw=.4)
        ax.scatter(np.flatnonzero(isref), seq.loc[isref, f], s=5, marker="s", color=S2, zorder=3, label="Reference",
                   edgecolor=SURF, lw=.4)
        ax.set_ylabel(f"{f} [Hz]")
        scenario_ticks(ax, seq)
    axes[0].legend(loc="lower center", bbox_to_anchor=(.5, 1.0), ncol=2)
    axes[-1].set_xlabel("Z24 scenario (AVT setups 01–09 in order)")
    fig.align_ylabels(axes)
    save(fig, out / "fig_natural_frequencies.png", outs)
    return seq


def fig_sequence_detection(R, cfg, seq, out, outs):
    comp = pd.read_csv(R / "change_detection" / "sequence_detection_comparison.csv")
    h, hc = cfg["sequence_bocpd"]["primary_hazard_rate"], cfg["cusum"]["primary_h"]
    pb = comp[(comp.method == "BOCPD") & comp.primary].iloc[0]
    pc = comp[(comp.method == "CUSUM") & comp.primary].iloc[0]
    fig, axes = plt.subplots(3, 1, figsize=(COL_W, 3.6), sharex=True)
    for f, c in zip(["z_f1", "z_f2", "z_f3", "z_f4"], [S1, S2, S3, S4]):
        if f in seq:
            axes[0].plot(seq[f], color=c, lw=.9, label=f.replace("z_", ""))
    axes[0].set_ylabel("Standardised\nfrequency [σ]")
    axes[0].legend(ncol=4, loc="lower center", bbox_to_anchor=(.5, 1.0))
    axes[1].plot(seq[f"bocpd_rmap_h{h}"], color=S1, lw=1)
    for d in json.loads(pb["detection_list"]):
        axes[1].plot(d["t"], seq[f"bocpd_rmap_h{h}"][d["t"]], "v", color=S8, ms=4)
    axes[1].set_ylabel(f"BOCPD MAP run\nlength (h={h})")
    axes[2].plot(seq[f"cusum_stat_h{hc}"], color=S7, lw=1)
    axes[2].axhline(hc, color=INK2, ls="--", lw=.7)
    for d in json.loads(pc["detection_list"]):
        axes[2].plot(d["t"], seq[f"cusum_stat_h{hc}"][d["t"]], "v", color=S8, ms=4)
    axes[2].set_ylabel(f"CUSUM statistic\n(k={cfg['cusum']['k']}, h={hc})")
    for ax in axes:
        scenario_ticks(ax, seq)
    axes[-1].set_xlabel("Z24 scenario; ▼ = detection")
    fig.align_ylabels(axes)
    save(fig, out / "fig_sequence_detection.png", outs)


def ci_panel(ax, res, labels, metric, index_cols):
    for j, scheme in enumerate(["loso", "group_kfold"]):
        if not (res.scheme == scheme).any():
            continue
        sub = res[res.scheme == scheme].set_index(index_cols).loc[labels]
        yy = np.arange(len(labels)) + (j - .5) * .36
        err = np.vstack([sub[metric] - sub[f"{metric}_ci_low"], sub[f"{metric}_ci_high"] - sub[metric]])
        ax.errorbar(sub[metric], yy, xerr=err, fmt="o" if j == 0 else "s", color=S1 if j == 0 else S2, ms=3, capsize=1.5,
                    lw=.9, label="Leave-one-scenario-out" if j == 0 else "Grouped 5-fold (recording)")
    ax.axvline(.5, color=INK2, ls=":", lw=.7)


def fig_classification(R, out, outs):
    p = R / "classification" / "main_results.csv"
    if not p.exists():
        return
    res = pd.read_csv(p)
    res = res[res.level == "window"]
    fig, axes = plt.subplots(1, 2, figsize=(COL_W, 2.1), sharey=True)
    for ax, metric, title in zip(axes, ["balanced_accuracy", "roc_auc"], ["Balanced accuracy", "ROC-AUC"]):
        ci_panel(ax, res, ORDER, metric, "model")
        ax.set_title(title)
        ax.set_xlim(0.2, 1.02)
    axes[0].set_yticks(range(len(ORDER)))
    axes[0].set_yticklabels([MODEL_NAME[m] for m in ORDER])
    axes[0].invert_yaxis()
    axes[0].legend(loc="lower center", bbox_to_anchor=(1.0, 1.08), ncol=2)
    save(fig, out / "fig_classification_comparison.png", outs)


def fig_ablation(R, out, outs):
    p = R / "classification" / "ablation_results.csv"
    if not p.exists():
        return
    abl = pd.read_csv(p)
    abl = abl[abl.level == "window"]
    labels = [("time", "none"), ("spectral", "none"), ("modal", "none"), ("time+spectral", "none"),
              ("time+spectral", "zscore"), ("all", "none"), ("all", "zscore")]
    fig, ax = plt.subplots(figsize=(COL_W, 2.2))
    ci_panel(ax, abl, labels, "roc_auc", ["features", "normalisation"])
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels([f"{f}{' + z-score' if n == 'zscore' else ''}" for f, n in labels])
    ax.invert_yaxis()
    ax.set_xlabel("ROC-AUC (XGBoost), 95% bootstrap CI over recordings")
    if abl.scheme.nunique() > 1:
        ax.legend(loc="lower center", bbox_to_anchor=(.5, 1.0), ncol=2)
    save(fig, out / "fig_ablation.png", outs)


def fig_per_scenario(R, ref_sc, out, outs):
    p = R / "classification" / "main_per_scenario.csv"
    if not p.exists():
        return
    ps = pd.read_csv(p, dtype={"scenario": str})
    fig, ax = plt.subplots(figsize=(COL_W, 2.0))
    width = .16
    for j, m in enumerate(ORDER):
        sub = ps[(ps.model == m) & (ps.scheme == "loso")].sort_values("scenario")
        ax.bar(np.arange(len(sub)) + (j - 2) * width, sub["recording_accuracy"], width * .9, color=MODEL_COLOR[m], label=MODEL_NAME[m])
    sc = sorted(ps["scenario"].unique())
    ax.set_xticks(range(len(sc)))
    ax.set_xticklabels([f"{s}\n{'REF' if s in ref_sc else 'ALT'}" for s in sc])
    ax.set_ylabel("Recording accuracy\n(held-out scenario)")
    ax.set_ylim(0, 1.05)
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(.5, 1.0))
    save(fig, out / "fig_per_scenario_loso.png", outs)


def fig_shap(R, out, outs):
    p = R / "shap" / "global_feature_importance.csv"
    q = R / "shap" / "importance_by_sensor_group.csv"
    if not (p.exists() and q.exists()):
        return
    g = pd.read_csv(p).head(12).iloc[::-1]
    grp = pd.read_csv(q).sort_values("channel_attribution")
    direc = pd.read_csv(R / "shap" / "importance_by_sensor_direction.csv").sort_values("channel_attribution")
    fig, axes = plt.subplots(1, 2, figsize=(COL_W, 2.5), gridspec_kw={"width_ratios": [1.25, 1]})
    cmap = {"modal": S1, "spectral": S3, "time": S2}
    axes[0].barh(g["feature"].str.replace("band_energy_", "E ").str.replace("_", " "), g["mean_abs_shap"],
                 color=[cmap[x] for x in g["group"]], height=.7)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("mean |SHAP| (log scale)")
    axes[0].legend(handles=[Patch(color=c, label=k) for k, c in cmap.items()], loc="lower center",
                   bbox_to_anchor=(.35, 1.0), ncol=3, frameon=False, handlelength=1, columnspacing=.8)
    rows = [(r.sensor_group.replace("line_", "line "), r.channel_attribution) for r in grp.itertuples()]
    rows += [("", np.nan)] + [(r.sensor_direction, r.channel_attribution) for r in direc.itertuples()]
    yy = np.arange(len(rows))
    axes[1].barh(yy, [v for _, v in rows], color=S1, height=.6)
    axes[1].set_yticks(yy)
    axes[1].set_yticklabels([k for k, _ in rows])
    axes[1].set_xlim(0, 0.035)
    axes[1].set_xticks([0, 0.015, 0.03])
    axes[1].set_xlabel("channel |SHAP|")
    for ax in axes:
        ax.grid(axis="y", visible=False)
    fig.tight_layout(w_pad=.6)
    save(fig, out / "fig_shap.png", outs)


def fig_fdd(cfg, paths, out, outs):
    from src.modal_analysis import fdd
    from src.preprocessing import preprocess_signal
    from src.z24_catalog import build_catalog, load_recording
    recs, _ = build_catalog(cfg["dataset"]["root_path"])
    pre = dict(cfg["preprocessing"], normalization="none")
    fig, ax = plt.subplots(figsize=(COL_W, 1.8))
    lo = 0.0
    for scn, c in ((1, S1), (6, S2), (17, S3)):
        specs = []
        for r in recs:
            if r.scenario == scn and r.test == "avt":
                L = load_recording(r, 100.0)
                x = np.column_stack([preprocess_signal(L.data[:, i], 100.0, pre) for i in range(L.data.shape[1])])
                res = fdd(x, 100.0, L.sensors, nperseg=cfg["modal"]["nperseg"])
                specs.append(np.log10(res.s[:, 0] / res.s[:, 0].max()))
        mean = np.mean(specs, axis=0)
        band = (res.freqs >= 1.5) & (res.freqs <= 15)
        lo = min(lo, float(mean[band].min()))
        ax.plot(res.freqs, mean, color=c, lw=.9, label=f"S{scn:02d}")
    for f in json.loads((paths["processed"] / "modal_baseline.json").read_text()):
        ax.axvline(f["frequency_hz"], color=INK2, ls=":", lw=.6)
    ax.set_xlim(1.5, 15)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylim(np.floor(lo) - 0.2, 0.9)
    ax.set_ylabel(r"Normalised $s_1$ [log$_{10}$]")
    ax.legend(loc="upper right", ncol=3, frameon=False)
    save(fig, out / "fig_fdd_spectrum.png", outs)


def fig_states(R, ref_sc, out, outs):
    p = R / "risk" / "scenario_states.csv"
    if not p.exists():
        return
    st = pd.read_csv(p, dtype={"scenario": str})
    st = st[st.variant == "gated"].sort_values("scenario")
    fig, ax = plt.subplots(figsize=(COL_W, 1.8))
    x = np.arange(len(st))
    bottom = np.zeros(len(st))
    for col, c, lab in (("normal", GOOD, "NORMAL"), ("watch", WARN, "WATCH"), ("potential_anomaly", CRIT, "POTENTIAL ANOMALY")):
        ax.bar(x, st[col], .7, bottom=bottom, color=c, label=lab, edgecolor=SURF, lw=.8)
        bottom += st[col].to_numpy()
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}\n{'REF' if s in ref_sc else 'ALT'}" for s in st["scenario"]])
    ax.set_ylabel("AVT recordings")
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(.5, 1.0))
    ax.grid(axis="x", visible=False)
    save(fig, out / "fig_twin_states.png", outs)


def fig_pipeline(out, outs):
    fig, ax = plt.subplots(figsize=(16.9 / 2.54, 7.2 / 2.54))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 44)
    ax.axis("off")

    def box(x, y, w, h, text, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.0", fc=color + "22", ec=color, lw=.8))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=4.8, color=INK, linespacing=1.2)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", lw=.6, color=INK2, shrinkA=0, shrinkB=0))

    W, H = 16, 10
    box(0.5, 17, W, H, "Z24 PDT data\n.mat / .aaa / .zip\n17 scenarios, AVT+FVT,\n9 setups", S7)
    box(20.5, 17, W, H, "Catalogue and\nduplicate check\nunique IDs, 100 Hz", S7)
    box(40.5, 30, W, H, "Preprocessing\ndetrend, band-pass\n0.5–40 Hz; 10 s windows", S7)
    box(40.5, 4, W, H, "Modal identification\nFDD / EFDD\nf1–f4, ζ1–ζ2", S1)
    box(60.5, 30, W, H, "Window features\ntime + spectral;\nwithin-record BOCPD", S1)
    box(60.5, 17, W, H, "Classifiers\nXGBoost, RF, SVM, LR\nLOSO / grouped CV", S3)
    box(60.5, 4, W, H, "Scenario sequence:\nBOCPD vs CUSUM;\nEMS temp. compens.", S2)
    box(80.5, 23.5, W, H, "TreeSHAP\nfeature and sensor\nranking", S3)
    box(80.5, 7, W, H, "Gated risk score\n(AUC ≥ 0.7)\ndigital-twin states", S4)
    for a in ((16.8, 22, 20.2, 22), (36.8, 24, 40.2, 33), (36.8, 20, 40.2, 10), (56.8, 35, 60.2, 35), (56.8, 33, 60.2, 24),
              (56.8, 11, 60.2, 20), (56.8, 9, 60.2, 9), (76.8, 24, 80.2, 28), (76.8, 20, 80.2, 14), (76.8, 9, 80.2, 11),
              (88.5, 23.2, 88.5, 17.3)):
        arrow(*a)
    save(fig, out / "fig_pipeline.png", outs)


def main():
    cfg = load_config()
    paths = ensure_output_dirs(cfg)
    run = start_run("figures", cfg)
    out = paths["results"] / "figures" / "paper"
    out.mkdir(parents=True, exist_ok=True)
    outs: list = []
    R = paths["results"]
    rec = pd.read_csv(paths["processed"] / "recordings.csv", dtype={"scenario": str, "setup": str})
    ref_sc = {f"{s:02d}" for s in cfg["dataset"]["reference_scenarios"]}
    seq = fig_natural_frequencies(R, rec, ref_sc, out, outs)
    fig_sequence_detection(R, cfg, seq, out, outs)
    fig_classification(R, out, outs)
    fig_ablation(R, out, outs)
    fig_per_scenario(R, ref_sc, out, outs)
    fig_shap(R, out, outs)
    fig_states(R, ref_sc, out, outs)
    fig_pipeline(out, outs)
    if "--skip-fdd" not in sys.argv:
        fig_fdd(cfg, paths, out, outs)
    run.finish(outs)
    print("\n".join(map(str, outs)))


if __name__ == "__main__":
    main()
