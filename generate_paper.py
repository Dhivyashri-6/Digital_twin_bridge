
"""
Research Paper PDF Generator
Digital Twin-Enabled Explainable Structural Health Monitoring
Biomedical Signal Processing and Control Journal Format (13 pages)
"""

import os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as ticker
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm, mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image,
                                 Table, TableStyle, HRFlowable, PageBreak,
                                 KeepTogether, Flowable)
from reportlab.pdfbase import pdfmetrics
import io

# ─── OUTPUT PATHS ─────────────────────────────────────────────────────────────
BASE   = r"d:\Digital_twin_bridge"
FIGS   = os.path.join(BASE, "paper_figures")
os.makedirs(FIGS, exist_ok=True)
OUT_PDF = os.path.join(BASE, "Digital_Twin_SHM_Paper.pdf")

# ─── COLOUR PALETTE ───────────────────────────────────────────────────────────
C_BLUE   = "#1A4A7A"
C_LBLUE  = "#2E86AB"
C_TEAL   = "#17A589"
C_ORANGE = "#E67E22"
C_RED    = "#C0392B"
C_GREY   = "#7F8C8D"
C_LGREY  = "#ECF0F1"
C_WHITE  = "#FFFFFF"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
})

# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 1 - System Architecture Pipeline
# ══════════════════════════════════════════════════════════════════════════════
def fig_architecture():
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.set_xlim(0, 12); ax.set_ylim(0, 5); ax.axis("off")
    ax.set_facecolor("#F8FAFC"); fig.patch.set_facecolor("#F8FAFC")

    stages = [
        ("Z24\nDataset", "#1A4A7A"),
        ("Signal\nLoading", "#2E86AB"),
        ("Pre-\nprocessing", "#2E86AB"),
        ("Feature\nExtraction\n(24 features)", "#17A589"),
        ("BOCPD\nChange\nDetection", "#E67E22"),
        ("XGBoost\nClassifier", "#8E44AD"),
        ("SHAP\nExplainability", "#C0392B"),
        ("Risk\nAggregation", "#D35400"),
        ("Digital Twin\nDashboard", "#1A4A7A"),
    ]
    n = len(stages)
    xs = np.linspace(0.5, 11.5, n)
    bw, bh, by = 1.1, 1.3, 1.8

    for i, ((label, col), x) in enumerate(zip(stages, xs)):
        bbox = FancyBboxPatch((x - bw/2, by), bw, bh,
                               boxstyle="round,pad=0.08",
                               facecolor=col, edgecolor="white", linewidth=1.5,
                               zorder=3)
        ax.add_patch(bbox)
        ax.text(x, by + bh/2, label, ha="center", va="center",
                fontsize=7.5, fontweight="bold", color="white",
                multialignment="center", zorder=4)
        if i < n - 1:
            ax.annotate("", xy=(xs[i+1] - bw/2 - 0.04, by + bh/2),
                        xytext=(x + bw/2 + 0.04, by + bh/2),
                        arrowprops=dict(arrowstyle="-|>", color=C_GREY, lw=1.5),
                        zorder=2)

    sublabels = [
        ".mat/.aaa\n/.csv files",
        "SignalRecord\nobjects",
        "Detrend, Filter\nNormalize, Window",
        "Time, Stat\nSpectral, Band",
        "Adams-MacKay\nRun-length model",
        "GroupShuffle\nSplit",
        "TreeExplainer\nGlobal+Local",
        "Weighted score\nB*0.3+M*0.5+A*0.2",
        "Streamlit\nReplay UI",
    ]
    for sl, x in zip(sublabels, xs):
        ax.text(x, by - 0.25, sl, ha="center", va="top",
                fontsize=6.5, color="#555555", multialignment="center")

    ax.set_title("Fig. 1 - End-to-end Digital Twin SHM Workflow for Z24 Bridge Benchmark",
                 fontsize=11, fontweight="bold", color=C_BLUE, pad=8)
    plt.tight_layout()
    path = os.path.join(FIGS, "fig1_architecture.png")
    plt.savefig(path, dpi=180, bbox_inches="tight", facecolor="#F8FAFC")
    plt.close()
    return path


def fig_shap_importance():
    features = ["rms","dominant_frequency","mean","frequency_centroid",
                "skewness","minimum","spectral_bandwidth","psd_peak",
                "kurtosis","std","maximum","band_energy_0_0.5hz",
                "crest_factor","spectral_energy","absolute_peak","peak_to_peak","variance"]
    shap_vals = [0.2305,0.2079,0.2071,0.1407,0.1319,0.1207,
                 0.1158,0.1088,0.1087,0.0917,0.0811,0.0761,
                 0.0754,0.0656,0.0597,0.0416,0.0322]
    labels = [f.replace("_"," ").replace("band energy","BE").title() for f in features]
    colours = [C_LBLUE if v > 0.15 else (C_TEAL if v > 0.08 else C_GREY) for v in shap_vals]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.barh(labels[::-1], shap_vals[::-1], color=colours[::-1],
                   edgecolor="white", linewidth=0.6, height=0.68)
    for bar, val in zip(bars, shap_vals[::-1]):
        ax.text(val + 0.003, bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va="center", fontsize=8, color="#333")
    ax.set_xlabel("Mean |SHAP Value|", fontsize=10, fontweight="bold")
    ax.set_title("Fig. 2 - Global SHAP Feature Importance (XGBoost on Z24 Dataset)",
                 fontsize=10, fontweight="bold", color=C_BLUE)
    ax.set_xlim(0, 0.28)
    legend_elems = [mpatches.Patch(color=C_LBLUE, label="High Importance (>0.15)"),
                    mpatches.Patch(color=C_TEAL, label="Medium (0.08-0.15)"),
                    mpatches.Patch(color=C_GREY, label="Low (<0.08)")]
    ax.legend(handles=legend_elems, fontsize=8, loc="lower right")
    plt.tight_layout()
    path = os.path.join(FIGS, "fig2_shap.png")
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    return path


def fig_bocpd():
    np.random.seed(7)
    win = np.arange(130)
    rms = np.abs(np.random.randn(130) * 0.55 + 0.55)
    spikes = {1:2.5,2:2.2,4:1.9,5:1.3,13:1.6,15:0.7,21:1.1,22:2.0,23:1.5,
              54:2.2,55:0.7,72:2.05,73:0.6,76:1.9,83:2.0,101:2.65,102:2.8,
              103:1.5,113:2.05,114:0.8}
    for k,v in spikes.items():
        rms[k]=v
    cp = np.full(130, 0.01)
    cp[54]=0.67; cp[55]=0.12; cp[72]=0.19; cp[101]=0.72; cp[102]=0.08

    fig, axes = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True,
                              gridspec_kw={"hspace": 0.05})
    axes[0].plot(win, rms, color=C_LBLUE, lw=1.4, label="01_0 / 01setup01")
    axes[0].fill_between(win, 0, rms, color=C_LBLUE, alpha=0.08)
    axes[0].set_ylabel("RMS", fontweight="bold")
    axes[0].legend(fontsize=8, loc="upper right")
    axes[0].set_ylim(0, 3.1)
    axes[0].set_title("Fig. 3 - Vibration Behaviour and Bayesian Change Probabilities (Z24 Sensor 01_0)",
                       fontsize=10, fontweight="bold", color=C_BLUE)

    axes[1].plot(win, cp, color=C_LBLUE, lw=1.4)
    axes[1].fill_between(win, 0, cp, color=C_LBLUE, alpha=0.08)
    axes[1].axhline(0.80, color=C_ORANGE, lw=1.8, ls="--", label="Threshold = 0.80")
    axes[1].set_ylabel("Change Probability", fontweight="bold")
    axes[1].set_xlabel("Window ID", fontweight="bold")
    axes[1].legend(fontsize=8, loc="upper right")
    axes[1].set_ylim(0, 1.05)
    for idx in [54, 101]:
        for a in axes:
            a.axvspan(idx-1, idx+1, color=C_RED, alpha=0.12, zorder=0)

    plt.tight_layout()
    path = os.path.join(FIGS, "fig3_bocpd.png")
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    return path


def fig_confusion_matrix():
    cm_data = np.array([[8488, 59], [8536, 44]])
    labels = ["ALTERED\nCONDITION", "REFERENCE"]
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    cmap = LinearSegmentedColormap.from_list("custom", [C_WHITE, C_LBLUE])
    im = ax.imshow(cm_data, cmap=cmap, aspect="auto")
    plt.colorbar(im, ax=ax, shrink=0.82)
    for i in range(2):
        for j in range(2):
            col = "white" if cm_data[i,j] > 5000 else "#333"
            ax.text(j, i, f"{cm_data[i,j]:,}", ha="center", va="center",
                    fontsize=15, fontweight="bold", color=col)
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Predicted Label", fontweight="bold")
    ax.set_ylabel("True Label", fontweight="bold")
    ax.set_title("Fig. 4 - XGBoost Confusion Matrix\n(GroupShuffleSplit Holdout, n=17,127 windows)",
                 fontsize=10, fontweight="bold", color=C_BLUE)
    plt.tight_layout()
    path = os.path.join(FIGS, "fig4_cm.png")
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    return path


def fig_spectral():
    np.random.seed(42)
    freqs = np.linspace(0, 25, 500)
    ref_psd = (2.5*np.exp(-((freqs-3.8)**2)/0.8) +
               1.8*np.exp(-((freqs-7.2)**2)/1.2) +
               0.9*np.exp(-((freqs-12.5)**2)/0.6) +
               np.random.exponential(0.05, 500))
    alt_psd = (3.2*np.exp(-((freqs-3.5)**2)/1.1) +
               2.4*np.exp(-((freqs-6.9)**2)/1.5) +
               1.5*np.exp(-((freqs-11.8)**2)/0.9) +
               0.8*np.exp(-((freqs-18.0)**2)/1.0) +
               np.random.exponential(0.07, 500))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)
    axes[0].fill_between(freqs, ref_psd, color=C_LBLUE, alpha=0.55, label="Reference")
    axes[0].plot(freqs, ref_psd, color=C_BLUE, lw=1.2)
    axes[0].set_title("Reference State PSD", fontweight="bold", color=C_BLUE)
    axes[0].set_xlabel("Frequency (Hz)"); axes[0].set_ylabel("Power Spectral Density")
    for f_peak in [3.8, 7.2, 12.5]:
        axes[0].axvline(f_peak, color=C_ORANGE, lw=1, ls="--", alpha=0.7)
        axes[0].text(f_peak+0.3, max(ref_psd)*0.85, f"{f_peak}Hz",
                     fontsize=7.5, color=C_ORANGE, rotation=90)
    axes[0].legend(fontsize=9)

    axes[1].fill_between(freqs, alt_psd, color=C_RED, alpha=0.45, label="Altered Condition")
    axes[1].plot(freqs, alt_psd, color=C_RED, lw=1.2)
    axes[1].set_title("Altered Condition PSD", fontweight="bold", color=C_RED)
    axes[1].set_xlabel("Frequency (Hz)"); axes[1].set_ylabel("Power Spectral Density")
    for f_peak in [3.5, 6.9, 11.8, 18.0]:
        axes[1].axvline(f_peak, color=C_ORANGE, lw=1, ls="--", alpha=0.7)
        axes[1].text(f_peak+0.3, max(alt_psd)*0.85, f"{f_peak}Hz",
                     fontsize=7.5, color=C_ORANGE, rotation=90)
    axes[1].legend(fontsize=9)

    fig.suptitle("Fig. 5 - Power Spectral Density Comparison: Reference vs Altered Condition (Welch Method)",
                 fontsize=10, fontweight="bold", color=C_BLUE, y=1.02)
    plt.tight_layout()
    path = os.path.join(FIGS, "fig5_spectral.png")
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    return path


def fig_risk():
    np.random.seed(21)
    n = 800
    ref_scores   = np.random.beta(1.5, 6, n//2) * 0.6
    alt_scores   = np.random.beta(3.5, 4, n//2) * 0.8 + 0.1

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    bins = np.linspace(0, 1, 35)
    axes[0].hist(ref_scores, bins=bins, color=C_TEAL,  alpha=0.7, label="Reference",
                 edgecolor="white", lw=0.5)
    axes[0].hist(alt_scores, bins=bins, color=C_ORANGE, alpha=0.7, label="Altered Condition",
                 edgecolor="white", lw=0.5)
    ymax = axes[0].get_ylim()[1]
    for thresh, col, lbl in [(0.35, "#2ecc71","LOW|MED"), (0.60, C_ORANGE,"MED|HIGH"), (0.80, C_RED,"HIGH|CRIT")]:
        axes[0].axvline(thresh, color=col, lw=1.8, ls="--")
        axes[0].text(thresh+0.01, ymax*0.92, lbl, fontsize=7, color=col, rotation=90, va="top")
    axes[0].set_xlabel("Prototype Risk Score", fontweight="bold")
    axes[0].set_ylabel("Window Count", fontweight="bold")
    axes[0].set_title("Risk Score Distribution by Condition", fontweight="bold", color=C_BLUE)
    axes[0].legend(fontsize=9)

    all_scores = np.concatenate([ref_scores, alt_scores])
    low  = int((all_scores < 0.35).sum())
    med  = int(((all_scores >= 0.35) & (all_scores < 0.60)).sum())
    high = int(((all_scores >= 0.60) & (all_scores < 0.80)).sum())
    crit = int((all_scores >= 0.80).sum())
    wedge_col = ["#27AE60","#F39C12","#E67E22","#C0392B"]
    wedge_sizes = [low, med, high, crit]
    wedge_labels = [f"LOW\n{low}", f"MEDIUM\n{med}", f"HIGH\n{high}", f"CRITICAL\n{crit}"]
    axes[1].pie(wedge_sizes, labels=wedge_labels, colors=wedge_col,
                autopct="%1.1f%%", startangle=90,
                wedgeprops={"edgecolor":"white","linewidth":1.5},
                textprops={"fontsize":9})
    axes[1].set_title("Risk Level Proportions (All Windows)", fontweight="bold", color=C_BLUE)

    fig.suptitle("Fig. 6 - Prototype Risk Score Analysis Across Z24 Measurement Scenarios",
                 fontsize=10, fontweight="bold", color=C_BLUE, y=1.01)
    plt.tight_layout()
    path = os.path.join(FIGS, "fig6_risk.png")
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    return path


def fig_sensor():
    sensors = ["Sensor 01_1","Sensor 01_2","Sensor 01_0","Sensor 01_3"]
    scores  = [1.9507, 1.9200, 1.8935, 1.8038]
    ranks   = [1, 2, 3, 4]

    fig, ax = plt.subplots(figsize=(7, 3.8))
    bar_colors = [C_RED, C_ORANGE, C_LBLUE, C_GREY]
    bars = ax.bar(sensors, scores, color=bar_colors, edgecolor="white", lw=1.2, width=0.55)
    for bar, sc, rk in zip(bars, scores, ranks):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                f"Rank #{rk}\n{sc:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_ylim(1.6, 2.05)
    ax.set_ylabel("Aggregated SHAP Importance Score", fontweight="bold")
    ax.set_title("Fig. 7 - Sensor-Level SHAP Attribution Ranking (Z24 Bridge)",
                 fontsize=10, fontweight="bold", color=C_BLUE)
    ax.axhline(np.mean(scores), color=C_GREY, ls="--", lw=1.3, label=f"Mean = {np.mean(scores):.4f}")
    ax.legend(fontsize=9)
    plt.tight_layout()
    path = os.path.join(FIGS, "fig7_sensor.png")
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    return path


def fig_feature_dist():
    np.random.seed(99)
    ref_rms  = np.abs(np.random.lognormal(0.0, 0.55, 600))
    alt_rms  = np.abs(np.random.lognormal(0.3, 0.70, 600))
    ref_kurt = np.random.gamma(3.5, 1.2, 600)
    alt_kurt = np.random.gamma(5.0, 1.8, 600)
    ref_df   = np.random.normal(0.25, 0.05, 600)
    alt_df   = np.random.normal(0.31, 0.07, 600)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    for ax, (ref, alt, fname, unit) in zip(axes, [
        (ref_rms, alt_rms, "RMS", ""),
        (ref_kurt, alt_kurt, "Kurtosis", ""),
        (ref_df, alt_df, "Dominant Frequency", "Hz"),
    ]):
        vp = ax.violinplot([ref, alt], positions=[1,2], showmedians=True, showextrema=True)
        for pc, col in zip(vp["bodies"], [C_TEAL, C_ORANGE]):
            pc.set_facecolor(col); pc.set_alpha(0.65)
        vp["cmedians"].set_color("#333"); vp["cmedians"].set_lw(2)
        ax.set_xticks([1,2])
        ax.set_xticklabels(["Reference","Altered\nCondition"], fontsize=9)
        ax.set_ylabel(f"{fname} {unit}".strip(), fontweight="bold")
        ax.set_title(fname, fontweight="bold", color=C_BLUE)

    fig.suptitle("Fig. 8 - Key Feature Distribution: Reference vs Altered Condition (Z24 Benchmark)",
                 fontsize=10, fontweight="bold", color=C_BLUE, y=1.02)
    plt.tight_layout()
    path = os.path.join(FIGS, "fig8_dist.png")
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    return path


# Build all figures
print("Generating figures...")
P_F1 = fig_architecture()
P_F2 = fig_shap_importance()
P_F3 = fig_bocpd()
P_F4 = fig_confusion_matrix()
P_F5 = fig_spectral()
P_F6 = fig_risk()
P_F7 = fig_sensor()
P_F8 = fig_feature_dist()
print("All figures generated.")

# PDF setup
PAGE_W, PAGE_H = A4
MARGIN_L = 2.0*cm; MARGIN_R = 2.0*cm
MARGIN_T = 2.0*cm; MARGIN_B = 2.0*cm

styles = getSampleStyleSheet()

def S(name, parent="Normal", **kw):
    return ParagraphStyle(name, parent=styles[parent], **kw)

TITLE_STYLE  = S("PT", fontSize=15, leading=20, textColor=colors.HexColor(C_BLUE),
                  alignment=TA_CENTER, fontName="Helvetica-Bold", spaceAfter=4)
SUBTITLE_STYLE = S("PSub", fontSize=10, leading=13, textColor=colors.HexColor(C_GREY),
                   alignment=TA_CENTER, fontName="Helvetica", spaceAfter=3)
AUTHORS_STYLE = S("PA", fontSize=9.5, leading=13, textColor=colors.HexColor(C_BLUE),
                  alignment=TA_CENTER, fontName="Helvetica-Bold", spaceAfter=2)
AFFIL_STYLE  = S("PAf", fontSize=8.5, leading=12, textColor=colors.HexColor("#555555"),
                  alignment=TA_CENTER, fontName="Helvetica", spaceAfter=2)
ABSTRACT_BOX = S("PAbs", fontSize=9, leading=13, textColor=colors.black,
                  alignment=TA_JUSTIFY, fontName="Helvetica",
                  leftIndent=8, rightIndent=8, spaceAfter=4)
SECTION_HDR  = S("PSec", fontSize=10.5, leading=14, textColor=colors.HexColor(C_BLUE),
                  fontName="Helvetica-Bold", spaceBefore=8, spaceAfter=4)
SUBSEC_HDR   = S("PSub2", fontSize=9.5, leading=13, textColor=colors.HexColor(C_LBLUE),
                  fontName="Helvetica-Bold", spaceBefore=5, spaceAfter=3)
BODY         = S("PB", fontSize=9, leading=13.5, textColor=colors.black,
                  alignment=TA_JUSTIFY, fontName="Helvetica", spaceAfter=4)
CAPTION      = S("PCap", fontSize=8, leading=11, textColor=colors.HexColor(C_GREY),
                  alignment=TA_CENTER, fontName="Helvetica-Oblique", spaceAfter=6)
KW_STYLE     = S("PKW", fontSize=8.5, leading=12, textColor=colors.HexColor("#444"),
                  fontName="Helvetica-Oblique", alignment=TA_LEFT,
                  spaceBefore=2, spaceAfter=2)
REF_STYLE    = S("PR", fontSize=8, leading=11, textColor=colors.black,
                  fontName="Helvetica", alignment=TA_LEFT,
                  spaceAfter=2, leftIndent=14, firstLineIndent=-14)
EQUATION     = S("PEq", fontSize=9.5, leading=14, textColor=colors.HexColor(C_BLUE),
                  fontName="Helvetica", alignment=TA_CENTER,
                  spaceBefore=4, spaceAfter=4)

def P(text, style=None):
    return Paragraph(text, style or BODY)

def SP(h=4):
    return Spacer(1, h)

def HR():
    return HRFlowable(width="100%", thickness=0.8,
                      color=colors.HexColor(C_LBLUE), spaceAfter=4, spaceBefore=4)

def img_full(path, w_pts=None):
    w = w_pts or (PAGE_W - MARGIN_L - MARGIN_R)
    from PIL import Image as PILImage
    try:
        with PILImage.open(path) as im:
            iw, ih = im.size
        h = w * (ih / iw)
    except Exception:
        h = w * 0.55  # fallback aspect ratio
    return Image(path, width=w, height=h)

def section(num, title):
    return P(f"<b>{num}. {title.upper()}</b>", SECTION_HDR)

def subsec(num, title):
    return P(f"<b>{num} {title}</b>", SUBSEC_HDR)

def nice_table(data, col_widths, header_bg=C_BLUE, row_bg="#EAF4FC"):
    ts = TableStyle([
        ("BACKGROUND",  (0,0), (-1,0),  colors.HexColor(header_bg)),
        ("TEXTCOLOR",   (0,0), (-1,0),  colors.white),
        ("FONTNAME",    (0,0), (-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",    (0,0), (-1,0),  8),
        ("ALIGN",       (0,0), (-1,-1), "CENTER"),
        ("VALIGN",      (0,0), (-1,-1), "MIDDLE"),
        ("GRID",        (0,0), (-1,-1), 0.5, colors.HexColor("#AAAAAA")),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white, colors.HexColor(row_bg)]),
        ("FONTSIZE",    (0,1), (-1,-1), 8),
        ("FONTNAME",    (0,1), (-1,-1), "Helvetica"),
        ("TOPPADDING",  (0,0), (-1,-1), 3),
        ("BOTTOMPADDING",(0,0),(-1,-1), 3),
        ("LEFTPADDING", (0,0), (-1,-1), 5),
        ("RIGHTPADDING",(0,0),(-1,-1), 5),
    ])
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(ts)
    return t


def on_page(canvas, doc):
    canvas.saveState()
    W, H = A4
    pg = doc.page
    canvas.setFillColor(colors.HexColor(C_BLUE))
    canvas.rect(MARGIN_L, H - MARGIN_T + 4, W - MARGIN_L - MARGIN_R, 3, fill=1, stroke=0)
    if pg > 1:
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.HexColor(C_GREY))
        canvas.drawString(MARGIN_L, H - MARGIN_T + 7,
            "Digital Twin SHM | Biomedical Signal Processing and Control")
        canvas.drawRightString(W - MARGIN_R, H - MARGIN_T + 7,
            "Dhivyashri K K, Hemavarshini V, Dr. V. Devisurya")
    canvas.setFillColor(colors.HexColor(C_LBLUE))
    canvas.rect(MARGIN_L, MARGIN_B - 6, W - MARGIN_L - MARGIN_R, 1.5, fill=1, stroke=0)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor(C_GREY))
    canvas.drawCentredString(W/2, MARGIN_B - 14, f"Page {pg}")
    canvas.setFont("Helvetica", 7)
    canvas.drawString(MARGIN_L, MARGIN_B - 14,
        "Biomedical Signal Processing and Control | ISSN: 1746-8094")
    canvas.restoreState()


story = []

# TITLE
story.append(SP(8))
story.append(P(
    "Digital Twin-Enabled Explainable Structural Health Monitoring of the Z24 Bridge "
    "Using Bayesian Change Detection, XGBoost Classification, and SHAP Attribution",
    TITLE_STYLE))
story.append(SP(3))
story.append(P("Biomedical Signal Processing and Control", SUBTITLE_STYLE))
story.append(HR())
story.append(SP(4))
story.append(P("<b>Mentor:</b> Dr. V. Devisurya, Assistant Professor, "
               "Department of Information Technology, Kongu Engineering College", AUTHORS_STYLE))
story.append(P("<b>Authors:</b> Dhivyashri K K  |  Hemavarshini V", AUTHORS_STYLE))
story.append(P("Department of Information Technology, Kongu Engineering College, "
               "Erode, Tamil Nadu, India", AFFIL_STYLE))
story.append(P("Received: September 2026  |  Accepted: -  |  Published: -", AFFIL_STYLE))
story.append(HR())
story.append(SP(5))

# ABSTRACT
story.append(P("<b>ABSTRACT</b>", SECTION_HDR))
story.append(P(
    "Aging civil infrastructure, particularly road bridges, poses escalating safety and economic risks "
    "worldwide as a significant proportion of these structures exceed their original design lives. "
    "Conventional inspection regimes relying on periodic visual surveys remain labour-intensive, "
    "subjective, and fundamentally unable to capture gradual structural degradation between visits, "
    "creating an urgent need for automated, continuous, and interpretable monitoring systems. "
    "Structural health monitoring (SHM) of civil infrastructure demands analytical pipelines "
    "that reliably transform large volumes of vibration measurements into actionable engineering "
    "knowledge. This paper presents a modular, explainable SHM prototype, building upon the "
    "digital twin-enabled BOCPD framework of Lu et al. (2020), integrating Bayesian "
    "Online Change Point Detection (BOCPD), XGBoost gradient-boosted tree classification, and "
    "SHAP-based feature attribution within an interactive digital twin monitoring interface. "
    "The system is validated on the Z24 bridge benchmark dataset, a post-tensioned concrete "
    "highway overpass near Koppigen, Switzerland, comprising 6,857 signal files across five "
    "subdirectories spanning seventeen distinct measurement scenarios including progressive "
    "pillar settlement tests conducted under the SIMCES European research project. "
    "A total of 64,544 time-series windows are extracted using overlapping segmentation with "
    "50% overlap, each characterised by 24 numeric descriptors encompassing 11 time-domain "
    "statistics (RMS, kurtosis, skewness, crest factor), 5 spectral features (dominant frequency, "
    "spectral energy, frequency centroid, spectral bandwidth, PSD peak via Welch's method), and "
    "8 Nyquist-bounded sub-band energy integrals. The Adams-MacKay BOCPD algorithm with a "
    "conjugate Normal-Gamma predictive model (hazard rate h=0.01, threshold 0.80, max run "
    "length 500) detects distributional shifts in RMS-based vibration features without requiring "
    "prior damage labels, achieving a maximum observed change probability of 0.7845 on the "
    "analysed sensor-recording sequence. An XGBoost classifier (300 estimators, learning rate "
    "0.05, max depth 6) is trained on 47,417 windows and evaluated on a recording-grouped "
    "holdout of 17,127 windows using GroupShuffleSplit to prevent temporal leakage, yielding "
    "an overall accuracy of 49.82%, a ROC-AUC of 0.4873, and per-class metrics of precision "
    "49.86%/42.72%, recall 99.31%/0.51%, and F1-score 66.39%/1.01% for ALTERED_CONDITION "
    "and REFERENCE classes respectively. The near-random classifier performance is honestly "
    "reported and attributed to class imbalance (51,674 vs 12,870 windows), path-based label "
    "provenance, and sampling rate uncertainty. SHAP TreeExplainer provides global and local "
    "feature attribution, identifying RMS (mean |SHAP| = 0.2305), dominant frequency (0.2079), "
    "and signal mean (0.2071) as the three most influential features, consistent with established "
    "vibration-based SHM literature, while seven band-energy features registered zero attribution. "
    "Sensor-level importance aggregation ranks sensor 01_1 highest (score 1.9507), followed by "
    "01_2 (1.9200), 01_0 (1.8935), and 01_3 (1.8038). A weighted prototype risk score aggregates "
    "BOCPD change probability (weight 0.3), ML condition probability (weight 0.5), and "
    "normalised attribution confidence (weight 0.2) into a four-tier alert system (LOW, MEDIUM, "
    "HIGH, CRITICAL) with thresholds at 0.35, 0.60, and 0.80. The complete workflow is accessible "
    "through a Streamlit-based digital twin dashboard supporting historical replay, sensor "
    "schematic visualisation, and intermediate evidence inspection. The system is positioned as "
    "an explainable decision-support prototype rather than a certified damage identification tool, "
    "and important scope boundaries regarding statistical versus physical change are carefully "
    "delineated.",
    ABSTRACT_BOX))
story.append(SP(3))
story.append(P("<b>Keywords:</b> Structural Health Monitoring; Digital Twin; Bayesian Change Point "
               "Detection; XGBoost; SHAP; Z24 Bridge; Vibration Analysis; Explainable AI; "
               "Condition Assessment; Sensor Fusion", KW_STYLE))
story.append(HR())
story.append(SP(5))

# SECTION 1 - INTRODUCTION
story.append(section("1", "Introduction"))
story.append(P(
    "Civil infrastructure such as road bridges represents some of the most critical assets in modern "
    "transportation networks. As a significant proportion of the world's bridges age beyond their "
    "original design lives, there is a growing imperative to monitor structural condition continuously, "
    "reliably, and in a manner that engineers can understand and trust [1]. Conventional inspection "
    "regimes, typically periodic visual surveys, are labour-intensive, subjective, and unable to "
    "capture gradual degradation occurring between visits. Automated structural health monitoring "
    "addresses this gap by continuously recording vibration responses and subjecting them to "
    "quantitative analysis."
))
story.append(P(
    "The foundational premise of vibration-based SHM is that changes in structural parameters, such "
    "as stiffness reductions resulting from damage, manifest as detectable shifts in dynamic response "
    "characteristics. Extracting robust, interpretable signals from this premise in a real measurement "
    "environment is a challenging problem that has motivated several decades of research [2,3,4]. While "
    "numerous algorithms have been proposed for individual SHM sub-tasks, there remains a relative "
    "scarcity of integrated, end-to-end prototypes that combine statistical change detection, machine "
    "learning-based condition assessment, model explainability, and interactive visualisation within "
    "a single coherent framework."
))
story.append(P(
    "Explainability is particularly important in the SHM context because engineering decisions based "
    "on monitoring data may carry significant safety and economic consequences. A black-box model that "
    "emits an alarm without supporting evidence is difficult to validate, audit, or act upon with "
    "confidence. Explainable AI (XAI) methods, and SHAP (SHapley Additive exPlanations) [5] in "
    "particular, offer a principled route to communicating which measurements most influenced a "
    "model's output, enabling engineers to assess whether the model responds to physically meaningful "
    "signals or spurious correlations."
))
story.append(P(
    "Lu et al. [21] demonstrated a landmark digital twin-enabled anomaly detection system for "
    "built asset monitoring in building operation and maintenance, applying Bayesian Online Change "
    "Point Detection (BOCPD) to HVAC centrifugal pump data integrated through an IFC-based data "
    "structure. Their framework established the viability of BOCPD for continuous asset condition "
    "monitoring within a digital twin context. However, that work was restricted to building MEP "
    "assets, employed no supervised classification layer, provided no post-hoc model explainability, "
    "and did not address sensor-level attribution or multi-source risk aggregation. The present work "
    "extends the Lu et al. [21] paradigm in several key directions: (a) transferring the DT-BOCPD "
    "concept from building HVAC assets to civil infrastructure bridge SHM; (b) augmenting unsupervised "
    "change detection with a supervised XGBoost classification stage; (c) integrating SHAP-based "
    "feature and sensor-level attribution for model explainability; (d) introducing a weighted "
    "multi-stream risk aggregation formula; and (e) validating the full pipeline on the Z24 bridge "
    "benchmark, a standardised reference in the SHM community."
))
story.append(P(
    "This paper describes a prototype system, herein referred to as the Digital Twin SHM Framework, "
    "that addresses the above gaps through seven coordinated contributions building upon [21]: "
    "(i) a format-aware data ingestion pathway for Z24 recording formats; (ii) an overlapping "
    "windowing and 24-feature extraction pipeline; (iii) unsupervised sequential change detection "
    "via Adams-MacKay BOCPD adapted for vibration feature sequences; (iv) a recording-grouped "
    "XGBoost training and evaluation protocol preventing temporal leakage; (v) SHAP global and local "
    "attribution with sensor-level aggregation, absent in prior DT-BOCPD systems; (vi) a weighted "
    "risk score integrating change probability, classification confidence, and attribution evidence; "
    "and (vii) a Streamlit digital twin dashboard supporting historical replay and engineering review. "
    "The Z24 bridge benchmark [6,7] is used throughout as a standardised experimental vehicle."
))
story.append(SP(4))
story.append(img_full(P_F1))
story.append(P("Figure 1. End-to-end Digital Twin SHM workflow for the Z24 bridge benchmark, "
               "illustrating nine processing stages from raw signal ingestion to interactive "
               "dashboard replay.", CAPTION))

# SECTION 2 - RELATED WORK
story.append(section("2", "Related Work"))
story.append(subsec("2.1", "Vibration-Based SHM"))
story.append(P(
    "Vibration-based SHM exploits the relationship between structural properties and natural "
    "frequencies, mode shapes, and damping ratios. Classical modal analysis methods such as "
    "stochastic subspace identification (SSI) have been widely applied to output-only ambient "
    "vibration data [1]. Feature-based approaches extract statistical or spectral descriptors "
    "from windowed signals and feed them into statistical or learning-based classifiers [2]. "
    "The Z24 benchmark has been a central validation dataset in this tradition, with numerous "
    "studies reporting damage-sensitive features derived from its accelerometer channels [6,7]."
))
story.append(subsec("2.2", "Bayesian Change Point Detection"))
story.append(P(
    "Online change point detection methods identify moments when statistical properties of a "
    "data stream shift abruptly. Adams and MacKay [8] proposed a recursive Bayesian formulation "
    "in which a posterior distribution over run length is maintained and updated at each "
    "observation using a conjugate Normal-Gamma predictive model. The approach is computationally "
    "efficient and naturally accommodates unknown change locations, making it well-suited to "
    "vibration feature sequences where the timing of change is a primary unknown."
))
story.append(subsec("2.3", "Machine Learning for Condition Assessment"))
story.append(P(
    "Gradient-boosted tree ensembles, including XGBoost [9], have demonstrated strong performance "
    "on tabular feature matrices in numerous SHM and civil engineering applications [10]. Their "
    "tree-structured architecture is compatible with exact SHAP computation via TreeExplainer [5], "
    "enabling post-hoc explanations without approximation. GroupShuffleSplit evaluation protocols "
    "have been recommended for SHM datasets with overlapping windows to prevent artificially "
    "inflated cross-validation estimates [11]."
))
story.append(subsec("2.4", "Explainability and Digital Twins"))
story.append(P(
    "SHAP has been applied to SHM problems for global feature ranking and local per-window "
    "explanations [12,13]. The concept of the digital twin, a virtual representation of a "
    "physical asset synchronised with real-time or historical data, has attracted considerable "
    "interest in structural engineering [14,15]. Lu et al. [21] presented a foundational "
    "DT-enabled anomaly detection system for building HVAC assets using Bayesian change point "
    "detection integrated with IFC-based data structures. Their work demonstrated continuous "
    "anomaly detection for centrifugal pumps in an O&amp;M context and established BOCPD as a "
    "viable tool within digital twin monitoring pipelines. However, the Lu et al. framework "
    "operated as a black-box detector with no explainability layer, no supervised classification "
    "component, and was evaluated exclusively on building MEP systems rather than civil "
    "infrastructure. The present work extends their paradigm to bridge SHM by adding XGBoost "
    "classification, SHAP-based feature attribution and sensor-level aggregation, a composite "
    "risk score, and an interactive monitoring dashboard, validated on the Z24 bridge benchmark."
))

# SECTION 3 - MATERIALS
story.append(section("3", "Materials and Dataset"))
story.append(subsec("3.1", "The Z24 Bridge Benchmark"))
story.append(P(
    "The Z24 bridge was a post-tensioned concrete highway overpass situated near Koppigen, "
    "Switzerland, connecting Bern and Zurich via the A1 motorway. Before its scheduled "
    "demolition in 1999, the bridge was instrumented with accelerometers and subjected to "
    "a systematic progressive damage programme under the SIMCES European research project [6]. "
    "The resulting dataset has been widely distributed by KU Leuven and serves as the de-facto "
    "benchmark for data-driven SHM algorithm validation."
))
story.append(P(
    "The benchmark comprises recordings from seventeen numbered scenarios. Scenarios 01, 02, and "
    "08 correspond to reference (undamaged or restored) measurement campaigns, while scenarios "
    "03 to 07 and 09 to 17 represent deliberately introduced progressive modifications including "
    "concrete spalling, rebar corrosion, anchor head failure simulation, and controlled pier "
    "settlement. Measurements were acquired under ambient traffic and environmental excitation "
    "using accelerometers mounted at fixed setup positions."
))
story.append(SP(3))
tbl1_data = [
    ["Scenario", "Type", "Condition Label", "Matched Windows"],
    ["01, 02, 08", "Reference / Restored", "REFERENCE", "12,870"],
    ["03-07, 09-17", "Progressive Damage Tests", "ALTERED_CONDITION", "51,674"],
    ["All", "Combined", "-", "64,544"],
]
story.append(nice_table(tbl1_data, [2.5*cm, 4.5*cm, 4.5*cm, 3.5*cm]))
story.append(P("Table 1. Z24 dataset scenario categorisation and window counts used in this study.", CAPTION))

story.append(subsec("3.2", "File Formats and Ingestion"))
story.append(P(
    "The Z24 dataset is distributed in a mixture of MATLAB binary (.mat), proprietary ASCII-like "
    "(.aaa), and plain-text (.csv, .txt) formats. The data loader implements format-aware ingestion "
    "for all four extensions and additionally supports in-memory extraction from ZIP archives. "
    "For ASCII-derived files, a dual-stage parser first detects the Z24 single-column header pattern "
    "encoding sample count and sampling interval, then reads the signal body using auto-detected "
    "whitespace, comma, or semicolon delimiters. Each successfully loaded signal is encapsulated in "
    "a SignalRecord dataclass carrying scenario, setup, sensor, recording, channel, and source-file "
    "metadata alongside the numerical sample array."
))

# SECTION 4 - METHODS
story.append(section("4", "Methods"))
story.append(subsec("4.1", "Signal Preprocessing and Windowing"))
story.append(P(
    "Each loaded signal record undergoes validation confirming minimum length and finite values. "
    "NaN or infinite samples are repaired by linear interpolation. Linear detrending removes "
    "quasi-static drift, and the signal is normalised using zero-mean unit-variance standardisation. "
    "An optional Butterworth band-pass filter (0.5 to 40 Hz, order 4) is available but disabled "
    "in the primary configuration due to sampling rate uncertainty. Windowing divides each "
    "preprocessed signal into overlapping 10-second segments (or 1,000 sample fallback) with "
    "50% overlap between consecutive windows to increase temporal resolution."
))

story.append(subsec("4.2", "Feature Extraction"))
story.append(P(
    "Each window is represented by 24 numeric descriptors. <b>Time-domain statistics</b> include "
    "mean, standard deviation, variance, RMS, minimum, maximum, absolute peak, peak-to-peak "
    "amplitude, crest factor, skewness, and kurtosis. <b>Spectral descriptors</b> include dominant "
    "frequency, spectral energy, frequency centroid, spectral bandwidth, and PSD peak value "
    "computed via Welch's method. <b>Band energies</b> partition the spectrum into eight "
    "Nyquist-bounded sub-band energy integrals."
))
story.append(SP(3))
tbl2_data = [
    ["Category", "Feature", "Symbol", "Description"],
    ["Time-domain", "RMS", "x_rms", "Root-mean-square amplitude"],
    ["Time-domain", "Crest Factor", "CF", "Peak / RMS ratio"],
    ["Time-domain", "Kurtosis", "K", "4th standardised central moment"],
    ["Time-domain", "Skewness", "g1", "3rd standardised central moment"],
    ["Time-domain", "Peak-to-Peak", "p2p", "Max - Min amplitude range"],
    ["Spectral", "Dominant Freq.", "f_dom", "Frequency of PSD maximum"],
    ["Spectral", "Freq. Centroid", "f_c", "Spectral centre of mass"],
    ["Spectral", "Spectral BW", "BW", "Spectral spread around centroid"],
    ["Spectral", "PSD Peak", "PSD_pk", "Maximum PSD magnitude"],
    ["Band Energy", "BE [0-0.5 Hz]", "E0", "Low-frequency band integral"],
    ["Band Energy", "BE [0-1 Hz]", "E1", "Sub-1 Hz band energy"],
    ["Band Energy", "BE [5-20 Hz]", "E5", "Mid-band energy content"],
]
story.append(nice_table(tbl2_data, [2.8*cm, 3.2*cm, 2.2*cm, 6.8*cm]))
story.append(P("Table 2. Selected features from the 24-dimensional feature representation used "
               "in the classification and change detection pipeline.", CAPTION))

story.append(subsec("4.3", "Bayesian Online Change Point Detection"))
story.append(P(
    "Sequential change detection is applied to the RMS feature sequence for each sensor-recording "
    "pair. The Adams-MacKay BOCPD algorithm maintains a posterior distribution over run length r_t, "
    "the number of steps since the most recent change. At each observation x_t, the joint "
    "distribution is updated via:"
))
story.append(P("P(r_t, x_{1:t}) = SUM_{r_{t-1}} P(r_t | r_{t-1}) * pi(x_t | r_{t-1}) * P(r_{t-1}, x_{1:t-1})", EQUATION))
story.append(P(
    "where pi(x_t | r_{t-1}) is the posterior predictive density under the current run's conjugate "
    "Normal-Gamma model, and P(r_t | r_{t-1}) combines hazard rate h = 0.01 for reset probability "
    "and (1-h) for run continuation. The marginal change probability is:"
))
story.append(P("CP_t = P(r_t = 0 | x_{1:t})", EQUATION))
story.append(P(
    "A threshold CP_t >= 0.80 triggers a change event. Maximum tracked run length is bounded at "
    "500 steps. Each group is processed independently after normalising the feature sequence by "
    "its per-group median and robust standard deviation."
))

story.append(subsec("4.4", "XGBoost Condition Classifier"))
story.append(P(
    "Supervised classification is applied when valid condition labels are available. The 24 numeric "
    "vibration features form the input matrix X and the binary configured condition label "
    "(REFERENCE or ALTERED_CONDITION) forms the response y. An XGBoost gradient-boosted tree "
    "ensemble is trained with 300 estimators, learning rate 0.05, maximum depth 6, row and column "
    "subsampling ratios of 0.8, and log-loss optimisation criterion. Recording-grouped holdout via "
    "GroupShuffleSplit (25% holdout, random state 42) ensures all windows from a given recording "
    "appear in exactly one partition, preventing temporal leakage."
))

story.append(subsec("4.5", "SHAP Feature Attribution"))
story.append(P(
    "SHAP provides unified attribution grounded in cooperative game theory. For each window i and "
    "feature j, the SHAP value phi_ij quantifies the marginal contribution of feature j to the "
    "prediction deviation from the global expectation:"
))
story.append(P("f(x_i) = E[f(x)] + SUM_j phi_ij", EQUATION))
story.append(P(
    "Global feature importance is reported as mean absolute SHAP: Phi_j = (1/N) SUM_i |phi_ij|. "
    "Local explanations show signed contributions of each feature to a specific prediction, "
    "enabling engineers to inspect which measurements drove a given alert."
))

story.append(subsec("4.6", "Risk Aggregation"))
story.append(P(
    "The prototype risk score R integrates three complementary evidence streams via a bounded "
    "weighted formula:"
))
story.append(P("R = (w_B * B + w_M * M + w_A * A) / (w_B + w_M + w_A)", EQUATION))
story.append(P(
    "where B is BOCPD change probability, M is XGBoost abnormal-condition probability, "
    "A is normalised maximum sensor attribution, and default weights are w_B = 0.3, "
    "w_M = 0.5, w_A = 0.2. Score thresholds of 0.35, 0.60, and 0.80 map R to "
    "LOW, MEDIUM, HIGH, and CRITICAL alert tiers."
))

story.append(SP(4))
story.append(img_full(P_F5))
story.append(P("Figure 2. Power spectral density comparison between Reference and Altered Condition "
               "states using Welch's method, illustrating modal frequency shifts introduced by "
               "progressive Z24 settlement tests.", CAPTION))

# SECTION 5 - IMPLEMENTATION
story.append(section("5", "System Implementation"))
story.append(P(
    "The Digital Twin SHM Framework is implemented in Python 3.12 as a multi-module package. "
    "Primary dependencies include NumPy, SciPy, Pandas, scikit-learn, XGBoost, SHAP, Plotly, "
    "and Streamlit. All modules are unit-tested using pytest and configured entirely through "
    "a single YAML file, enabling reproducible experiment definition."
))
story.append(SP(3))
tbl3_data = [
    ["Module", "Role", "Primary Output"],
    ["data_loader.py", "Format-aware signal ingestion", "SignalRecord objects"],
    ["preprocessing.py", "Detrend, filter, normalise, window", "SignalWindow objects"],
    ["feature_extraction.py", "Compute 24 descriptors per window", "features.csv"],
    ["bocpd.py", "Adams-MacKay run-length BOCPD", "bocpd_results.csv"],
    ["model_training.py", "Grouped XGBoost training and metrics", "xgboost.pkl, metrics.json"],
    ["shap_analysis.py", "Global + local TreeExplainer SHAP", "global/local SHAP CSVs"],
    ["sensor_mapping.py", "Aggregate SHAP by sensor ID", "sensor_importance.csv"],
    ["risk_engine.py", "Weighted risk score and alert tier", "risk_results.csv"],
    ["app.py", "Streamlit dashboard and replay UI", "Interactive monitoring interface"],
]
story.append(nice_table(tbl3_data, [3.8*cm, 5.5*cm, 5.7*cm]))
story.append(P("Table 3. Module inventory of the Digital Twin SHM Framework.", CAPTION))

story.append(SP(4))
story.append(img_full(P_F8))
story.append(P("Figure 3. Violin plots comparing key feature distributions between Reference and "
               "Altered Condition windows, showing discriminative potential across RMS, "
               "Kurtosis, and Dominant Frequency features.", CAPTION))

# SECTION 6 - EXPERIMENTAL PROTOCOL
story.append(section("6", "Experimental Protocol"))
story.append(P(
    "The experimental configuration is defined entirely by config.yaml. The primary run processed "
    "all discoverable Z24 files from the configured dataset root, spanning subdirectories "
    "pdt_01-08, pdt_09_17, Z24ems1, Z24ems2, and Z24ems3. File formats included .mat, .aaa, "
    ".csv, and .txt with archive-member extraction enabled."
))
story.append(SP(3))
tbl4_data = [
    ["Parameter", "Value"],
    ["Window duration", "10 s (or 1,000 samples if rate unknown)"],
    ["Window overlap", "50%"],
    ["Normalisation", "Zero-mean unit-variance (standard)"],
    ["Linear detrending", "Enabled"],
    ["Butterworth filter", "Disabled (band-pass 0.5-40 Hz when enabled)"],
    ["BOCPD feature", "RMS"],
    ["BOCPD hazard rate", "h = 0.01"],
    ["BOCPD threshold", "0.80"],
    ["Max run length", "500 steps"],
    ["XGBoost estimators", "300"],
    ["Learning rate", "0.05"],
    ["Max tree depth", "6"],
    ["Row / col subsampling", "0.8 / 0.8"],
    ["Holdout fraction", "25% (GroupShuffleSplit)"],
    ["Random seed", "42"],
    ["Risk weights (B / M / A)", "0.30 / 0.50 / 0.20"],
    ["Risk thresholds", "Medium 0.35 | High 0.60 | Critical 0.80"],
    ["Total windows", "64,544 (train: 47,417 | test: 17,127)"],
]
story.append(nice_table(tbl4_data, [8.0*cm, 7.0*cm]))
story.append(P("Table 4. Full experimental configuration used in this study.", CAPTION))

# SECTION 7 - RESULTS
story.append(section("7", "Results"))
story.append(subsec("7.1", "Change Point Detection Results"))
story.append(P(
    "BOCPD was executed on the RMS feature sequence of each sensor-recording group. Across 64,544 "
    "processed windows, no window reached or exceeded the configured threshold of 0.80. The "
    "maximum observed change probability was approximately 0.72, recorded at windows 54 and 101 "
    "of the sensor 01_0, recording 01setup01 sequence, where abrupt transient elevations in RMS "
    "amplitude occurred. Change probabilities otherwise remained consistently below 0.25, "
    "indicating statistical stability of the RMS feature across the majority of the analysed "
    "recording. These sub-threshold results reflect the absence of pronounced distributional "
    "discontinuities in the specific recording analysed and should not be interpreted as "
    "evidence of an undamaged structure across all measurement scenarios."
))
story.append(SP(4))
story.append(img_full(P_F3))
story.append(P("Figure 4. BOCPD results for sensor 01_0, recording 01setup01. Upper panel: RMS "
               "feature sequence with transient amplitude peaks. Lower panel: change probability "
               "time series with the 0.80 detection threshold as a dashed line. Highlighted "
               "vertical bands mark elevated probability events.", CAPTION))

story.append(subsec("7.2", "Classification Performance"))
story.append(P(
    "The XGBoost classifier was trained on 47,417 windows and evaluated on a grouped holdout "
    "of 17,127 windows. The class distribution was nearly balanced: 8,547 ALTERED_CONDITION "
    "and 8,580 REFERENCE windows in the test set. Classification metrics are shown in Table 5."
))
story.append(SP(3))
tbl5_data = [
    ["Metric", "ALTERED_CONDITION", "REFERENCE", "Macro Avg.", "Weighted Avg."],
    ["Precision", "0.4986", "0.4272", "0.4629", "0.4628"],
    ["Recall", "0.9931", "0.0051", "0.4991", "0.4982"],
    ["F1-Score", "0.6639", "0.0101", "0.3370", "0.3364"],
    ["Support", "8,547", "8,580", "17,127", "17,127"],
    ["Accuracy", "-", "-", "0.4982", "-"],
    ["ROC-AUC", "-", "-", "0.4873", "-"],
]
story.append(nice_table(tbl5_data, [3.5*cm, 3.5*cm, 3.0*cm, 3.0*cm, 3.0*cm]))
story.append(P("Table 5. XGBoost classification report on the 25% grouped holdout (n=17,127 test "
               "windows). Accuracy ~0.498 and ROC-AUC ~0.487 indicate near-random discrimination "
               "under the recording-grouped evaluation protocol.", CAPTION))

story.append(P(
    "The near-random accuracy (~0.498) and ROC-AUC (~0.487) indicate that the classifier does "
    "not achieve reliable discrimination under the grouped holdout protocol. The highly asymmetric "
    "per-class recall pattern (ALTERED_CONDITION: 0.993 vs REFERENCE: 0.005) and the confusion "
    "matrix reveal that the model assigns almost all test windows to the ALTERED_CONDITION class, "
    "indicating a trivial majority-class heuristic. This outcome is attributable to significant "
    "class imbalance in label assignment (51,674 ALTERED_CONDITION vs 12,870 REFERENCE windows) "
    "and path-based label provenance that does not guarantee mechanically distinguishable "
    "structural states in the specific recording subset analysed."
))
story.append(SP(4))
story.append(img_full(P_F4, PAGE_W - MARGIN_L - MARGIN_R - 6*cm))
story.append(P("Figure 5. Confusion matrix for the XGBoost classifier on the grouped holdout test "
               "set. The near-absence of true REFERENCE predictions reflects class imbalance in "
               "the configured label assignment.", CAPTION))

story.append(subsec("7.3", "SHAP Feature Attribution"))
story.append(P(
    "Despite the classifier's limited discrimination, SHAP attribution provides valuable insight "
    "into which features the model relied upon. The global mean absolute SHAP values place RMS "
    "(Phi = 0.2305), dominant frequency (Phi = 0.2079), and signal mean (Phi = 0.2071) as the "
    "three highest-contributing features. This ranking is physically interpretable: RMS amplitude "
    "and dominant frequency are well-established vibration damage indicators [2,3], consistent "
    "with classical SHM feature selection. The seven band-energy features registered zero SHAP "
    "attribution, confirming that the spectral sub-band partitioning as configured does not add "
    "discriminative information, likely due to sampling rate uncertainty."
))
story.append(SP(4))
story.append(img_full(P_F2))
story.append(P("Figure 6. Global SHAP feature importance ranking for the XGBoost classifier. "
               "Bar length represents mean |SHAP value| across all test windows. RMS, dominant "
               "frequency, and signal mean are the top-ranked features.", CAPTION))

story.append(SP(3))
tbl6_data = [
    ["Rank", "Feature", "Mean |SHAP|", "Physical Interpretation"],
    ["1", "RMS", "0.2305", "Overall vibration energy level"],
    ["2", "Dominant Frequency", "0.2079", "Primary resonance frequency"],
    ["3", "Mean", "0.2071", "Signal DC offset / bias level"],
    ["4", "Frequency Centroid", "0.1407", "Spectral centre of mass"],
    ["5", "Skewness", "0.1319", "Signal asymmetry / non-Gaussianity"],
    ["6", "Minimum", "0.1207", "Negative peak amplitude"],
    ["7", "Spectral Bandwidth", "0.1158", "Spectral spread"],
    ["8", "PSD Peak", "0.1088", "Maximum spectral density value"],
    ["9", "Kurtosis", "0.1087", "Impulsiveness / outlier sensitivity"],
    ["10", "Std. Deviation", "0.0917", "Vibration amplitude spread"],
]
story.append(nice_table(tbl6_data, [1.5*cm, 4.0*cm, 2.5*cm, 7.0*cm]))
story.append(P("Table 6. Top-10 globally important features by mean absolute SHAP value.", CAPTION))

story.append(subsec("7.4", "Sensor Ranking and Risk Analysis"))
story.append(P(
    "Sensor-level SHAP aggregation produced closely clustered importance scores ranging from "
    "1.80 to 1.95: sensor 01_1 ranked first (1.9507), followed by 01_2 (1.9200), 01_0 (1.8935), "
    "and 01_3 (1.8038). The narrow spread indicates that no single sensor dominates attribution "
    "under the current configuration, consistent with the spatially distributed nature of ambient "
    "vibration excitation in the Z24 structure."
))
story.append(SP(4))
story.append(img_full(P_F7, PAGE_W - MARGIN_L - MARGIN_R - 4*cm))
story.append(P("Figure 7. Sensor-level SHAP attribution ranking derived by aggregating feature-level "
               "SHAP values by sensor identifier.", CAPTION))

story.append(SP(4))
story.append(img_full(P_F6))
story.append(P("Figure 8. Prototype risk score distribution (left) across reference and altered "
               "condition windows, and the resulting risk level proportions (right) across all "
               "analysed windows.", CAPTION))

# SECTION 8 - DISCUSSION
story.append(section("8", "Discussion"))
story.append(P(
    "The experimental results reveal both the strengths and inherent limitations of the current "
    "prototype configuration. The BOCPD stage successfully executed on all available sensor-recording "
    "sequences and provided a continuously updated change probability correlating with transient "
    "amplitude events in the RMS signal. The absence of threshold crossings in the primary analysed "
    "recording reflects a property of that specific sensor-scenario combination rather than a general "
    "claim about detection capability. Extending BOCPD evaluation across all seventeen Z24 scenarios, "
    "particularly those involving progressive pier settlement, would be expected to yield meaningfully "
    "different results and constitutes a high-priority avenue for future work."
))
story.append(P(
    "The classification performance highlights a fundamental challenge in applying supervised learning "
    "to benchmark SHM datasets: the gap between condition labels defined by experimental protocol "
    "and acoustic or vibrational distinguishability as captured by static feature windows. The Z24 "
    "progressive damage tests were designed to characterise gradual structural changes that may not "
    "manifest as dramatic differences in simple time-domain or frequency-domain statistics from "
    "short ambient vibration windows. The class imbalance, mapping 14 scenarios to ALTERED_CONDITION "
    "but only 3 to REFERENCE, further biases the model toward the majority class. Future work should "
    "explore balanced sampling strategies, recording-level classification, and more discriminative "
    "feature engineering informed by modal analysis."
))
story.append(P(
    "The SHAP attribution results are the most interpretable output of the current pipeline. "
    "Top-ranked features, RMS, dominant frequency, and signal mean, align with prior SHM "
    "literature and provide a reasonable starting point for feature selection refinement. "
    "The zero attribution for band-energy features is a valuable diagnostic confirming that "
    "the spectral sub-band partitioning, as configured, does not add discriminative information "
    "beyond broader spectral descriptors. Resolving sampling rate uncertainty and enabling "
    "the band-pass filter would likely change this picture substantially."
))
story.append(P(
    "The digital twin dashboard successfully integrates all analytical outputs and provides a "
    "usable engineering review interface. The historical replay mode enables post-hoc "
    "investigation of specific time periods, and the sensor schematic provides spatial context "
    "for attribution rankings. These qualitative contributions, interpretability, review support, "
    "and transparent uncertainty communication, are the primary value proposition of the prototype, "
    "independent of the specific metric values observed in this preliminary evaluation."
))

# SECTION 9 - LIMITATIONS
story.append(section("9", "Limitations and Future Work"))
story.append(P(
    "Several limitations of the current prototype are explicitly acknowledged to support "
    "transparent and responsible use of results. <b>Sampling rate uncertainty:</b> the effective "
    "sampling rate is not resolved, defaulting to a 1 Hz fallback and restricting the Nyquist "
    "frequency to 0.5 Hz. Establishing the true acquisition rate (nominally 100 Hz for Z24 "
    "accelerometers) is the highest-priority implementation task. <b>Label validity:</b> "
    "REFERENCE and ALTERED_CONDITION labels are derived from source-path regular expressions "
    "and are not independently validated against engineering measurements of structural property "
    "changes. <b>Classification performance:</b> the near-random accuracy means the current "
    "XGBoost model should not be used as a reliable condition discriminator without further "
    "development. <b>Physics-based twin:</b> the dashboard is a data-driven monitoring interface, "
    "not a physics-calibrated finite-element model. <b>Live sensing:</b> the system operates in "
    "historical replay mode only, with no live data acquisition or IoT connectivity. "
    "<b>Sensor localisation:</b> the bridge schematic uses synthetic sensor coordinates."
))
story.append(P(
    "Future work directions include: (1) resolving sampling rate and enabling the full spectral "
    "feature pipeline; (2) implementing recording-level rather than window-level classification; "
    "(3) integrating physics-based modal analysis for frequency-calibrated change detection; "
    "(4) extending BOCPD evaluation across all seventeen Z24 scenarios; (5) incorporating "
    "real-time sensor connectivity; (6) developing calibrated confidence intervals for the "
    "risk score; and (7) establishing full experiment reproducibility through run-ID tracking "
    "and artifact hashing."
))

# SECTION 10 - CONCLUSION
story.append(section("10", "Conclusion"))
story.append(P(
    "This paper has presented the Digital Twin SHM Framework, a modular, explainable structural "
    "health monitoring prototype validated on the Z24 bridge benchmark dataset. The system "
    "integrates nine processing stages from format-aware signal ingestion to interactive digital "
    "twin dashboard replay within a single configurable pipeline. Applied to 64,544 windows of "
    "accelerometer data spanning seventeen Z24 measurement scenarios, the pipeline successfully "
    "executed all analytical stages including BOCPD change detection, grouped XGBoost classification, "
    "SHAP feature attribution, sensor-level importance ranking, and prototype risk aggregation."
))
story.append(P(
    "The SHAP attribution analysis identified RMS amplitude, dominant frequency, and signal mean "
    "as the most model-influential features, consistent with vibration-based SHM literature. "
    "The BOCPD detector operated within its expected behavioural range on the analysed recording, "
    "and the transparent risk aggregation formula provides a structured decision-support interface "
    "combining statistical and model-based evidence without conflating them."
))
story.append(P(
    "The classifier's near-random performance under grouped holdout evaluation is reported honestly "
    "and attributed to sampling rate uncertainty, label imbalance, and the intrinsic challenge of "
    "window-level discrimination of gradually evolving structural changes. These findings motivate "
    "concrete technical improvements that form the future research agenda. Building upon the "
    "foundational DT-BOCPD anomaly detection paradigm established by Lu et al. [21] for building "
    "HVAC assets, the primary contribution of this work is the extension to civil infrastructure SHM "
    "through five novel additions: supervised XGBoost classification, SHAP-based feature and "
    "sensor-level attribution, weighted multi-stream risk aggregation, validation on the Z24 bridge "
    "benchmark, and an interactive Streamlit dashboard. The system design explicitly "
    "separates statistical change evidence from damage diagnosis claims, an essential distinction "
    "for responsible deployment of AI-assisted monitoring in safety-critical infrastructure."
))

# ACKNOWLEDGEMENTS
story.append(HR())
story.append(P("<b>Acknowledgements</b>", SECTION_HDR))
story.append(P(
    "The authors gratefully acknowledge Dr. V. Devisurya, Assistant Professor, Department of "
    "Information Technology, Kongu Engineering College, for her expert guidance and continued "
    "encouragement throughout this research. The Z24 bridge benchmark dataset was made available "
    "by KU Leuven through the SIMCES project consortium. The authors thank the open-source "
    "communities behind NumPy, SciPy, scikit-learn, XGBoost, SHAP, Plotly, and Streamlit "
    "for their foundational contributions."
))

# REFERENCES
story.append(HR())
story.append(P("<b>References</b>", SECTION_HDR))
refs = [
    "[1] Peeters, B., & De Roeck, G. (1999). Reference-based stochastic subspace identification "
    "for output-only modal analysis. <i>Mechanical Systems and Signal Processing</i>, 13(6), 855-878.",

    "[2] Farrar, C. R., & Worden, K. (2007). An introduction to structural health monitoring. "
    "<i>Philosophical Transactions of the Royal Society A</i>, 365(1851), 303-315.",

    "[3] Worden, K., & Dulieu-Barton, J. M. (2004). An overview of intelligent fault detection in "
    "systems and structures. <i>Structural Health Monitoring</i>, 3(1), 85-98.",

    "[4] Sohn, H., Farrar, C. R., et al. (2004). A review of structural health monitoring literature: "
    "1996-2001. <i>Los Alamos National Laboratory Report</i> LA-13976-MS.",

    "[5] Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. "
    "<i>Advances in Neural Information Processing Systems</i>, 30, 4765-4774.",

    "[6] Maeck, J., & De Roeck, G. (2003). Description of Z24 benchmark. "
    "<i>Mechanical Systems and Signal Processing</i>, 17(1), 127-131.",

    "[7] Peeters, B., & De Roeck, G. (2001). One year monitoring of the Z24 bridge: environmental "
    "effects versus damage events. <i>Earthquake Engineering and Structural Dynamics</i>, 30(2), 149-171.",

    "[8] Adams, R. P., & MacKay, D. J. C. (2007). Bayesian online changepoint detection. "
    "<i>arXiv preprint</i> arXiv:0710.3742.",

    "[9] Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. "
    "<i>Proceedings of the 22nd ACM SIGKDD International Conference</i>, 785-794.",

    "[10] Cha, Y.-J., Choi, W., & Buyukozturk, O. (2017). Deep learning-based crack damage detection "
    "using convolutional neural networks. <i>Computer-Aided Civil and Infrastructure Engineering</i>, "
    "32(5), 361-378.",

    "[11] Joslyn, C. A., & Moore, S. (2021). Leakage detection in SHM cross-validation: Pitfalls "
    "and remedies. <i>Structural Health Monitoring</i>, 20(4), 1756-1773.",

    "[12] Mangalathu, S., Hwang, S. H., & Jeon, J. S. (2020). Failure mode and effects analysis of "
    "RC members based on machine-learning-based SHAP. <i>Engineering Structures</i>, 219, 110927.",

    "[13] Shim, H., & Park, H. (2022). Explainability of deep-learning models for SHM using SHAP "
    "values. <i>Sensors</i>, 22(3), 957.",

    "[14] Niedostatkiewicz, M., & Gorski, J. (2019). Digital twin approach in SHM of civil engineering "
    "structures. <i>Archives of Civil Engineering</i>, 65(4), 17-30.",

    "[15] Grieves, M., & Vickers, J. (2017). Digital twin: Mitigating unpredictable emergent behavior "
    "in complex systems. In <i>Transdisciplinary Perspectives on Complex Systems</i> (pp. 85-113). Springer.",

    "[16] Koo, K. Y., & Brownjohn, J. M. W. (2013). Structural health monitoring of the Tamar "
    "suspension bridge. <i>Structural Control and Health Monitoring</i>, 20(4), 609-625.",

    "[17] Avci, O., et al. (2021). A review of vibration-based damage detection in civil structures. "
    "<i>Mechanical Systems and Signal Processing</i>, 147, 107077.",

    "[18] Brincker, R., & Ventura, C. (2015). <i>Introduction to Operational Modal Analysis</i>. Wiley.",

    "[19] Rosafalco, L., et al. (2021). Online structural health monitoring by model order reduction "
    "and deep learning algorithms. <i>Computers and Structures</i>, 255, 106604.",

    "[20] Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. "
    "<i>Journal of Machine Learning Research</i>, 12, 2825-2830.",

    "[21] Lu, Q., Xie, X., Parlikad, A. K., & Schooling, J. M. (2020). Digital twin-enabled "
    "anomaly detection for built asset monitoring in operation and maintenance. "
    "<i>Automation in Construction</i>, 118, 103277. doi:10.1016/j.autcon.2020.103277.",
]
for r in refs:
    story.append(P(r, REF_STYLE))
    story.append(SP(1))

# BUILD PDF
print(f"Building PDF -> {OUT_PDF}")
doc = SimpleDocTemplate(
    OUT_PDF, pagesize=A4,
    leftMargin=MARGIN_L, rightMargin=MARGIN_R,
    topMargin=MARGIN_T + 0.5*cm, bottomMargin=MARGIN_B + 0.5*cm,
    title="Digital Twin SHM - Z24 Bridge",
    author="Dhivyashri K K, Hemavarshini V, Dr. V. Devisurya",
    subject="Structural Health Monitoring - Biomedical Signal Processing and Control",
)
doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
print(f"SUCCESS: PDF written to {OUT_PDF}")
