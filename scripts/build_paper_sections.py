"""Section 5, abstract, Sections 6-7 and document assembly for the BPASTS paper (numbers from result files)."""
from __future__ import annotations

import copy
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import docx
import numpy as np
import pandas as pd

from build_paper import DST, FIG, MODEL, SRC, ci, load, methods_text, num, row
from paper_refs import INTRO, REFERENCES, RELATED
from paper_docx import W, Templates, body_elements, find_index, replace_range, set_paragraph_text, set_rich_text, text_of

PENDING = "[PENDING – result file not yet produced]"
SCHEME = {"loso": "LOSO", "group_kfold": "GKF"}


def c2(r, m):
    return f"{r[m]:.2f} [{r[m + '_ci_low']:.2f}, {r[m + '_ci_high']:.2f}]"


def seq_row(d, method, param):
    q = d["seq"][(d["seq"].method == method) & (d["seq"].parameter == param)]
    return q.iloc[0]


# ---------------------------------------------------------------------------------------------- Section 5
def section5(d, T: Templates, doc) -> list:
    cfg, c, rec = d["cfg"], d["counts"], d["rec"]
    avt = rec[rec.test == "avt"]
    E = []
    P = lambda k, s, rich=False: E.append(T.para(k, s, rich))  # noqa: E731
    tabn = iter(range(4, 20))
    fign = iter(range(2, 20))

    def table(title, rows, widths):
        n = next(tabn)
        E.append(T.para("tablabel", f"Table {n}"))
        E.append(T.para("tabtitle", title))
        E.append(T.table(rows, widths))
        E.append(T.para("empty", ""))
        return n

    def figure(img, caption, n):
        E.append(T.figure(doc, img, 8.0))
        E.append(T.caption(f"Fig. {n}.", caption))
        return n

    E.append(T.para("h1", "5. EXPERIMENTAL RESULTS"))
    # 5.1 -------------------------------------------------------------------------------------------
    P("h2", "5.1. Experimental setup")
    env = None
    runs = sorted((Path(DST).parent / "results" / "runs").glob("*/manifest.json"))
    if runs:
        env = json.loads(runs[-1].read_text())["environment"]
    pk = env["packages"] if env else {}
    t_proc = [json.loads(m.read_text()) for m in runs if "_process_" in m.parent.name]
    wall = t_proc[-1]["wall_time_s"] if t_proc else None
    P("body", f"The pipeline is implemented in Python {env['python'].split()[0] if env else ''} with NumPy {pk.get('numpy','')}, "
              f"SciPy {pk.get('scipy','')}, scikit-learn {pk.get('sklearn','')}, XGBoost {pk.get('xgboost','')} and SHAP "
              f"{pk.get('shap','')}, and was run on an {env['cpu_count'] if env else ''}-core Windows PC. Every stage stores a run "
              "identifier, the configuration, package versions, seeds, wall time and SHA-256 hashes of its outputs"
              + (f"; loading, preprocessing and feature extraction of all 306 recordings took {wall / 60:.1f} min." if wall else "."))
    ev, sb = cfg["evaluation"], cfg["sequence_bocpd"]
    # 5.2 modal ---------------------------------------------------------------------------------------
    P("h2", "5.2. Natural frequencies")
    g = avt.groupby("scenario")[["f1", "f2", "f3", "f4", "zeta1"]].mean()
    s01 = g.loc["01"]
    idr = {k: avt[k].notna().mean() for k in ("f1", "f2", "f3", "f4")}
    dmin = {k: (g[k].idxmin(), 100 * (g[k].min() - s01[k]) / s01[k]) for k in ("f1", "f2", "f3")}
    P("body", f"The four baseline modes of scenario 01 lie at {', '.join(f'{b['frequency_hz']:.2f}' for b in d['base'])} Hz. "
              f"They were identified in {100 * idr['f1']:.1f}%, {100 * idr['f2']:.1f}%, {100 * idr['f3']:.1f}% and {100 * idr['f4']:.1f}% "
              f"of the 153 AVT recordings, with a mean ζ1 of {100 * avt['zeta1'].mean():.2f}% (Fig. 2). The largest decreases "
              f"relative to scenario 01 are {dmin['f1'][1]:.1f}% for f1 (scenario {dmin['f1'][0]}), {dmin['f2'][1]:.1f}% for f2 "
              f"(scenario {dmin['f2'][0]}) and {dmin['f3'][1]:.1f}% for f3 (scenario {dmin['f3'][0]}). The reference scenarios are not "
              f"identical: relative to scenario 01, f1 of scenario 02 differs by {100 * (g.loc['02','f1'] - s01['f1']) / s01['f1']:.2f}% and "
              f"f2 of scenario 08 by {100 * (g.loc['08','f2'] - s01['f2']) / s01['f2']:.2f}%, which is comparable to the shifts caused "
              f"by the smallest interventions (scenario 03: f1 {100 * (g.loc['03','f1'] - s01['f1']) / s01['f1']:.2f}%).")
    if d.get("ssi"):
        ss_ = d["ssi"]
        P("body", "Covariance-driven stochastic subspace identification [17] (SSI-COV, reference sensors, model orders 20–60 with a "
                  "stabilisation check) confirmed the FDD modes: a stable SSI pole within 5% was found for "
                  f"{100 * min(v['ssi_found_rate'] for v in ss_.values()):.0f}–{100 * max(v['ssi_found_rate'] for v in ss_.values()):.0f}% "
                  "of the identified modes, with median frequency differences of "
                  f"{min(v['median_abs_rel_diff_pct'] for v in ss_.values()):.2f}–{max(v['median_abs_rel_diff_pct'] for v in ss_.values()):.2f}% "
                  f"and a median ζ1 of {ss_['mode1']['median_zeta_ssi_pct']:.2f}% (EFDD {ss_['mode1']['median_zeta_efdd_pct']:.2f}%).")
    figure(FIG / "fig_natural_frequencies.png", "Natural frequencies f1–f4 of the 153 AVT recordings ordered by scenario and "
           "setup (squares: reference scenarios)", 2)

    # 5.3 change detection ---------------------------------------------------------------------------
    P("h2", "5.3. Change detection")
    w = d["within"]
    P("body", f"Within records, BOCPD processed {num(w['sequences'])} channel sequences ({num(w['windows'])} windows). "
              f"{w['detections']} windows crossed CP ≥ {w['threshold']} (maximum {w['max_cp']:.3f}) and {num(w['windows_cp_ge_05'])} "
              f"exceeded 0.5, because the state of the bridge is constant within a recording.")
    h, k0 = sb["primary_hazard_rate"], sb["primary_kappa0"]
    rows = [["Detector", "Det.", "TP", "Prec.", "Recall", "F1", "Delay"]]
    for k_ in sb["kappa0_values"]:
        r = seq_row(d, "BOCPD", f"hazard={h},kappa0={k_}")
        rows.append([f"BOCPD, κ_{{0}} = {k_}", str(int(r.detections)), str(int(r.true_positives)), f"{r.precision:.2f}",
                     f"{r.recall:.2f}", f"{r.f1:.2f}", f"{r.mean_delay:.2f}"])
    for hh in cfg["cusum"]["h"]:
        r = seq_row(d, "CUSUM", f"k={cfg['cusum']['k']},h={hh}")
        rows.append([f"CUSUM, h = {hh:g}", str(int(r.detections)), str(int(r.true_positives)), f"{r.precision:.2f}",
                     f"{r.recall:.2f}", f"{r.f1:.2f}", f"{r.mean_delay:.2f}"])
    pb = seq_row(d, "BOCPD", f"hazard={h},kappa0={k0}")
    pc = seq_row(d, "CUSUM", f"k={cfg['cusum']['k']},h={cfg['cusum']['primary_h']}")
    bb = d["seq"][(d["seq"].method == "BOCPD") & d["seq"].parameter.str.endswith(f"kappa0={k0}")]
    perb = d["perb"]
    missed_both = perb[~perb.bocpd_detected & ~perb.cusum_detected]
    maxd = missed_both[[c_ for c_ in perb.columns if c_.startswith("delta_f") and c_ != "delta_f4_pct"]].abs().max(axis=1)
    det_both = perb[perb.bocpd_detected & perb.cusum_detected]
    P("body", f"Across scenarios, the 153 AVT recordings contain 16 true boundaries. With h = {h} (Table 4), BOCPD with the default "
              f"prior κ0 = 1 detected only {int(seq_row(d, 'BOCPD', f'hazard={h},kappa0=1.0').true_positives)} boundaries, because the "
              f"prior predictive of a new segment is centred within about one baseline standard deviation of scenario 01 whereas later "
              f"states lie tens of standard deviations away. With the vague prior κ0 = {k0}, BOCPD reached a precision of "
              f"{pb.precision:.3f}, a recall of {pb.recall:.3f} (F1 {pb.f1:.3f}) and a mean delay of {pb.mean_delay:.2f} records. CUSUM "
              f"(h = {cfg['cusum']['primary_h']:g}) reached a precision of {pc.precision:.3f} and a recall of {pc.recall:.3f} "
              f"(F1 {pc.f1:.2f}); for every h tested CUSUM produced more false alarms than BOCPD. Both detectors missed "
              f"{', '.join(f'{int(a):02d}→{int(b):02d}' for a, b in zip(missed_both.from_scenario, missed_both.to_scenario))}, "
              f"for which |Δf1–f3| ≤ {maxd.max():.2f}% (Fig. 3).")
    table("Detection of the 16 scenario boundaries (AVT sequence, tolerance 4 records; delay in records)", rows,
          [1403, 480, 420, 600, 600, 600, 600])
    figure(FIG / "fig_sequence_detection.png", f"Change detection along the AVT scenario sequence: standardised f1–f4, BOCPD MAP "
           f"run length (h = {h}, κ0 = {k0}) and CUSUM statistic (h = {cfg['cusum']['primary_h']:g}); ▼ detections", 3)

    # 5.4 classification ------------------------------------------------------------------------------
    P("h2", "5.4. Condition classification")
    m = d["main"]
    if m is None:
        P("body", PENDING)
    else:
        cd = d["clfdata"]
        mw = m[m.level == "window"]
        mr = m[m.level == "recording"]
        order = ["xgboost", "random_forest", "svm", "logistic_regression", "freq_threshold"]
        rows = [["Model", "CV", "BA", "Macro-F1", "ROC-AUC", "MCC"]]
        phys = d["phys"]
        extra = [] if phys is None else [("xgboost_monotone", "modal")]
        for sch in ("loso", "group_kfold"):
            for mod in order:
                r = row(mw, model=mod, scheme=sch)
                rows.append([MODEL[mod], SCHEME[sch], c2(r, "balanced_accuracy"), c2(r, "macro_f1"), c2(r, "roc_auc"), c2(r, "mcc")])
            if phys is not None:
                for mod, fs in extra:
                    q = phys[(phys.level == "window") & (phys.model == mod) & (phys.features == fs) & (phys.scheme == sch)]
                    if len(q):
                        r = q.iloc[0]
                        rows.append([f"{MODEL[mod]} (f, ζ)", SCHEME[sch], c2(r, "balanced_accuracy"), c2(r, "macro_f1"), c2(r, "roc_auc"), c2(r, "mcc")])
        lo = mw[mw.scheme == "loso"].set_index("model")
        gk = mw[mw.scheme == "group_kfold"].set_index("model")
        best_lo = lo["roc_auc"].idxmax()
        best_gk = gk["roc_auc"].idxmax()
        P("body", f"The classification data contain {num(cd['windows'])} non-overlapping AVT windows from {cd['recordings']} recordings "
                  f"({cd['recordings_by_label']['REFERENCE']} REFERENCE, {cd['recordings_by_label']['ALTERED_CONDITION']} ALTERED). "
                  f"Table 5 compares the models. Under grouped 5-fold CV, where other setups of the held-out scenario remain "
                  f"in training, the best ROC-AUC was {gk.loc[best_gk, 'roc_auc']:.3f} ({MODEL[best_gk]}; BA "
                  f"{gk.loc[best_gk, 'balanced_accuracy']:.3f}, MCC {gk.loc[best_gk, 'mcc']:.3f}). Under LOSO, where the whole scenario "
                  f"is unseen, the natural-frequency threshold reached a ROC-AUC of {lo.loc['freq_threshold', 'roc_auc']:.3f} (BA "
                  f"{lo.loc['freq_threshold', 'balanced_accuracy']:.3f}), the best ML model {lo.drop('freq_threshold')['roc_auc'].max():.3f} "
                  f"({MODEL[lo.drop('freq_threshold')['roc_auc'].idxmax()]}) and XGBoost {lo.loc['xgboost', 'roc_auc']:.3f} "
                  f"(BA {lo.loc['xgboost', 'balanced_accuracy']:.3f}, MCC {lo.loc['xgboost', 'mcc']:.3f}).")
        ps = d["main_ps"]
        pl = ps[(ps.scheme == "loso") & (ps.model == "xgboost")].set_index("scenario")
        refacc = pl[pl.label == "REFERENCE"]["recording_accuracy"]
        altacc = pl[pl.label == "ALTERED"]["recording_accuracy"]
        P("body", f"For XGBoost under LOSO, the recording accuracy on the held-out reference "
                  f"scenarios was {', '.join(f'{v:.2f} (S{k})' for k, v in refacc.items())}, while the altered scenarios reached "
                  f"{altacc.min():.2f}–{altacc.max():.2f} (median {altacc.median():.2f}): a classifier trained on two reference states "
                  "did not recognise the third, whose frequencies differ as much as those of the smallest interventions (Section 5.2).")
        table("Classification results with 95% bootstrap CIs over recordings (window level; mean over seeds)", rows,
              [920, 470, 830, 830, 830, 823])

    # 5.5 ablation --------------------------------------------------------------------------------
    P("h2", "5.5. Ablation")
    a = d["abl"]
    if a is None:
        P("body", PENDING)
    else:
        aw = a[a.level == "window"]
        rows = [["Features", "Norm.", "ROC-AUC", "BA", "MCC"]]
        for fs, nm in (("time", "none"), ("spectral", "none"), ("modal", "none"), ("time+spectral", "none"),
                       ("time+spectral", "zscore"), ("all", "none"), ("all", "zscore")):
            rl_ = row(aw, features=fs, normalisation=nm, scheme="loso")
            rows.append([fs, "z" if nm == "zscore" else "–", c2(rl_, "roc_auc"), c2(rl_, "balanced_accuracy"), c2(rl_, "mcc")])
        P("body", f"Table 6 isolates the feature groups and the normalisation (XGBoost, {int(aw.n_seeds.max())} seed{'s' if aw.n_seeds.max() > 1 else ''}; the seed s.d. of the main XGBoost AUC is below the CI width). "
                  + ablation_sentence(aw))
        table("Ablation of feature groups and per-record normalisation (XGBoost, LOSO, 95% CIs)", rows, [1100, 450, 1050, 1050, 1053])
        ph = d["phys"]
        if ph is not None:
            pw = ph[ph.level == "window"]
            q = pw[(pw.model == "xgboost_monotone") & (pw.features == "modal") & (pw.scheme == "loso")].iloc[0]
            q2 = pw[(pw.model == "freq_decrease_threshold") & (pw.scheme == "loso")].iloc[0]
            P("body", "Two physically motivated variants, added after the first LOSO results, are XGBoost with P(ALTERED) monotone "
                      "non-increasing in each natural frequency and a threshold on the largest relative frequency decrease; under "
                      f"LOSO they reached ROC-AUC {c2(q, 'roc_auc')} (BA "
                      f"{q['balanced_accuracy']:.3f}) and {c2(q2, 'roc_auc')} (BA {q2['balanced_accuracy']:.3f}).")

    nv = d.get("novelty")
    if nv is not None:
        a4 = nv["f1-f4"]
        c4 = nv.get("S01+S02:f1-f4")
        P("body", "A label-free novelty score, the Mahalanobis distance of a recording's natural frequencies from a baseline, "
                  f"ranked altered above reference recordings with a ROC-AUC of {a4['roc_auc_altered_vs_reference']:.3f} "
                  f"[{a4['roc_auc_ci'][0]:.3f}, {a4['roc_auc_ci'][1]:.3f}] with the {a4['baseline_records']} scenario-01 records as "
                  f"baseline, but a χ² (99%) threshold flagged {100 * a4['alarm_rate_reference']:.0f}% of the reference recordings."
                  + (" With both pre-intervention measurements as baseline (scenarios 01 and 02) and an empirical threshold, the "
                     "largest leave-one-out baseline distance (chosen after the χ² result), it flagged "
                     f"{100 * c4['alarm_rate_reference_empirical']:.0f}% of the held-out reference and "
                     f"{100 * c4['alarm_rate_altered_empirical']:.0f}% of the altered recordings (ROC-AUC "
                     f"{c4['roc_auc_altered_vs_reference']:.3f}, BA {c4['balanced_accuracy_empirical']:.3f} "
                     f"[{c4['balanced_accuracy_empirical_ci'][0]:.3f}, {c4['balanced_accuracy_empirical_ci'][1]:.3f}])."
                     if c4 is not None else ""))

    # 5.6 SHAP ---------------------------------------------------------------------------------------
    P("h2", "5.6. SHAP attribution and sensor ranking")
    if d["shap"] is None:
        P("body", PENDING)
    else:
        s, gg, ss = d["shap"], d["shap_g"], d["shap_s"]
        grp = d["shap_grp"].set_index("group")["share"]
        top = gg.head(5)
        sg = d["shap_sg"].sort_values("channel_attribution", ascending=False)
        sd = d["shap_dir"].sort_values("channel_attribution", ascending=False)
        P("body", f"TreeSHAP values were computed for all {num(s['explained_windows'])} held-out LOSO windows "
                  f"({s['explained_recordings']} recordings, seed{'s' if len(s['seeds']) > 1 else ''} {', '.join(map(str, s['seeds']))}); XGBoost's internal TreeSHAP "
                  + ("agreed exactly with the SHAP library. " if s['treeshap_crosscheck']['max_abs_difference'] == 0 else
                     f"agreed with the SHAP library to within {s['treeshap_crosscheck']['max_abs_difference']:.1e}. ")
                  + f"The modal, spectral and "
                  f"time-domain groups account for {100 * grp.get('modal', 0):.1f}%, {100 * grp.get('spectral', 0):.1f}% and "
                  f"{100 * grp.get('time', 0):.1f}% of the total mean |SHAP|; the five most important features are "
                  f"{', '.join(f'{r.feature} ({r.mean_abs_shap:.3f})' for r in top.itertuples())} (Fig. 4). All {s['sensors_ranked']} "
                  f"sensors were ranked (highest: {', '.join(ss.head(3).sensor)}).")
        big = sg[sg.sensors >= 5]
        spread = 100 * (big.channel_attribution.max() - big.channel_attribution.min()) / big.channel_attribution.mean()
        P("body", f"The attributions of the sensor groups with at least five sensors differ by only {spread:.1f}% of their mean, "
                  "so the ranking documents what the model uses, a record-level modal signature, and does not localise the "
                  "interventions."
                  + (f" Adding channel-level mode-shape changes relative to scenario 01 (|Δφ1–4|, scenario 01 used as baseline "
                     f"and excluded) left the LOSO ROC-AUC unchanged ({d['mshape']['classification'][0]['roc_auc']:.3f} vs "
                     f"{d['mshape']['classification'][1]['roc_auc']:.3f} on scenarios 02–17) and received "
                     f"{100 * d['mshape']['shap_share_mode_shape_features']:.2f}% of the attribution." if d.get("mshape") else ""))
        figure(FIG / "fig_shap.png", "Mean |SHAP| of the twelve most important features (left, log scale) and mean channel-level "
               "attribution by sensor group and direction (right), complete held-out LOSO sets", 4)

    # 5.7 risk + EMS -----------------------------------------------------------------------------------
    P("h2", "5.7. Digital-twin states and long-term monitoring")
    if d["risk"] is not None:
        rk, rl_ = d["risk"], d["risk_lab"]
        gl = rl_[rl_.variant == "gated"].set_index("label")
        ul = rl_[rl_.variant == "ungated"].set_index("label")
        inc = rk["classifier_term_included"]
        P("body", f"The LOSO ROC-AUC of XGBoost ({rk['classifier_auc']['loso']:.3f}) is "
                  f"{'above' if inc else 'below'} the required {rk['min_auc']}, so the classifier term is "
                  f"{'included in' if inc else 'excluded from'} the risk score (its grouped 5-fold ROC-AUC, "
                  f"{rk['classifier_auc']['group_kfold']:.3f}, does not represent a new state of the bridge). With the gated score, "
                  f"{int(gl.loc['REFERENCE', 'flagged'])} of {int(gl.loc['REFERENCE', 'recordings'])} reference recordings and "
                  f"{int(gl.loc['ALTERED_CONDITION', 'flagged'])} of {int(gl.loc['ALTERED_CONDITION', 'recordings'])} altered recordings "
                  f"were not in the NORMAL state (mean risk {gl.loc['REFERENCE', 'mean_risk']:.3f} vs "
                  f"{gl.loc['ALTERED_CONDITION', 'mean_risk']:.3f}).")
        pb_ = d["perb"]
        into_ref = pb_[pb_.to_scenario.astype(int).isin([2, 8])]
        det_ref = [f"{int(r.from_scenario):02d}→{int(r.to_scenario):02d}" for r in into_ref.itertuples() if bool(r.bocpd_detected)]
        missed_alt = [f"{int(r.to_scenario):02d}" for r in pb_.itertuples() if not bool(r.bocpd_detected) and int(r.to_scenario) not in (2, 8)]
        if gl.loc["REFERENCE", "mean_risk"] > gl.loc["ALTERED_CONDITION", "mean_risk"]:
          P("body", "The higher mean risk of the reference recordings follows from B, the probability of a change within the last "
                  "four records: it is 1 by construction for the first records of the sequence (scenario 01), "
                  + (f"the boundaries into the reference scenarios ({', '.join(det_ref)}) were detected, " if det_ref else "")
                  + f"and scenarios whose entry boundary BOCPD missed ({', '.join(missed_alt)}) score low; the score marks recent "
                  "changes of state, not their severity.")
    if d["ems"] is not None:
        e, ed = d["ems"], d["ems_det"].set_index(["compensation", "method"])
        tm = e["temperature_model"]["f1"]
        rb, rc = ed.loc[("regression", "BOCPD")], ed.loc[("regression", "CUSUM")]
        P("body", f"In the EMS data of the one-year campaign [15] ({num(e['counts']['usable'])} usable hourly records, {e['first_timestamp'][:10]} to "
                  f"{e['last_timestamp'][:10]}), the four modes were tracked in "
                  f"{min(e['counts']['identification_rate'].values()) * 100:.1f}–{max(e['counts']['identification_rate'].values()) * 100:.1f}% "
                  f"of the records. A piecewise-linear temperature model fitted until {e['train_end']} explains R² = {tm['train_r2']:.3f} "
                  f"of the f1 variance (slope below 0 °C: {tm['coef_intercept_slope_hinge'][1] + tm['coef_intercept_slope_hinge'][2]:.3f} "
                  f"Hz/°C). On the daily compensated frequencies, BOCPD raised {int(rb.alarms_in_training_period)} alarms in the "
                  f"training period, {int(rb.alarms_validation_jul01_aug04)} between 1 July and 4 August 1998, and its first alarm "
                  f"after the start of the tests (5 August) on {rb.first_alarm_on_or_after_aug05} (delay "
                  f"{int(rb.delay_days_from_aug05)} d); CUSUM raised {int(rc.alarms_validation_jul01_aug04)} validation alarms and "
                  f"first alarmed on {rc.first_alarm_on_or_after_aug05} (Fig. 5).")
        figure(FIG / "fig_ems.png", "EMS monitoring year: hourly f1, median temperature and daily temperature-compensated f1; "
               "dashed line: end of training period, shaded: onset window of the progressive damage tests", 5)
    if d["cost"] is not None:
        xc = d["cost"].set_index("model").loc["xgboost"]
        P("body", f"The final XGBoost model occupies {xc.serialized_size_bytes / 1024:.0f} kB, was trained on {num(xc.train_rows)} "
                  f"windows in {xc.fit_seconds:.1f} s and scores 1000 windows in {xc.inference_ms_per_1000_windows:.1f} ms.")
    return E


def ablation_sentence(aw) -> str:
    g = lambda fs, nm="none", sch="loso": row(aw, features=fs, normalisation=nm, scheme=sch)  # noqa: E731
    return (f"Under LOSO, time-domain features alone gave ROC-AUC {g('time')['roc_auc']:.3f}, spectral features "
            f"{g('spectral')['roc_auc']:.3f}, modal features {g('modal')['roc_auc']:.3f} and all features {g('all')['roc_auc']:.3f}; "
            f"per-record z-score normalisation changed the time+spectral result from {g('time+spectral')['roc_auc']:.3f} to "
            f"{g('time+spectral', 'zscore')['roc_auc']:.3f}, and the all-feature result from {g('all')['roc_auc']:.3f} to "
            f"{g('all', 'zscore')['roc_auc']:.3f}.")


# ---------------------------------------------------------------------------------------------- assembly
def build(draft: bool = False):
    d = load()
    doc = docx.Document(str(SRC))
    idx = lambda p, s=0: find_index(doc, p, s)  # noqa: E731
    i5 = idx("5. EXPERIMENTAL RESULTS")
    T = Templates(doc, {"h1": i5, "h2": idx("5.1. Experimental setup"), "body": i5 + 2, "tablabel": idx("Table 4"),
                        "tabtitle": idx("Table 4") + 1, "table": idx("Table 4") + 2, "empty": idx("Table 4") + 3,
                        "figure": idx("Fig. 2.") - 1, "figcap": idx("Fig. 2.")})
    # Section 5
    i6 = idx("6. DISCUSSION")
    replace_range(doc, i5, i6, section5(d, T, doc))
    # Methods (3.2 - 4.6)
    mt = methods_text(d)
    els = body_elements(doc)
    set_paragraph_text(els[idx("The pipeline processed one file per scenario")], mt["3.2"])
    set_paragraph_text(els[idx("Labels are assigned by regular-expression")], mt["3.3"])
    set_paragraph_text(els[idx("Each channel is validated")], mt["4.1"])
    set_paragraph_text(els[idx("4.2. Feature extraction")], "4.2. Window and modal features")
    set_paragraph_text(els[idx("Each window is described by 24")], mt["4.2"])
    set_paragraph_text(els[idx("4.3. Bayesian online")], "4.3. Change detection")
    set_paragraph_text(els[idx("BOCPD [1] is applied to the RMS")], mt["4.3a"])
    set_paragraph_text(els[idx("where πt(0) is evaluated")], mt["4.3b"])
    eq3 = els[idx("CPt = γt(0)")]
    els[idx("where πt(0) uses")].addprevious(eq3)
    hz = els[idx("The hazard is h = 0.01")]
    hz.getparent().remove(hz)
    els = body_elements(doc)
    set_paragraph_text(els[idx("4.4. XGBoost condition classifier")], "4.4. Classification and evaluation")
    set_paragraph_text(els[idx("The classifier is an XGBoost")], mt["4.4"])
    set_paragraph_text(els[idx("SHAP [2] decomposes")], mt["4.5"])
    set_paragraph_text(els[idx("where Φj is the global importance")], mt["4.5b"])
    set_paragraph_text(els[idx("with wB = 0.3")], mt["4.6"])
    # Table 2 (data)
    c = d["counts"]
    t2 = idx("Table 2")
    new_t2 = T.table([["Item", "Value"],
                      ["Recordings (scenarios × tests × setups)", f"{c['recordings']} (17 × 2 × 9)"],
                      ["Unique recording identifiers", str(c["unique_recording_ids"])],
                      [".mat / .aaa / zipped .mat files", f"{c['catalogue']['mat_files']} / {c['catalogue']['aaa_files']} / {c['catalogue']['zip_mat_members']}"],
                      ["Duplicates verified identical", f"{int(c['provenance_totals']['aaa_duplicate_identical'])} + {int(c['provenance_totals']['zip_duplicate_identical'])}"],
                      ["Sampling rate (all .aaa headers)", "100 Hz"],
                      ["Unique channels (AVT / FVT)", f"{num(c['unique_channels'])} ({num(c['channels_by_test']['avt'])} / {num(c['channels_by_test']['fvt'])})"],
                      ["Windows, 10 s, 50% overlap", num(c["windows_total"])],
                      ["REFERENCE / ALTERED windows", f"{num(c['windows_by_label']['REFERENCE'])} / {num(c['windows_by_label']['ALTERED_CONDITION'])}"]],
                     [2603, 2100])
    els = body_elements(doc)
    old = els[t2 + 2]
    old.addprevious(new_t2)
    old.getparent().remove(old)
    # Fig. 1
    els = body_elements(doc)
    i_f1 = idx("The implemented workflow is shown in Fig. 1")
    blip = els[i_f1].find(".//{http://schemas.openxmlformats.org/drawingml/2006/main}blip")
    if blip is not None and (FIG / "fig_pipeline.png").exists():
        rid, _ = doc.part.get_or_add_image(str(FIG / "fig_pipeline.png"))
        blip.set("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed", rid)
    # Introduction (citations mapped to the new reference list), related work, Section 3.1 citations, references
    els = body_elements(doc)
    for prefix, text in INTRO.items():
        set_paragraph_text(els[idx(prefix)], text)
    for key, heading in (("2.1", "2.1. Vibration-based SHM"), ("2.2", "2.2. Digital twins"), ("2.3", "2.3. Change detection")):
        els = body_elements(doc)
        first = els[idx(heading) + 1]
        set_paragraph_text(first, RELATED[key][0])
        prev = first
        for extra in RELATED[key][1:]:
            new_p = copy.deepcopy(first)
            set_paragraph_text(new_p, extra)
            prev.addnext(new_p)
            prev = new_p
    els = body_elements(doc)
    for prefix in ("The Z24 bridge was a post-tensioned", "Progressive damage test scenarios"):
        el = els[idx(prefix)]
        set_paragraph_text(el, text_of(el).replace("[20]", "[14]").replace("[21]", "[15]"))
    # Abstract, Sections 6-7
    els = body_elements(doc)
    set_paragraph_text(els[idx("Bridges require continuous")], abstract(d))
    i61 = idx("6. DISCUSSION")
    iref = idx("REFERENCES")
    replace_range(doc, i61, iref, discussion(d, T))
    replace_references(doc, renumber_citations(doc))
    apply_bpasts_style(doc)
    out = DST if not draft else DST.with_name(DST.stem + "_DRAFT.docx")
    doc.save(str(out))
    words = sum(len(text_of(e).split()) for e in body_elements(doc))
    print("saved", out, "words", words)
    return out


BPASTS_BLUE = "1B4F8B"  # heading colour of the BPASTS layout


def _style_runs(par, color=None, bold=None, font=None):
    for r in par.iter(W + "r"):
        rpr = r.find(W + "rPr")
        if rpr is None:
            rpr = r.makeelement(W + "rPr", {})
            r.insert(0, rpr)
        if font:
            f = rpr.find(W + "rFonts")
            if f is None:
                f = rpr.makeelement(W + "rFonts", {})
                rpr.insert(0, f)
            for att in ("ascii", "hAnsi", "cs"):
                f.set(W + att, font)
        if bold:
            if rpr.find(W + "b") is None:
                rpr.append(rpr.makeelement(W + "b", {}))
        if color:
            c = rpr.find(W + "color")
            if c is None:
                c = rpr.makeelement(W + "color", {})
                rpr.append(c)
            c.set(W + "val", color)


def apply_bpasts_style(doc):
    """Typographic conventions of the BPASTS layout: blue sans-serif title and section headings, bold sans-serif
    subsection headings, inline 'Abstract.' and blue 'Keywords:' labels, bold author line, blue caption labels."""
    import re
    els = body_elements(doc)
    paras = [e for e in els if e.tag == W + "p"]
    _style_runs(paras[0], BPASTS_BLUE, True, "Arial")
    for e in paras:
        t = text_of(e).strip()
        if re.match(r"^\d+\. [A-Z][A-Z ,&\-]+$", t) or t == "REFERENCES":
            _style_runs(e, BPASTS_BLUE, True, "Arial")
        elif re.match(r"^\d+\.\d+\. \S", t) and len(t) < 90:
            _style_runs(e, None, True, "Arial")
        elif re.match(r"^(Fig\. \d+\.|Table \d+)$", t) or re.match(r"^Fig\. \d+\.", t):
            first = next((r for r in e.iter(W + "r") if r.find(W + "t") is not None), None)
            if first is not None and re.match(r"^(Fig\. \d+\.|Table \d+)", "".join(first.itertext())):
                wrapper = e.makeelement(W + "p", {})
                wrapper.append(copy.deepcopy(first))
                _style_runs(wrapper, BPASTS_BLUE, True, None)
                first.getparent().replace(first, wrapper[0])
    au = next(e for e in paras if text_of(e).strip().startswith("Dr. V. Devisurya"))
    _style_runs(au, None, True, None)
    kw = next(e for e in paras if text_of(e).strip().startswith("Keywords:"))
    first = next(r for r in kw.iter(W + "r") if r.find(W + "t") is not None)
    wrapper = kw.makeelement(W + "p", {})
    wrapper.append(copy.deepcopy(first))
    _style_runs(wrapper, BPASTS_BLUE, True, "Arial")
    first.getparent().replace(first, wrapper[0])
    head = next(e for e in paras if text_of(e).strip() == "Abstract")
    body = head.getnext()
    while body is not None and not text_of(body).strip():
        body = body.getnext()
    runs = [r for r in body.iter(W + "r") if r.find(W + "t") is not None]
    lab = copy.deepcopy(runs[0])
    for t in lab.findall(W + "t"):
        lab.remove(t)
    t = lab.makeelement(W + "t", {})
    t.text = "Abstract. "
    t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    lab.append(t)
    wrapper = body.makeelement(W + "p", {})
    wrapper.append(lab)
    _style_runs(wrapper, BPASTS_BLUE, True, "Arial")
    runs[0].addprevious(wrapper[0])
    head.getparent().remove(head)


CITE_GROUP = re.compile(r"\[\d+\](?:\s*[,–]\s*\[\d+\])*")


def _group_numbers(group: str) -> list[int]:
    nums, prev = [], None
    for sep, n in re.findall(r"(^|,|–)\s*\[(\d+)\]", group):
        n = int(n)
        if sep == "–" and prev is not None:
            nums.extend(range(prev + 1, n + 1))
        else:
            nums.append(n)
        prev = n
    return nums


def _format_group(nums: list[int]) -> str:
    nums = sorted(set(nums))
    parts, i = [], 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        if j - i >= 2:
            parts.append(f"[{nums[i]}]–[{nums[j]}]")
        else:
            parts.extend(f"[{n}]" for n in nums[i:j + 1])
        i = j + 1
    return ", ".join(parts)


def renumber_citations(doc) -> list[int]:
    """Renumber citations in order of first appearance; returns the old numbers in their new order."""
    els = body_elements(doc)
    i0 = find_index(doc, "REFERENCES")
    texts = [t for e in els[:i0] for t in e.iter(W + "t") if t.text]
    order = []
    for t in texts:
        for g in CITE_GROUP.findall(t.text):
            for n in _group_numbers(g):
                if n not in order:
                    order.append(n)
    order += [n for n in range(1, len(REFERENCES) + 1) if n not in order]
    new_of = {old: k for k, old in enumerate(order, start=1)}
    for t in texts:
        t.text = CITE_GROUP.sub(lambda m: _format_group([new_of[n] for n in _group_numbers(m.group(0))]), t.text)
    return order


def replace_references(doc, order: list[int] | None = None):
    """Replace the reference list of the template by REFERENCES (in ``order``, venues in italics)."""
    order = order or list(range(1, len(REFERENCES) + 1))
    els = body_elements(doc)
    i0 = find_index(doc, "REFERENCES")
    old = [e for e in els[i0 + 1:] if e.tag == W + "p"]
    tmpl = copy.deepcopy(old[0])
    runs = [r for r in tmpl.iter(W + "r")]
    num_run, tab_run, text_run = runs[0], runs[1], copy.deepcopy(runs[2])
    for t in tab_run.findall(W + "t"):  # the template run holds the tab and the first author text
        tab_run.remove(t)
    for r in runs[2:]:
        r.getparent().remove(r)
    for e in old:
        e.getparent().remove(e)
    anchor = body_elements(doc)[i0]
    for k, (pre, venue, post) in enumerate([REFERENCES[o - 1] for o in order], start=1):
        para = copy.deepcopy(tmpl)
        rr = [r for r in para.iter(W + "r")]
        rr[0].find(W + "t").text = f"[{k}]"
        for text, italic in ((pre, False), (venue, True), (post, False)):
            r = copy.deepcopy(text_run)
            rpr = r.find(W + "rPr")
            if rpr is None:
                rpr = r.makeelement(W + "rPr", {})
                r.insert(0, rpr)
            for tag in ("i", "iCs"):
                for x in rpr.findall(W + tag):
                    rpr.remove(x)
            if italic:
                rpr.append(rpr.makeelement(W + "i", {}))
            for t in r.findall(W + "t"):
                r.remove(t)
            t = r.makeelement(W + "t", {})
            t.text = text
            t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
            r.append(t)
            para.append(r)
        anchor.addnext(para)
        anchor = para


def abstract(d) -> str:
    return "[ABSTRACT – written after all results are available]" if d["main"] is None else abstract_text(d)


def abstract_text(d) -> str:
    """Structure of the original abstract ("Final Paper 1"), with the exact results of the pipeline."""
    m = d["main"]
    mw = m[m.level == "window"]
    lo = mw[mw.scheme == "loso"].set_index("model")
    gk = mw[mw.scheme == "group_kfold"].set_index("model")
    pb = seq_row(d, "BOCPD", f"hazard={d['cfg']['sequence_bocpd']['primary_hazard_rate']},kappa0={d['cfg']['sequence_bocpd']['primary_kappa0']}")
    pc = seq_row(d, "CUSUM", f"k={d['cfg']['cusum']['k']},h={d['cfg']['cusum']['primary_h']}")
    c = d["counts"]
    xg, xl = gk.loc["xgboost"], lo.loc["xgboost"]
    txt = ("Bridges require continuous and interpretable condition monitoring, yet change detection, condition classification "
           "and model explanation are rarely combined and evaluated together on a common benchmark. This paper presents a "
           "digital-twin monitoring prototype that integrates Bayesian online change point detection (BOCPD), an XGBoost "
           "condition classifier, SHAP attribution and a validation-gated risk score in one configurable pipeline, and reports "
           f"its exact behaviour on the Z24 bridge benchmark. All {c['recordings']} recordings of the progressive damage tests "
           f"(17 scenarios, {num(c['unique_channels'])} channels at 100 Hz) were divided into {num(c['windows_total'])} windows "
           "described by time-domain, spectral and modal features, the latter comprising four natural frequencies identified by "
           "frequency domain decomposition. Applied to the scenario sequence of natural frequencies, BOCPD detected the state "
           f"changes with a precision of {pb.precision:.3f} and a recall of {pb.recall:.3f} (CUSUM: {pc.precision:.3f} and "
           f"{pc.recall:.3f}). The classifier reached a ROC-AUC of {xg.roc_auc:.3f} (95% CI {xg.roc_auc_ci_low:.3f}–"
           f"{xg.roc_auc_ci_high:.3f}) under recording-grouped cross-validation but {xl.roc_auc:.3f} "
           f"({xl.roc_auc_ci_low:.3f}–{xl.roc_auc_ci_high:.3f}) when whole scenarios were held out, below a natural-frequency "
           f"shift threshold ({lo.loc['freq_threshold', 'roc_auc']:.3f})")
    if d["risk"] is not None:
        txt += ("; the validation gate therefore " + ("kept it in" if d["risk"]["classifier_term_included"] else "excluded it from")
                + " the risk score")
    txt += "."
    if d["shap_g"] is not None and d["shap_grp"] is not None:
        top = ", ".join(d["shap_g"].head(3).feature)
        txt += (f" SHAP analysis of all held-out windows attributed {100 * d['shap_grp'].set_index('group')['share'].get('modal', 0):.1f}% "
                f"of the model output to the modal features, led by {top}.")
    nv = (d["novelty"] or {}).get("S01+S02:f1-f4")
    if nv is not None:
        txt += (" A label-free novelty score calibrated on the two pre-intervention measurements reached a balanced accuracy of "
                f"{nv['balanced_accuracy_empirical']:.3f}")
    if d["ems"] is not None:
        rb = d["ems_det"].set_index(["compensation", "method"]).loc[("regression", "BOCPD")]
        dd = int(rb.delay_days_from_aug05)
        txt += (", and on the one-year monitoring data temperature-compensated BOCPD raised its first alarm "
                f"{'one day' if dd == 1 else f'{dd} days'} after the start of the damage tests")
    txt += (". The prototype runs end-to-end and exposes every intermediate evidence stream to the engineer, while the results "
            "show that natural-frequency features and baseline-relative change detection generalise better to unseen damage "
            "states than the supervised classifier.")
    return txt


def discussion(d, T) -> list:
    E = [T.para("h1", "6. DISCUSSION")]
    if d["main"] is None:
        E.append(T.para("body", PENDING))
    else:
        E += discussion_text(d, T)
    return E


def discussion_text(d, T) -> list:
    m = d["main"]
    mw = m[m.level == "window"]
    lo = mw[mw.scheme == "loso"].set_index("model")
    gk = mw[mw.scheme == "group_kfold"].set_index("model")
    pb = seq_row(d, "BOCPD", f"hazard={d['cfg']['sequence_bocpd']['primary_hazard_rate']},kappa0={d['cfg']['sequence_bocpd']['primary_kappa0']}")
    phys_dec = None
    if d["phys"] is not None:
        q = d["phys"][(d["phys"].level == "window") & (d["phys"].scheme == "loso") & (d["phys"].model == "freq_decrease_threshold")]
        phys_dec = float(q["roc_auc"].iloc[0]) if len(q) else None
    rlv = m[(m.level == "recording") & (m.scheme == "loso")]
    rl_max = float(rlv["roc_auc"].max()) if len(rlv) else None
    b = d["perb"].copy()
    b["m"] = b[["delta_f1_pct", "delta_f2_pct", "delta_f3_pct"]].abs().max(axis=1)
    miss = b[~b.bocpd_detected.astype(bool) & ~b.cusum_detected.astype(bool)]
    both = b[b.bocpd_detected.astype(bool) & b.cusum_detected.astype(bool)]
    avt = d["rec"][d["rec"].test == "avt"]
    scat = [avt.groupby("scenario")[f].agg(lambda x: 100 * x.std() / x.mean()).median() for f in ("f2", "f3")]
    missed_txt = (f"The boundaries missed by both detectors ({', '.join(f'{int(r.from_scenario):02d}→{int(r.to_scenario):02d}' for r in miss.itertuples())}) "
                  f"shifted f1–f3 by at most {miss.m.min():.2f}–{miss.m.max():.2f}%, comparable to the setup-to-setup scatter within a "
                  f"scenario (median coefficient of variation {scat[0]:.2f}% for f2 and {scat[1]:.2f}% for f3), whereas every boundary "
                  f"found by both shifted at least one frequency by {both.m.min():.2f}% or more.") if len(miss) else ""
    nov = d["novelty"] or {}
    cal = nov.get("S01+S02:f1-f4")
    E = [T.para("h2", "6.1. Interpretation of the results"),
         T.para("body", "With the true sampling rate, all setups and modal identification, the natural frequencies reproduce the "
                "known sensitivity of the Z24 bridge to the pier settlement and tendon interventions, and a detector that observes "
                f"the sequence of states finds most state changes (BOCPD recall {pb.recall:.3f} at precision {pb.precision:.3f}). "
                + missed_txt),
         T.para("body", f"The supervised stage shows why the evaluation protocol matters. ML models that reach ROC-AUC values up to "
                f"{gk.drop('freq_threshold')['roc_auc'].max():.3f} when other setups of a scenario are available for training fall to "
                f"{lo.drop('freq_threshold')['roc_auc'].max():.3f} ({MODEL[lo.drop('freq_threshold')['roc_auc'].idxmax()]}) when the "
                f"scenario is unseen, below the one-parameter frequency threshold ({lo.loc['freq_threshold', 'roc_auc']:.3f}), because "
                "the three reference states differ from each other as much as the smallest interventions. SHAP explains this: "
                + (f"{100 * d['shap_grp'].set_index('group')['share'].get('modal', 0):.1f}% of the attribution fell on the record-level "
                   "modal features, so the classifier is in effect a nonlinear function of the natural frequencies that overfits the "
                   "reference states. " if d["shap_grp"] is not None else "")
                + "For a structure in service, where only its own baseline is known, change detection relative to that baseline "
                "is the more appropriate task"
                + (f": calibrated on the two pre-intervention measurements, the label-free novelty score reached a BA of "
                   f"{cal['balanced_accuracy_empirical']:.3f} without damage labels" if cal else "")
                + ". A classifier should enter the decision only after it passes a scenario-level validation criterion, which is "
                "what the gated risk score enforces."),
         T.para("h2", "6.2. Limitations"),
         T.para("body", "The labels describe the test protocol, not measured damage, and the three reference states are not "
                "identical. Only four modes could be tracked consistently, and damping could be estimated reliably for the first two. "
                "The prior of the sequence BOCPD (κ0) was changed after the default value was found to suppress new segments, and the "
                "monotone XGBoost variant, the frequency-decrease threshold and the novelty calibration were added after the first "
                "results; all are reported next to the original settings. "
                "The EMS mode shapes were not comparable across the year, so EMS modes were tracked by frequency only, and the damage "
                "onset is known only to within the dates of the test reports. The risk score marks recent changes rather than damage and is high for the first records of any sequence. The sensor layout in the dashboard is schematic and the "
                "twin contains no finite-element model."),
         T.para("h1", "7. CONCLUSIONS AND FUTURE WORK"),
         T.para("body", "This paper presented a digital-twin monitoring framework for bridges that couples modal identification, "
                "BOCPD, supervised classification, SHAP and a validation-gated risk score, and evaluated it reproducibly on all Z24 "
                f"progressive-damage recordings. Natural frequencies from FDD tracked the interventions; sequence BOCPD detected state "
                f"changes with precision {pb.precision:.3f} and recall {pb.recall:.3f}; supervised ML classifiers of reference versus "
                f"altered states generalised poorly to unseen scenarios (best ML LOSO ROC-AUC {lo.drop('freq_threshold')['roc_auc'].max():.3f}), "
                f"below a natural-frequency threshold ({lo.loc['freq_threshold', 'roc_auc']:.3f}"
                + (f"; {phys_dec:.3f} when only frequency decreases are counted, a post-hoc variant" if phys_dec is not None else "")
                + ") and a label-free novelty score; and "
                + ("the gate kept the insufficiently validated classifier out of the fused risk score." if d["risk"] is not None and not d["risk"]["classifier_term_included"]
                 else "the gate admitted the classifier to the fused risk score." if d["risk"] is not None else "the risk gate is described in Section 4.6.")),
         T.para("body", "Future work will (1) calibrate the baseline-relative novelty score on longer baseline periods, (2) identify higher "
                "modes with stochastic subspace identification, (3) couple the twin to a finite-element model for localisation and "
                "(4) connect the dashboard to live sensor streams.")]
    return E


if __name__ == "__main__":
    build(draft="--draft" in sys.argv)
