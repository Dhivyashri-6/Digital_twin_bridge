"""Streamlit digital-twin dashboard: historical replay of the v2 study outputs along the Z24 scenario sequence.

Everything shown is read from files written by the pipeline (scripts/run_*.py); nothing is recomputed here.
Run the study first (python scripts/run_all_experiments.py), then: streamlit run app.py
"""
from __future__ import annotations

import json
import time

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.digital_twin import build_digital_twin
from src.utils import ensure_output_dirs, load_config

st.set_page_config(page_title="Bridge Digital Twin", page_icon="🌉", layout="wide")
st.markdown("""<style>
.stApp{background:linear-gradient(145deg,#071521,#0b2232 55%,#071820);color:#e9f4f8}.block-container{max-width:1500px;padding-top:1.2rem}
.hero{border:1px solid #28546b;background:linear-gradient(90deg,#0b2638,#103b4b);padding:20px 26px;border-radius:12px}.hero h1{margin:0}.hero p{margin:.25rem 0 0;color:#81d5e8;letter-spacing:.08em}
.section{font-size:1.05rem;font-weight:700;letter-spacing:.12em;color:#92dfed;border-bottom:1px solid #28546b;padding:8px 2px;margin:14px 0 10px}
[data-testid="stMetric"]{background:#0d2a3a;border:1px solid #28546b;border-radius:10px;padding:15px 18px;min-height:115px}
[data-testid="stMetric"] *{color:#e9f4f8 !important;opacity:1 !important}[data-testid="stMetricLabel"] *{color:#92dfed !important;font-weight:600}
[data-testid="stMetricValue"]{font-size:1.6rem !important}.stApp p,.stApp .stCaption,.stApp label{color:#c9dde4}.event{background:#102d3b;border-left:5px solid #ffbd45;padding:14px 18px;border-radius:7px}.fine{color:#86a9b5;font-size:.82rem}
</style><div class="hero"><h1>BRIDGE DIGITAL TWIN</h1><p>Z24 PROGRESSIVE DAMAGE TESTS · HISTORICAL REPLAY OF THE STUDY OUTPUTS</p></div>""",
            unsafe_allow_html=True)

cfg = load_config()
paths = ensure_output_dirs(cfg)
R, P = paths["results"], paths["processed"]
needed = {"recordings": P / "recordings.csv", "series": R / "change_detection" / "sequence_series.csv",
          "detections": R / "change_detection" / "sequence_detection_comparison.csv",
          "sensor_risk": R / "risk" / "sensor_risk.csv", "recording_states": R / "risk" / "recording_states.csv",
          "risk_summary": R / "risk" / "risk_summary.json", "novelty": R / "novelty" / "novelty_scores.csv"}
missing = [str(p) for p in needed.values() if not p.exists()]
if missing:
    st.error("Study outputs missing – run `python scripts/run_all_experiments.py` first:\n\n" + "\n".join(missing))
    st.stop()


@st.cache_data
def load():
    rec = pd.read_csv(needed["recordings"], dtype={"scenario": str, "setup": str})
    series = pd.read_csv(needed["series"], dtype={"scenario": str, "setup": str})
    seq = series.merge(rec[["recording", "label", "f1", "f2", "f3", "f4"]], on="recording", how="left").reset_index(drop=True)
    det = pd.read_csv(needed["detections"])
    sens = pd.read_csv(needed["sensor_risk"], dtype={"scenario": str, "sensor": str})
    states = pd.read_csv(needed["recording_states"], dtype={"scenario": str})
    nov = pd.read_csv(needed["novelty"], dtype={"scenario": str})
    summary = json.loads(needed["risk_summary"].read_text())
    return seq, det, sens, states, nov, summary


seq, det, sens, states, nov, risk_summary = load()
sb, sc = cfg["sequence_bocpd"], cfg["cusum"]
h = sb["primary_hazard_rate"]
primary = {m: det[(det.method == m) & det.primary.astype(bool)].iloc[0] for m in ("BOCPD", "CUSUM")}
detected = {m: [d["t"] for d in json.loads(primary[m].detection_list)] for m in primary}

with st.sidebar:
    st.header("Monitoring controls")
    st.text_input("Dataset path", cfg["dataset"]["root_path"], disabled=True)
    mode = st.selectbox("Analysis mode", ["Saved analysis", "Simulated real-time"])
    variant = st.selectbox("Risk score", ["gated", "ungated"],
                           help="gated: classifier term excluded unless its LOSO ROC-AUC reaches the configured minimum")
    bases = sorted(nov.baseline.unique(), key=lambda b: (b != "S01+S02", b))
    nov_base = st.selectbox("Novelty baseline", bases, help="S01+S02: both measurements taken before any intervention (calibrated setting)")
    nov_feat = st.selectbox("Novelty features", sorted(nov.features.unique(), reverse=True))
    speed = st.selectbox("Playback speed", [.5, 1, 2, 5], index=1, format_func=lambda x: f"{x:g}×")
    jump = st.selectbox("Jump to scenario", ["–"] + sorted(seq.scenario.unique()))

last = len(seq) - 1
if "play_index" not in st.session_state:
    st.session_state.play_index = last
if "playing" not in st.session_state:
    st.session_state.playing = False
b1, b2, b3, b4 = st.sidebar.columns(4)
if b1.button("▶"):
    st.session_state.playing = True
if b2.button("⏸"):
    st.session_state.playing = False
if b3.button("⏹"):
    st.session_state.playing, st.session_state.play_index = False, last
if b4.button("↺"):
    st.session_state.playing, st.session_state.play_index = False, 0
if jump != "–":
    st.session_state.play_index = int(seq.index[seq.scenario == jump].max())
elif mode == "Saved analysis":
    st.session_state.play_index = last
elif st.session_state.playing and st.session_state.play_index < last:
    st.session_state.play_index += 1
t = st.session_state.play_index
st.sidebar.progress(t / max(last, 1), text=f"Recording {t + 1} of {last + 1}")

cur = seq.iloc[t]
shown = seq.iloc[: t + 1]
nv = nov[(nov.baseline == nov_base) & (nov.features == nov_feat)].set_index("recording")
thr = float(nv.threshold_empirical.iloc[0])
nov_now = float(nv.score.get(cur.recording, float("nan")))
st_row = states[(states.variant == variant) & (states.recording == cur.recording)]
state = st_row.state.iloc[0] if len(st_row) else "UNAVAILABLE"
health = int(st_row.health_score.iloc[0]) if len(st_row) else 0
base_f1 = seq.loc[seq.scenario == f"{cfg['modal']['baseline_scenario']:02d}", "f1"].mean()

st.markdown('<div class="section">STRUCTURAL STATUS</div>', unsafe_allow_html=True)
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("RECORDING", cur.recording, f"scenario {cur.scenario} · {cur.label}", delta_color="off")
k2.metric("TWIN STATE", state, f"health score {health}%", delta_color="off")
k3.metric("RECENT CHANGE (BOCPD)", f"{cur[f'bocpd_recent_h{h}']:.1%}", f"P(r ≤ {sb.get('recent', 4)}) at h = {h}", delta_color="off")
k4.metric("NOVELTY DISTANCE", f"{nov_now:.2f}", "above threshold" if nov_now > thr else f"threshold {thr:.2f}",
          delta_color="inverse" if nov_now > thr else "off")
k5.metric("f1 SHIFT vs S01", f"{100 * (cur.f1 / base_f1 - 1):+.2f}%" if pd.notna(cur.f1) else "n/a", f"f1 = {cur.f1:.3f} Hz",
          delta_color="off")
ml = "included" if risk_summary["classifier_term_included"] else "excluded"
st.markdown(f'<div class="fine">Classifier term {ml} by the validation gate (LOSO ROC-AUC '
            f'{risk_summary["classifier_auc"]["loso"]:.3f}, minimum {risk_summary["min_auc"]}). The risk score marks recent '
            'changes of state, not damage severity; the health score is a monitoring indicator, not a safety certificate.</div>',
            unsafe_allow_html=True)

left, right = st.columns([1.35, 1])
with left:
    st.markdown('<div class="section">NATURAL FREQUENCIES ALONG THE SCENARIO SEQUENCE</div>', unsafe_allow_html=True)
    long = shown.reset_index().melt(id_vars=["index", "scenario"], value_vars=["f1", "f2", "f3", "f4"], var_name="mode",
                                    value_name="frequency_hz")
    fig = px.scatter(long, x="index", y="frequency_hz", color="mode", hover_data=["scenario"])
    for b in seq.index[seq.scenario.ne(seq.scenario.shift())][1:]:
        fig.add_vline(x=b - .5, line_width=.5, line_color="#3d6272")
    fig.update_layout(template="plotly_dark", height=330, xaxis_title="recording (scenario order)", yaxis_title="Hz",
                      xaxis_range=[-1, last + 1])
    st.plotly_chart(fig, width="stretch", theme=None)

    st.markdown('<div class="section">CHANGE DETECTION AND NOVELTY</div>', unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=shown.index, y=shown[f"bocpd_recent_h{h}"], name="BOCPD recent-change prob.", line={"color": "#5fd4e5"}))
    ns = shown.recording.map(nv.score)
    fig.add_trace(go.Scatter(x=shown.index, y=ns / max(thr, 1e-9), name="novelty / threshold", yaxis="y2", line={"color": "#ffbd45"}))
    fig.add_hline(y=1, line_dash="dash", line_color="#ffbd45", yref="y2")
    for x in [d for d in detected["BOCPD"] if d <= t]:
        fig.add_vline(x=x, line_color="#e74c3c", line_width=1)
    fig.update_layout(template="plotly_dark", height=330, xaxis_range=[-1, last + 1], xaxis_title="recording (scenario order)",
                      yaxis={"title": "probability", "range": [0, 1.05]},
                      yaxis2={"title": "novelty / threshold", "overlaying": "y", "side": "right", "type": "log",
                              "tickvals": [0.1, 0.3, 1, 3, 10], "ticktext": ["0.1", "0.3", "1", "3", "10"]},
                      legend={"orientation": "h", "y": 1.12})
    st.plotly_chart(fig, width="stretch", theme=None)
    st.caption("Red lines: BOCPD detections (primary setting). Novelty: Mahalanobis distance of the natural frequencies from "
               "the selected baseline, divided by the empirical threshold (largest leave-one-out baseline distance).")

with right:
    st.markdown('<div class="section">DIGITAL TWIN – SENSOR STATES</div>', unsafe_allow_html=True)
    ts = sens[(sens.variant == variant) & (sens.recording == cur.recording)].rename(columns={"B": "change_probability",
                                                                                           "A": "importance_score"})
    ts["status"] = ts["state"]
    st.plotly_chart(build_digital_twin(ts), width="stretch", theme=None)
    st.caption("Schematic sensor layout – not surveyed coordinates.")
    if len(ts):
        st.dataframe(ts[["sensor", "state", "risk_score", "risk_level", "change_probability", "importance_score"]]
                     .sort_values("risk_score", ascending=False), width="stretch", hide_index=True, height=260,
                     column_config={"risk_score": st.column_config.ProgressColumn("risk", min_value=0., max_value=1.),
                                    "importance_score": st.column_config.ProgressColumn("SHAP attribution", min_value=0., max_value=1.)})

left, right = st.columns(2)
with left:
    st.markdown('<div class="section">EXPLAINABLE AI – SHAP (LOSO, ALL HELD-OUT WINDOWS)</div>', unsafe_allow_html=True)
    sp = R / "shap" / "global_feature_importance.csv"
    if sp.exists():
        sh = pd.read_csv(sp).head(10).sort_values("mean_abs_shap")
        fig = px.bar(sh, x="mean_abs_shap", y="feature", color="group", orientation="h", log_x=True)
        fig.update_layout(template="plotly_dark", height=330)
        st.plotly_chart(fig, width="stretch", theme=None)
        st.caption("Model attribution, not physical causation; the record-level modal features dominate.")
    else:
        st.warning("SHAP output is unavailable.")
with right:
    st.markdown('<div class="section">SCENARIO SUMMARY</div>', unsafe_allow_html=True)
    ss = pd.read_csv(R / "risk" / "scenario_states.csv", dtype={"scenario": str})
    ss = ss[ss.variant == variant][["scenario", "label", "recordings", "mean_risk", "normal", "watch", "potential_anomaly"]]
    nsum = nv[nv.role == "test"].groupby("scenario").score.apply(lambda x: float((x > thr).mean()))
    ss["novelty_alarm_rate"] = ss.scenario.map(nsum)
    st.dataframe(ss, width="stretch", hide_index=True, height=330)

st.markdown('<div class="section">DETECTION LOG</div>', unsafe_allow_html=True)
logs = []
for m in ("BOCPD", "CUSUM"):
    for d in json.loads(primary[m].detection_list):
        if d["t"] <= t:
            r_ = seq.iloc[d["t"]]
            logs.append({"Recording #": d["t"] + 1, "Recording": r_.recording, "Scenario": r_.scenario, "Detector": m,
                         "Event": "state change detected", "Change located at #": d["location"] + 1})
for i, r_ in shown.iterrows():
    s_ = nv.score.get(r_.recording)
    if s_ is not None and s_ > thr and nv.role.get(r_.recording) == "test":
        logs.append({"Recording #": i + 1, "Recording": r_.recording, "Scenario": r_.scenario, "Detector": "novelty",
                     "Event": f"distance {s_:.2f} > {thr:.2f}", "Change located at #": None})
if logs:
    st.dataframe(pd.DataFrame(logs).sort_values("Recording #", ascending=False), width="stretch", hide_index=True, height=260)
else:
    st.info("No detection up to the current recording.")
if state == "POTENTIAL ANOMALY":
    st.markdown(f'<div class="event"><b>STRUCTURAL MONITORING EVENT</b><br>{cur.recording}: {state} · recent-change probability '
                f'{cur[f"bocpd_recent_h{h}"]:.1%} · novelty {nov_now:.2f} (threshold {thr:.2f})<br><br><b>Recommended action:</b> '
                'check the direction of the frequency shift and schedule an inspection if the frequencies decreased.</div>',
                unsafe_allow_html=True)
st.markdown('<div class="fine">Historical replay of recorded tests, not live IoT. Detections indicate statistical changes of '
            'the modal properties and do not by themselves confirm damage or safety.</div>', unsafe_allow_html=True)
if mode == "Simulated real-time" and st.session_state.playing and st.session_state.play_index < last:
    time.sleep(float(cfg["dashboard"]["refresh_interval"]) / speed)
    st.rerun()
