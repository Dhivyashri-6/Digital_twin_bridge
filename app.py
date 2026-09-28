from __future__ import annotations
import time
import pandas as pd
import plotly.express as px
import streamlit as st
from src.digital_twin import build_digital_twin
from src.model_prediction import load_model, predict_condition
from src.risk_engine import calculate_risk
from src.utils import ensure_output_dirs, load_config

st.set_page_config(page_title="Bridge Digital Twin",page_icon="🌉",layout="wide")
st.markdown("""<style>
.stApp{background:linear-gradient(145deg,#071521,#0b2232 55%,#071820);color:#e9f4f8}.block-container{max-width:1500px;padding-top:1.2rem}
.hero{border:1px solid #28546b;background:linear-gradient(90deg,#0b2638,#103b4b);padding:20px 26px;border-radius:12px}.hero h1{margin:0}.hero p{margin:.25rem 0 0;color:#81d5e8;letter-spacing:.08em}
.section{font-size:1.05rem;font-weight:700;letter-spacing:.12em;color:#92dfed;border-bottom:1px solid #28546b;padding:8px 2px;margin:14px 0 10px}
[data-testid="stMetric"]{background:#0d2a3a;border:1px solid #28546b;border-radius:10px;padding:15px 18px;min-height:115px}.event{background:#102d3b;border-left:5px solid #ffbd45;padding:14px 18px;border-radius:7px}.fine{color:#86a9b5;font-size:.82rem}
</style><div class="hero"><h1>BRIDGE DIGITAL TWIN</h1><p>EXPLAINABLE STRUCTURAL HEALTH MONITORING · Z24 HISTORICAL REPLAY</p></div>""",unsafe_allow_html=True)

c=load_config();paths=ensure_output_dirs(c);fp=paths["processed"]/"features.csv";bp=paths["processed"]/"bocpd_results.csv"
with st.sidebar:
 st.header("Monitoring controls");st.text_input("Dataset path",c["dataset"]["root_path"],disabled=True);mode=st.selectbox("Analysis mode",["Saved analysis","Simulated real-time"])
 threshold=st.slider("Change threshold",0.,1.,float(c["bocpd"]["threshold"]),.01);speed=st.selectbox("Playback speed",[.5,1,2,5],index=1,format_func=lambda x:f"{x:g}×")
if not fp.exists() or not bp.exists():st.error("Run `python scripts/run_pipeline.py --one-file-per-scenario --max-files 17` first.");st.stop()
features=pd.read_csv(fp,low_memory=False);changes=pd.read_csv(bp,low_memory=False)
for col in ("sensor","recording"):
 if col in features:features[col]=features[col].astype(str)
 if col in changes:changes[col]=changes[col].astype(str)
scenarios=sorted(features.scenario.astype(str).unique());scenario=st.sidebar.selectbox("Scenario",["All"]+scenarios)
if scenario!="All":
 sources=features.loc[features.scenario.astype(str)==scenario,"source_file"].unique();features=features[features.source_file.isin(sources)];changes=changes[changes.recording.isin(set(features.recording))]
sensors=sorted(features.sensor.unique());sensor_filter=st.sidebar.selectbox("Sensor",["All"]+sensors)
max_window=int(changes.window_id.max()) if len(changes) else 0
if "play_index" not in st.session_state:st.session_state.play_index=max_window
if "playing" not in st.session_state:st.session_state.playing=False
b1,b2,b3,b4=st.sidebar.columns(4)
if b1.button("▶"):st.session_state.playing=True
if b2.button("⏸"):st.session_state.playing=False
if b3.button("⏹"):st.session_state.playing=False;st.session_state.play_index=max_window
if b4.button("↺"):st.session_state.playing=False;st.session_state.play_index=0
if mode=="Saved analysis":st.session_state.play_index=max_window
else:
 st.session_state.play_index=min(st.session_state.play_index,max_window)
 if st.session_state.playing and st.session_state.play_index<max_window:st.session_state.play_index+=1
shown=changes[changes.window_id<=st.session_state.play_index]
st.sidebar.progress(st.session_state.play_index/max(max_window,1),text=f"Window {st.session_state.play_index} of {max_window}")
snapshot=shown.sort_values("window_id").groupby("sensor",as_index=False).tail(1) if len(shown) else pd.DataFrame()
visible_features=features[features.window_id<=st.session_state.play_index]
feature_snapshot=visible_features.sort_values("window_id").groupby("sensor",as_index=False).tail(1)
predictions={};mp=paths["models"]/"xgboost.pkl";cp=paths["models"]/"feature_columns.json"
if mp.exists() and cp.exists() and len(feature_snapshot):
 model,cols=load_model(mp,cp)
 for (_,row),pred in zip(feature_snapshot.iterrows(),predict_condition(model,cols,feature_snapshot)):predictions[str(row.sensor)]=pred
ip=paths["processed"]/"sensor_importance.csv";importance=pd.read_csv(ip) if ip.exists() else pd.DataFrame(columns=["sensor","importance_score"]);importance["sensor"]=importance.sensor.astype(str)
max_imp=float(importance.importance_score.max()) if len(importance) and importance.importance_score.max()>0 else 1.
rows=[]
for _,row in snapshot.iterrows():
 sid=str(row.sensor);change=float(row.change_probability);pred=predictions.get(sid);altered=pred["class_probabilities"].get("ALTERED_CONDITION",0.) if pred else 0.;match=importance[importance.sensor==sid];raw=float(match.iloc[0].importance_score) if len(match) else 0.
 risk=calculate_risk(change,altered,raw/max_imp,c["risk"]);votes=int(change>=threshold)+int(altered>=.5);state="ANOMALY" if votes==2 or risk["risk_score"]>=.6 else "WARNING" if votes==1 or risk["risk_score"]>=.35 else "NORMAL"
 rows.append({"sensor":sid,"status":state,"change_probability":change,"altered_probability":altered,"importance_score":raw,"risk_score":risk["risk_score"],"risk_level":risk["risk_level"],"condition":pred["predicted_class"] if pred else "Unavailable"})
status=pd.DataFrame(rows)
if len(status):
 mean_risk=float(status.risk_score.mean());health=int(round(100*(1-mean_risk)));affected=int(status.status.isin(["WARNING","ANOMALY"]).sum());agreement=affected/len(status);latest=float(status.change_probability.max());overall="POTENTIAL ANOMALY" if (status.status=="ANOMALY").any() or agreement>=.3 else "WATCH" if affected else "NORMAL";overall_risk="HIGH" if mean_risk>=.6 else "MEDIUM" if mean_risk>=.35 else "LOW";top=str(status.sort_values(["risk_score","importance_score"],ascending=False).iloc[0].sensor)
else:health,affected,latest,overall,overall_risk,top=100,0,0.,"INITIALIZING","LOW","Unavailable"

st.markdown('<div class="section">STRUCTURAL STATUS</div>',unsafe_allow_html=True)
k1,k2,k3,k4,k5=st.columns(5);k1.metric("CURRENT STATE",overall);k2.metric("PROTOTYPE HEALTH SCORE",f"{health}%",help="100 × (1 − mean prototype sensor risk)");k3.metric("MAX CHANGE PROB.",f"{latest:.1%}");k4.metric("AFFECTED SENSORS",f"{affected} / {len(status)}");k5.metric("PROTOTYPE RISK",overall_risk)
st.markdown('<div class="fine">Health score is a monitoring indicator, not structural safety certification.</div>',unsafe_allow_html=True)
st.markdown('<div class="section">SENSOR STATUS</div>',unsafe_allow_html=True)
if len(status):
 d=status.copy();d["change_probability"]=d.change_probability.map(lambda x:f"{x:.1%}");d["altered_probability"]=d.altered_probability.map(lambda x:f"{x:.1%}");d["risk_score"]=d.risk_score.map(lambda x:f"{x:.1%}")
 if sensor_filter!="All":d=d[d.sensor==sensor_filter]
 st.dataframe(d,width="stretch",hide_index=True,column_config={"importance_score":st.column_config.ProgressColumn("SHAP importance",min_value=0.,max_value=max_imp)})
else:st.info("Press Start to begin historical replay.")
left,right=st.columns([1.35,1])
with left:
 st.markdown('<div class="section">VIBRATION BEHAVIOUR & CHANGE POINTS</div>',unsafe_allow_html=True);pf=visible_features if sensor_filter=="All" else visible_features[visible_features.sensor==sensor_filter]
 fig=px.line(pf,x="window_id",y="rms",color="sensor",title="Window RMS vibration indicator");fig.update_layout(template="plotly_dark",height=330);st.plotly_chart(fig,width="stretch")
 pc=shown if sensor_filter=="All" else shown[shown.sensor==sensor_filter]
 if len(pc):
  fig=px.line(pc,x="window_id",y="change_probability",color="sensor",title="BOCPD change probability");fig.add_hline(y=threshold,line_dash="dash",line_color="#ffbd45");fig.update_layout(template="plotly_dark",height=330);st.plotly_chart(fig,width="stretch")
with right:
 st.markdown('<div class="section">DIGITAL TWIN</div>',unsafe_allow_html=True);ts=status if sensor_filter=="All" else status[status.sensor==sensor_filter];st.plotly_chart(build_digital_twin(ts),width="stretch");st.caption("Schematic sensor visualization—not verified coordinates.")
left,right=st.columns(2)
with left:
 st.markdown('<div class="section">XGBOOST — STRUCTURAL CONDITION</div>',unsafe_allow_html=True)
 if predictions:
  probs=pd.DataFrame([p["class_probabilities"] for p in predictions.values()]).mean().sort_values(ascending=False);fig=px.bar(x=probs.index,y=probs.values,labels={"x":"Condition","y":"Mean sensor probability"});fig.update_layout(template="plotly_dark",height=330);st.plotly_chart(fig,width="stretch")
 else:st.warning("No trained XGBoost model is available.")
with right:
 st.markdown('<div class="section">EXPLAINABLE AI — SHAP</div>',unsafe_allow_html=True);sp=paths["results"]/"shap/global_feature_importance.csv"
 if sp.exists():
  sh=pd.read_csv(sp).head(10).sort_values("mean_abs_shap");fig=px.bar(sh,x="mean_abs_shap",y="feature",orientation="h");fig.update_traces(marker_color="#5fd4e5");fig.update_layout(template="plotly_dark",height=330);st.plotly_chart(fig,width="stretch");st.caption("Model attribution, not physical causation.")
 else:st.warning("SHAP output is unavailable.")
st.markdown('<div class="section">DETECTION LOG</div>',unsafe_allow_html=True);events=shown[shown.change_probability>=threshold]
if len(events):
 logs=[]
 for _,e in events.tail(50).iloc[::-1].iterrows():
  pred=predictions.get(str(e.sensor));logs.append({"Time":f"t={e.timestamp:.2f}s" if pd.notna(e.timestamp) else f"Window {int(e.window_id)}","Sensor":str(e.sensor),"Event":"Statistical change detected","Probability":f"{e.change_probability:.1%}","XGBoost":pred["predicted_class"] if pred else "Unavailable","Explanation":f"Sensor {e.sensor} prioritized for review"})
 st.dataframe(pd.DataFrame(logs),width="stretch",hide_index=True)
else:st.info("No change probability crossed the selected threshold.")
if overall in {"POTENTIAL ANOMALY","WATCH"}:st.markdown(f'<div class="event"><b>STRUCTURAL MONITORING EVENT</b><br>State: {overall} · Influential sensor: {top} · Maximum change probability: {latest:.1%} · Prototype risk: {overall_risk}<br><br><b>Recommended action:</b> Prioritize inspection of the region associated with highlighted sensors.</div>',unsafe_allow_html=True)
st.markdown('<div class="fine">Historical replay, not live IoT. Statistical changes and ML classifications do not confirm damage or safety.</div>',unsafe_allow_html=True)
if mode=="Simulated real-time" and st.session_state.playing and st.session_state.play_index<max_window:time.sleep(float(c["dashboard"]["refresh_interval"])/speed);st.rerun()
