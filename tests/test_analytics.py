import numpy as np
import pandas as pd
from src.bocpd import StudentTBOCPD,run_bocpd
from src.risk_engine import calculate_risk
from src.sensor_mapping import extract_sensor_id,aggregate_sensor_importance

def test_bocpd_probabilities():
    d=StudentTBOCPD(.05,.5);out=[d.update(x) for x in np.r_[np.zeros(30),np.ones(30)*5]];assert all(0<=p<=1 and r>=0 for p,r in out)

def test_bocpd_frame():
    f=pd.DataFrame({"sensor":["1"]*5,"recording":["a"]*5,"window_id":range(5),"rms":[1,1,1,4,4]});out=run_bocpd(f,"rms");assert len(out)==5

def test_sensor_mapping():
    assert extract_sensor_id("RMS_Sensor_7")=="7";x=aggregate_sensor_importance(pd.DataFrame({"feature":["Sensor_7_rms","Sensor_7_std"],"mean_abs_shap":[.2,.3]}));assert x.iloc[0].importance_score==.5

def test_risk_engine():
    r=calculate_risk(1,1,1,{"bocpd_weight":.3,"ml_weight":.5,"attribution_weight":.2,"thresholds":{"medium":.35,"high":.6,"critical":.8}});assert r["risk_level"]=="CRITICAL";assert r["risk_score"]==1
