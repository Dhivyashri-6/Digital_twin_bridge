import numpy as np
import pandas as pd
import pytest
from src.model_training import prepare_supervised_data,train_xgboost

def test_training_rejects_missing_labels():
    with pytest.raises(ValueError): prepare_supervised_data(pd.DataFrame({"rms":[1,2],"label":["unknown","unknown"]}))

def test_xgboost_grouped(tmp_path):
    n=80;rng=np.random.default_rng(2);labels=np.array(["healthy"]*40+["damaged"]*40)
    f=pd.DataFrame({"rms":np.r_[rng.normal(0,1,40),rng.normal(4,1,40)],"std":rng.random(n),"label":labels,"recording":[f"r{i//10}" for i in range(n)]})
    result=train_xgboost(f,{"n_estimators":20,"random_state":1,"test_size":.25,"min_samples_for_training":20},tmp_path/"m",tmp_path/"metrics");assert 0<=result["metrics"]["accuracy"]<=1

def test_shap_optional_import_and_compute(tmp_path):
    pytest.importorskip("shap")
    from src.shap_analysis import compute_shap
    n=40;f=pd.DataFrame({"rms":np.r_[np.zeros(20),np.ones(20)],"std":np.linspace(0,1,n),"label":["a"]*20+["b"]*20,"recording":[f"r{i//5}" for i in range(n)]})
    trained=train_xgboost(f,{"n_estimators":10,"random_state":3,"test_size":.25,"min_samples_for_training":20},tmp_path/"m",tmp_path/"metrics");result=compute_shap(trained["model"],f[trained["features"]],tmp_path/"shap");assert not result["global_importance"].empty
