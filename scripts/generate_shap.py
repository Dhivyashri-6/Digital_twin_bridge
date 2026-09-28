from __future__ import annotations
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
from src.model_prediction import load_model
from src.shap_analysis import compute_shap
from src.sensor_mapping import aggregate_sensor_importance
from src.utils import ensure_output_dirs,load_config
def main():
 p=argparse.ArgumentParser();p.add_argument("--config",default="config.yaml");a=p.parse_args();c=load_config(a.config);paths=ensure_output_dirs(c);model,cols=load_model(paths["models"]/"xgboost.pkl",paths["models"]/"feature_columns.json");df=pd.read_csv(paths["processed"]/"features.csv");result=compute_shap(model,df[cols],paths["results"]/"shap");s=aggregate_sensor_importance(result["global_importance"]);s.to_csv(paths["processed"]/"sensor_importance.csv",index=False);print(s.head(10).to_string(index=False))
if __name__=="__main__":main()
