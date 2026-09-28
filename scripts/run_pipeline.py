from __future__ import annotations
import argparse,re,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
from src.bocpd import run_bocpd
from src.data_loader import load_dataset
from src.dataset_inspector import inspect_dataset,print_report
from src.feature_extraction import extract_features
from src.model_training import train_xgboost
from src.shap_analysis import compute_shap
from src.sensor_mapping import aggregate_sensor_importance
from src.preprocessing import preprocess_records
from src.risk_engine import calculate_risk
from src.utils import ensure_output_dirs,load_config,write_json

def apply_label_rules(frame,rules):
 frame=frame.copy(); mapping=[]
 for pattern,label in rules.items():
  mask=frame["source_file"].astype(str).str.contains(pattern,case=False,regex=True);frame.loc[mask,"label"]=str(label);mapping.append({"pattern":pattern,"label":label,"matched_windows":int(mask.sum())})
 return frame,{"source":"User-configured regex rules applied to source_file","rules":mapping,"warning":"No engineering meaning is inferred by the software."}

def main():
 p=argparse.ArgumentParser(description="Run the Z24 workflow without modifying raw files");p.add_argument("--config",default="config.yaml");p.add_argument("--dataset-root");p.add_argument("--max-files",type=int,default=17);p.add_argument("--archives",action="store_true");p.add_argument("--one-file-per-scenario",action="store_true");p.add_argument("--skip-inspection",action="store_true");a=p.parse_args()
 c=load_config(a.config);paths=ensure_output_dirs(c);root=Path(a.dataset_root or c["dataset"]["root_path"]).expanduser().resolve()
 if not a.skip_inspection: print_report(inspect_dataset(root,paths["results"]/"figures/dataset_inspection",paths["results"]/"reports/inspection_report.json",min(a.max_files,20)))
 records=load_dataset(root,a.max_files,a.archives,c["dataset"].get("sampling_rate"),a.one_file_per_scenario);print(f"Loaded {len(records)} signal channels")
 windows=preprocess_records(records,c["preprocessing"],c["window"]);features=extract_features(windows);features,label_map=apply_label_rules(features,c["dataset"].get("label_rules",{}));features.to_csv(paths["processed"]/"features.csv",index=False);write_json(paths["processed"]/"label_mapping.json",label_map);print(f"Saved {len(features)} feature windows")
 bc=c["bocpd"];bocpd=run_bocpd(features,bc["feature"],bc["hazard_rate"],bc["threshold"],bc.get("max_run_length",500));bocpd.to_csv(paths["processed"]/"bocpd_results.csv",index=False);print(f"BOCPD complete: {int(bocpd.change_detected.sum())} threshold crossings")
 model_result=None
 try:
  model_result=train_xgboost(features,c["model"],paths["models"],paths["results"]/"metrics");print(f"XGBoost accuracy: {model_result['metrics']['accuracy']:.4f}")
  explanation=compute_shap(model_result["model"],features[model_result["features"]],paths["results"]/"shap")
  explained_rows=features.iloc[:len(explanation["values"])].copy();explained_rows["importance_score"]=abs(explanation["values"]).sum(axis=1)
  sensor_importance=explained_rows.groupby("sensor",as_index=False)["importance_score"].mean().sort_values("importance_score",ascending=False);sensor_importance["rank"]=range(1,len(sensor_importance)+1)
  sensor_importance.to_csv(paths["processed"]/"sensor_importance.csv",index=False);print("SHAP and sensor importance saved")
 except ValueError as exc: write_json(paths["results"]/"metrics/model_status.json",{"status":"not_trained","reason":str(exc)});print(f"Supervised stage skipped safely: {exc}")
 risk_rows=[]
 for _,row in bocpd.iterrows(): risk_rows.append({**row.to_dict(),**calculate_risk(row.change_probability,None,None,c["risk"])})
 pd.DataFrame(risk_rows).drop(columns=["components","weights","disclaimer"]).to_csv(paths["processed"]/"risk_results.csv",index=False)
 print("Pipeline completed. All reported values were calculated from loaded data.")
if __name__=="__main__":main()
