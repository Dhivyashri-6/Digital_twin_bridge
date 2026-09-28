from __future__ import annotations
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
from src.model_training import train_xgboost
from src.utils import ensure_output_dirs,load_config
def main():
 p=argparse.ArgumentParser();p.add_argument("--config",default="config.yaml");a=p.parse_args();c=load_config(a.config);paths=ensure_output_dirs(c);result=train_xgboost(pd.read_csv(paths["processed"]/"features.csv"),c["model"],paths["models"],paths["results"]/"metrics");print(result["metrics"])
if __name__=="__main__":main()
