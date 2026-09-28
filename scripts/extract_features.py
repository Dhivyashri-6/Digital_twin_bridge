from __future__ import annotations
import argparse, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.data_loader import load_dataset
from src.preprocessing import preprocess_records
from src.feature_extraction import extract_features
from src.utils import ensure_output_dirs,load_config
def main():
 p=argparse.ArgumentParser();p.add_argument("--config",default="config.yaml");p.add_argument("--dataset-root");p.add_argument("--max-files",type=int,default=5);p.add_argument("--archives",action="store_true");a=p.parse_args();c=load_config(a.config);paths=ensure_output_dirs(c)
 rec=load_dataset(a.dataset_root or c["dataset"]["root_path"],a.max_files,a.archives,c["dataset"].get("sampling_rate"));df=extract_features(preprocess_records(rec,c["preprocessing"],c["window"]));df.to_csv(paths["processed"]/"features.csv",index=False);print(f"Saved {len(df)} feature rows")
if __name__=="__main__":main()
