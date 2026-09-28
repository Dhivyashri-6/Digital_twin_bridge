from __future__ import annotations
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
from src.bocpd import run_bocpd
from src.utils import ensure_output_dirs,load_config
def main():
 p=argparse.ArgumentParser();p.add_argument("--config",default="config.yaml");p.add_argument("--feature");a=p.parse_args();c=load_config(a.config);paths=ensure_output_dirs(c);df=pd.read_csv(paths["processed"]/"features.csv");bc=c["bocpd"];out=run_bocpd(df,a.feature or bc["feature"],bc["hazard_rate"],bc["threshold"],bc.get("max_run_length",500));out.to_csv(paths["processed"]/"bocpd_results.csv",index=False);print(f"Saved {len(out)} BOCPD rows; detected {int(out.change_detected.sum())} statistical changes")
if __name__=="__main__":main()
