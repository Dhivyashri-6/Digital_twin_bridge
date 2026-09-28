from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data_loader import load_dataset
from src.preprocessing import preprocess_records
from src.utils import ensure_output_dirs, load_config

def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",default="config.yaml"); p.add_argument("--dataset-root"); p.add_argument("--max-files",type=int,default=5); p.add_argument("--archives",action="store_true"); a=p.parse_args()
    c=load_config(a.config); paths=ensure_output_dirs(c); root=a.dataset_root or c["dataset"]["root_path"]
    records=load_dataset(root,a.max_files,a.archives,c["dataset"].get("sampling_rate")); windows=preprocess_records(records,c["preprocessing"],c["window"])
    manifest=[{k:getattr(w,k) for k in ("scenario","setup","sensor","window_id","start_sample","end_sample","start_time","end_time","label","source_file","recording")} for w in windows]
    (paths["processed"] / "windows_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8"); print(f"Created {len(windows)} windows from {len(records)} signals")
if __name__ == "__main__": main()
