from __future__ import annotations
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset_inspector import inspect_dataset, print_report
from src.utils import ensure_output_dirs, load_config, resolve_project_path

def main():
    parser=argparse.ArgumentParser(description="Inspect Z24 files without modifying the source dataset")
    parser.add_argument("--config", default="config.yaml"); parser.add_argument("--dataset-root"); parser.add_argument("--max-content-files", type=int, default=20)
    args=parser.parse_args(); config=load_config(args.config); paths=ensure_output_dirs(config)
    root=Path(args.dataset_root).resolve() if args.dataset_root else Path(config["dataset"]["root_path"]).expanduser().resolve()
    report=inspect_dataset(root, paths["results"] / "figures/dataset_inspection", paths["results"] / "reports/inspection_report.json", args.max_content_files)
    print_report(report)
if __name__ == "__main__": main()
