from __future__ import annotations

from collections import Counter
from pathlib import Path
import zipfile
import matplotlib.pyplot as plt
import numpy as np

from .data_loader import discover_files, infer_metadata, inspect_file, load_aaa_bytes
from .utils import write_json


def inspect_dataset(root: str | Path, figure_dir: str | Path, report_path: str | Path, max_content_files: int | None = None,
                    max_archives_to_open: int = 25) -> dict:
    root, figure_dir = Path(root).resolve(), Path(figure_dir)
    paths = discover_files(root, include_archives=True)
    counts = Counter(p.suffix.lower() for p in paths)
    folder_names = [p.name for p in root.iterdir() if p.is_dir()]
    details, errors, samples, variables = [], [], [], []
    content_paths = [p for p in paths if p.suffix.lower() != ".zip"]
    if max_content_files:
        representatives = []
        for extension in (".aaa", ".mat", ".csv", ".txt"):
            item = next((p for p in content_paths if p.suffix.lower() == extension), None)
            if item: representatives.append(item)
        content_paths = list(dict.fromkeys(representatives + content_paths))[:max_content_files]
    for path in content_paths:
        try:
            info = inspect_file(path)
            info.update(infer_metadata(path, root)); details.append(info)
            variables.extend({"file": str(path), **v} for v in info["variables"])
            if not samples:
                from .data_loader import load_file
                records = load_file(path, root)
                if records: samples.append(records[0])
        except Exception as exc:
            errors.append({"file": str(path), "error": str(exc)})
    archive_members = Counter()
    representative_archive = None
    archive_paths = [p for p in paths if p.suffix.lower() == ".zip"]
    for path in archive_paths[:max_archives_to_open]:
        try:
            with zipfile.ZipFile(path) as archive:
                for member in archive.infolist(): archive_members[Path(member.filename).suffix.lower()] += 1
                if not samples:
                    aaa = next((m for m in archive.infolist() if Path(m.filename).suffix.lower() == ".aaa"), None)
                    if aaa:
                        arr, meta = load_aaa_bytes(archive.read(aaa), f"{path}!{aaa.filename}")
                        representative_archive = {"archive": str(path), "member": aaa.filename, "shape": list(np.shape(arr)), "metadata": meta}
        except Exception as exc:
            errors.append({"file": str(path), "error": str(exc)})
    figure_dir.mkdir(parents=True, exist_ok=True)
    representative = None
    if samples:
        record = samples[0]; representative = {"file": record.source_file, "sensor": record.sensor, "samples": len(record.signal), "sampling_rate": record.sampling_rate}
        fig, ax = plt.subplots(figsize=(12, 4)); ax.plot(record.signal[:min(len(record.signal), 10000)], linewidth=.6)
        ax.set(title=f"Representative signal: {Path(record.source_file).name}", xlabel="Sample", ylabel="Amplitude"); ax.grid(alpha=.25)
        fig.tight_layout(); fig.savefig(figure_dir / "representative_signal.png", dpi=150); plt.close(fig)
    scenario_counts = Counter(str(d.get("scenario", "unknown")) for d in details)
    sensor_counts = Counter(str(d.get("sensor", "unknown")) for d in details)
    all_vars = [v for v in variables if v.get("mean") is not None]
    report = {"dataset_root": str(root), "folders_detected": folder_names, "total_files": len(paths), "file_types": dict(counts),
              "archive_member_types_sample": dict(archive_members), "archives_opened": min(len(archive_paths), max_archives_to_open),
              "archive_count": len(archive_paths), "files_inspected": len(details), "possible_scenarios": dict(scenario_counts),
              "possible_sensors": dict(sensor_counts), "representative_file": representative or representative_archive,
              "signal_lengths": [int(np.prod(v["shape"])) for v in all_vars],
              "missing_values": sum(v["nan_count"] for v in all_vars), "infinite_values": sum(v["inf_count"] for v in all_vars),
              "variables": variables, "errors": errors}
    write_json(Path(report_path), report)
    return report


def print_report(report: dict) -> None:
    line = "=" * 56
    print(f"{line}\nZ24 DATASET INSPECTION\n{line}")
    print(f"Dataset root: {report['dataset_root']}\n")
    print("Folders detected:"); [print(f"- {name}") for name in report["folders_detected"]]
    print("\nFile types:"); [print(f"{ext or '[none]'} : {count}") for ext, count in sorted(report["file_types"].items())]
    print(f"\nTotal files: {report['total_files']}\nFiles content-inspected: {report['files_inspected']}")
    print(f"Possible scenarios: {report['possible_scenarios']}")
    print(f"Possible sensors: {len(report['possible_sensors'])} identifiers")
    lengths = report["signal_lengths"]
    print(f"Signal lengths: min={min(lengths) if lengths else 'n/a'}, max={max(lengths) if lengths else 'n/a'}")
    print(f"Missing values: {report['missing_values']}; Inf values: {report['infinite_values']}")
    print(f"\n{line}\nREPRESENTATIVE FILE\n{line}")
    for key, value in (report.get("representative_file") or {}).items(): print(f"{key}: {value}")
    if report["errors"]: print(f"\nWarnings/errors: {len(report['errors'])} (saved in inspection_report.json)")
    print(line)
