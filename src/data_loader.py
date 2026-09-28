from __future__ import annotations

import io
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np
from scipy.io import loadmat


SUPPORTED_EXTENSIONS = {".aaa", ".mat", ".csv", ".txt"}


@dataclass
class SignalRecord:
    scenario: str = "unknown"
    setup: str = "unknown"
    sensor: str = "unknown"
    signal: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    sampling_rate: float | None = None
    source_file: str = "unknown"
    label: str = "unknown"
    recording: str = "unknown"
    channel: str = "unknown"
    metadata: dict = field(default_factory=dict)


def validate_dataset_root(root: str | Path, expected_folders: list[str] | None = None) -> Path:
    path = Path(root).expanduser().resolve()
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {path}")
    if expected_folders and not any((path / name).is_dir() for name in expected_folders):
        raise FileNotFoundError(f"None of the expected Z24 folders were found under {path}: {expected_folders}")
    return path


def discover_files(root: str | Path, extensions: set[str] | None = None, include_archives: bool = False) -> list[Path]:
    root = validate_dataset_root(root)
    extensions = {x.lower() for x in (extensions or SUPPORTED_EXTENSIONS)}
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in extensions]
    if include_archives:
        files.extend(p for p in root.rglob("*.zip") if p.is_file())
    return sorted(set(files))


def _numeric_lines(text: str) -> tuple[np.ndarray, dict]:
    rows: list[list[float]] = []
    header: list[str] = []
    delimiter_re = re.compile(r"[\s,;]+")
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", "%", "//")):
            continue
        tokens = [t for t in delimiter_re.split(line) if t]
        try:
            values = [float(t.replace("D", "E").replace("d", "e")) for t in tokens]
        except ValueError:
            header.append(line)
            continue
        if values:
            rows.append(values)
    if not rows:
        raise ValueError("No numerical samples were found in the ASCII content")
    widths = [len(row) for row in rows]
    common = max(set(widths), key=widths.count)
    data = np.asarray([row for row in rows if len(row) == common], dtype=float)
    # Common Z24 AAA layout: text header, sample count, sample interval, then values.
    sample_interval = None
    declared_samples = None
    if common == 1 and len(data) >= 3:
        candidate_n, candidate_dt = data[0, 0], data[1, 0]
        if candidate_n.is_integer() and candidate_n > 0 and 0 < candidate_dt < 100 and len(data) - 2 >= min(candidate_n, 10):
            declared_samples = int(candidate_n)
            sample_interval = float(candidate_dt)
            data = data[2 : 2 + declared_samples]
    return data.squeeze() if common == 1 else data, {"header": header, "declared_samples": declared_samples, "sample_interval": sample_interval}


def load_aaa_file(path: str | Path) -> tuple[np.ndarray, dict]:
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise ValueError(f"Could not read ASCII file {path}: {exc}") from exc
    return _numeric_lines(text)


def load_aaa_bytes(content: bytes, source: str) -> tuple[np.ndarray, dict]:
    try:
        return _numeric_lines(content.decode("utf-8", errors="replace"))
    except ValueError as exc:
        raise ValueError(f"Could not parse archived ASCII file {source}: {exc}") from exc


def _walk_mat(value, name: str = "") -> Iterator[tuple[str, np.ndarray]]:
    if isinstance(value, np.ndarray):
        if value.dtype.names:
            for field_name in value.dtype.names:
                yield from _walk_mat(value[field_name], f"{name}.{field_name}".strip("."))
        elif value.dtype == object:
            for index, item in enumerate(value.flat):
                yield from _walk_mat(item, f"{name}[{index}]")
        elif np.issubdtype(value.dtype, np.number) and value.size:
            yield name, value
    elif hasattr(value, "_fieldnames"):
        for field_name in value._fieldnames:
            yield from _walk_mat(getattr(value, field_name), f"{name}.{field_name}".strip("."))


def load_mat_file(path: str | Path) -> dict[str, np.ndarray]:
    path = Path(path)
    try:
        raw = loadmat(path, squeeze_me=True, struct_as_record=False)
    except NotImplementedError as exc:
        raise ValueError(f"{path} appears to use MATLAB v7.3/HDF5; install h5py or convert it first") from exc
    except Exception as exc:
        raise ValueError(f"Could not interpret MATLAB file {path}: {exc}") from exc
    arrays: dict[str, np.ndarray] = {}
    for key, value in raw.items():
        if key.startswith("__"):
            continue
        for nested_name, array in _walk_mat(value, key):
            arrays[nested_name] = np.asarray(array)
    if not arrays:
        raise ValueError(f"No numerical MATLAB arrays found in {path}; keys were {list(raw)}")
    return arrays


def infer_metadata(path: str | Path, dataset_root: str | Path | None = None) -> dict:
    path = Path(path)
    parts = path.parts
    rel_parts = path.relative_to(Path(dataset_root)).parts if dataset_root and path.is_relative_to(Path(dataset_root)) else parts
    scenario = next((p for p in rel_parts if re.fullmatch(r"\d{1,3}", p)), "unknown")
    setup_match = re.search(r"setup\s*0*(\d+)", path.stem, re.I)
    setup = setup_match.group(1) if setup_match else next((p for p in rel_parts if p.lower() in {"avt", "fvt"}), "unknown")
    sensor_match = re.search(r"(?:sensor|ch(?:annel)?|acc)?[_-]?(\d{1,3})(?:[xyzv])?$", path.stem, re.I)
    sensor = sensor_match.group(1) if sensor_match else path.stem
    return {"scenario": scenario, "setup": setup, "sensor": sensor, "recording": path.stem, "label": "unknown"}


def _records_from_array(array: np.ndarray, metadata: dict, source: str, sampling_rate: float | None, variable: str) -> list[SignalRecord]:
    array = np.asarray(array, dtype=float).squeeze()
    if array.ndim == 0:
        return []
    if array.ndim == 1:
        channels = [array]
    elif array.ndim == 2:
        oriented = array if array.shape[0] >= array.shape[1] else array.T
        channels = [oriented[:, i] for i in range(oriented.shape[1])]
    else:
        return []
    records = []
    for i, sig in enumerate(channels):
        fields = dict(metadata)
        fields["sensor"] = metadata["sensor"] if len(channels) == 1 else f"{metadata['sensor']}_{i}"
        records.append(SignalRecord(**fields, signal=np.asarray(sig), sampling_rate=sampling_rate, source_file=source,
                                    channel=f"{variable}:{i}", metadata={"variable": variable}))
    return records


def load_file(path: str | Path, dataset_root: str | Path | None = None, fallback_sampling_rate: float | None = None) -> list[SignalRecord]:
    path = Path(path)
    metadata = infer_metadata(path, dataset_root)
    if path.suffix.lower() in {".aaa", ".csv", ".txt"}:
        data, info = load_aaa_file(path)
        rate = 1.0 / info["sample_interval"] if info.get("sample_interval") else fallback_sampling_rate
        return _records_from_array(data, metadata, str(path), rate, "ascii")
    if path.suffix.lower() == ".mat":
        records = []
        for variable, array in load_mat_file(path).items():
            if array.size >= 8:
                records.extend(_records_from_array(array, metadata, str(path), fallback_sampling_rate, variable))
        return records
    raise ValueError(f"Unsupported signal format: {path.suffix}")


def load_archive(path: str | Path, dataset_root: str | Path | None = None, fallback_sampling_rate: float | None = None) -> list[SignalRecord]:
    path = Path(path)
    records = []
    with zipfile.ZipFile(path) as archive:
        for member in archive.infolist():
            suffix = Path(member.filename).suffix.lower()
            if suffix not in {".aaa", ".csv", ".txt"} or member.is_dir():
                continue
            data, info = load_aaa_bytes(archive.read(member), f"{path}!{member.filename}")
            virtual = Path(path.parent, member.filename)
            meta = infer_metadata(virtual, dataset_root)
            rate = 1.0 / info["sample_interval"] if info.get("sample_interval") else fallback_sampling_rate
            records.extend(_records_from_array(data, meta, f"{path}!{member.filename}", rate, "ascii"))
    return records


def load_dataset(root: str | Path, max_files: int | None = None, include_archives: bool = False, sampling_rate: float | None = None,
                 one_file_per_scenario: bool = False) -> list[SignalRecord]:
    paths = discover_files(root, include_archives=include_archives)
    if one_file_per_scenario:
        selected, seen = [], set()
        for path in paths:
            scenario = infer_metadata(path, root)["scenario"]
            if scenario != "unknown" and scenario not in seen:
                selected.append(path); seen.add(scenario)
        paths = selected
    if max_files:
        paths = paths[:max_files]
    records: list[SignalRecord] = []
    errors = []
    for path in paths:
        try:
            records.extend(load_archive(path, root, sampling_rate) if path.suffix.lower() == ".zip" else load_file(path, root, sampling_rate))
        except Exception as exc:
            errors.append(f"{path}: {exc}")
    if not records:
        detail = "\n".join(errors[:10])
        raise ValueError(f"No usable signals were loaded from {root}.\n{detail}")
    return records


def inspect_file(path: str | Path) -> dict:
    path = Path(path)
    arrays = load_mat_file(path) if path.suffix.lower() == ".mat" else {"ascii": load_aaa_file(path)[0]}
    variables = []
    for name, raw in arrays.items():
        arr = np.asarray(raw)
        finite = arr[np.isfinite(arr)] if np.issubdtype(arr.dtype, np.number) else np.array([])
        variables.append({"name": name, "shape": list(arr.shape), "dtype": str(arr.dtype), "min": float(finite.min()) if finite.size else None,
                          "max": float(finite.max()) if finite.size else None, "mean": float(finite.mean()) if finite.size else None,
                          "std": float(finite.std()) if finite.size else None, "nan_count": int(np.isnan(arr).sum()), "inf_count": int(np.isinf(arr).sum())})
    return {"path": str(path), "name": path.name, "extension": path.suffix.lower(), "parent": path.parent.name,
            "size_bytes": path.stat().st_size, "variables": variables}
