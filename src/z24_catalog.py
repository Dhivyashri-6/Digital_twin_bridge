"""Catalogue and loader for the Z24 progressive damage test (PDT) data.

Every recording is identified by scenario, test type (AVT/FVT) and setup, e.g.
``S06_avt_setup01``.  The same recording can be stored more than once in the raw
data (``06setup01.mat`` and the per-channel ASCII files in ``06/avt/01/*.aaa``, or
the ``avt.zip`` archive next to the extracted ``avt`` folder of scenario 08).
All sources are read, duplicates are verified sample by sample and counted, and
each physical channel enters the analysis exactly once.  Raw files are only read.
"""
from __future__ import annotations

import hashlib
import io
import re
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.io import loadmat

MAT_RE = re.compile(r"^(\d{2})setup(\d{2})\.mat$", re.I)
REFERENCE_SENSORS = ("R1V", "R2L", "R2T", "R2V", "R3V")


@dataclass
class Z24Recording:
    scenario: int
    test: str  # "avt" or "fvt"
    setup: int
    mat_path: Path | None = None
    zip_member: tuple[Path, str] | None = None
    aaa_paths: dict[str, Path] = field(default_factory=dict)

    @property
    def recording_id(self) -> str:
        return f"S{self.scenario:02d}_{self.test}_setup{self.setup:02d}"

    @property
    def file_stem(self) -> str:
        return f"{self.scenario:02d}setup{self.setup:02d}"


@dataclass
class LoadedRecording:
    recording: Z24Recording
    sensors: list[str]
    data: np.ndarray  # (n_samples, n_channels)
    sampling_rate: float
    sources: list[str]
    provenance: dict


def _sensor_from_label(label: str) -> str:
    return str(label).strip().upper()


def build_catalog(root: str | Path) -> tuple[list[Z24Recording], dict]:
    """Index every PDT recording under ``root`` (pdt_* folders)."""
    root = Path(root)
    recs: dict[tuple[int, str, int], Z24Recording] = {}
    counts: Counter = Counter()

    def get(sc: int, test: str, setup: int) -> Z24Recording:
        key = (sc, test, setup)
        if key not in recs:
            recs[key] = Z24Recording(sc, test, setup)
        return recs[key]

    for pdt in sorted(root.glob("pdt_*")):
        for path in sorted(pdt.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(pdt).parts
            suffix = path.suffix.lower()
            if suffix == ".mat":
                m = MAT_RE.match(path.name)
                if not m or len(rel) < 3:
                    continue
                counts["mat_files"] += 1
                sc, setup = int(m.group(1)), int(m.group(2))
                test = rel[1].lower()
                get(sc, test, setup).mat_path = path
            elif suffix == ".aaa":
                # <scenario>/<test>/<setup>/<sensor>.aaa
                if len(rel) != 4 or not rel[2].isdigit():
                    continue
                counts["aaa_files"] += 1
                get(int(rel[0]), rel[1].lower(), int(rel[2])).aaa_paths[_sensor_from_label(path.stem)] = path
            elif suffix == ".zip":
                counts["zip_files"] += 1
                with zipfile.ZipFile(path) as archive:
                    for member in archive.namelist():
                        m = MAT_RE.match(Path(member).name)
                        if not m:
                            continue
                        counts["zip_mat_members"] += 1
                        sc, setup = int(m.group(1)), int(m.group(2))
                        test = Path(member).parts[0].lower() if len(Path(member).parts) > 1 else path.stem.lower()
                        get(sc, test, setup).zip_member = (path, member)
    ordered = [recs[k] for k in sorted(recs)]
    counts["recordings"] = len(ordered)
    return ordered, dict(counts)


def _sha1(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


def read_aaa(path: Path) -> tuple[np.ndarray, float, int]:
    """Parse a Z24 ASCII file: ``name|8Averages``, sample count, sample interval, samples."""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    n = int(float(lines[1]))
    dt = float(lines[2])
    values = np.array(lines[3:3 + n], dtype=float)
    if values.size != n:
        raise ValueError(f"{path}: header declares {n} samples, found {values.size}")
    return values, dt, n


def _read_mat_bytes(content: bytes | Path) -> tuple[np.ndarray, list[str]]:
    source = io.BytesIO(content) if isinstance(content, bytes) else content
    m = loadmat(source, squeeze_me=True)
    labels = [_sensor_from_label(s) for s in np.atleast_1d(m["labelshulp"])]
    data = np.asarray(m["data"], dtype=float)
    if data.ndim != 2 or data.shape[1] != len(labels):
        raise ValueError(f"Unexpected MAT layout: data {data.shape}, {len(labels)} labels")
    return data, labels


def load_recording(rec: Z24Recording, sampling_rate: float = 100.0, verify_duplicates: bool = True) -> LoadedRecording:
    """Load one recording once, cross-checking every duplicate copy that exists."""
    prov: Counter = Counter()
    sources: list[str] = []
    data = labels = None
    if rec.mat_path is not None:
        data, labels = _read_mat_bytes(rec.mat_path)
        sources.append(str(rec.mat_path))
        prov["channels_from_mat"] = len(labels)
        if rec.zip_member is not None:
            zpath, member = rec.zip_member
            with zipfile.ZipFile(zpath) as archive:
                same = _sha1(archive.read(member)) == _sha1(rec.mat_path.read_bytes())
            prov["zip_duplicate_identical" if same else "zip_duplicate_different"] += 1
    elif rec.zip_member is not None:
        zpath, member = rec.zip_member
        with zipfile.ZipFile(zpath) as archive:
            data, labels = _read_mat_bytes(archive.read(member))
        sources.append(f"{zpath}!{member}")
        prov["channels_from_zip"] = len(labels)

    columns = {} if data is None else {lab: data[:, i] for i, lab in enumerate(labels)}
    aaa_dt = []
    for sensor, path in sorted(rec.aaa_paths.items()):
        values, dt, _ = read_aaa(path)
        aaa_dt.append(dt)
        if sensor in columns:
            prov["aaa_channels_checked"] += 1
            if verify_duplicates:
                ref = columns[sensor]
                n = min(ref.size, values.size)
                tol = 1e-6 * max(np.max(np.abs(ref[:n])), 1e-12)
                if ref.size == values.size and np.allclose(ref, values, rtol=1e-5, atol=tol):
                    prov["aaa_duplicate_identical"] += 1
                else:
                    prov["aaa_duplicate_different"] += 1
        else:
            columns[sensor] = values
            prov["channels_from_aaa"] += 1
            sources.append(str(path))
    if not columns:
        raise ValueError(f"No channels found for {rec.recording_id}")
    lengths = Counter(v.size for v in columns.values())
    n = min(lengths)
    sensors = list(columns)
    matrix = np.column_stack([columns[s][:n] for s in sensors])
    rate = float(sampling_rate)
    if aaa_dt:
        dts = set(np.round(aaa_dt, 9))
        if len(dts) != 1 or abs(1.0 / dts.pop() - rate) > 1e-6:
            raise ValueError(f"{rec.recording_id}: ASCII header sample interval disagrees with sampling_rate={rate}")
        prov["aaa_header_rate_hz"] = 1.0 / aaa_dt[0]
    return LoadedRecording(rec, sensors, matrix, rate, sources, dict(prov))


def verify_aaa_headers(recordings: list[Z24Recording]) -> dict:
    """Read the header of every .aaa file and summarise sample count and interval."""
    intervals: Counter = Counter()
    lengths: Counter = Counter()
    for rec in recordings:
        for path in rec.aaa_paths.values():
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                handle.readline()
                n = int(float(handle.readline()))
                dt = float(handle.readline())
            intervals[dt] += 1
            lengths[n] += 1
    return {"files": sum(intervals.values()), "sample_interval_s": {str(k): v for k, v in intervals.items()},
            "declared_samples": {str(k): v for k, v in lengths.items()},
            "sampling_rate_hz": {str(round(1 / k, 6)): v for k, v in intervals.items()}}


def scenario_label(scenario: int, reference: tuple[int, ...] = (1, 2, 8)) -> str:
    return "REFERENCE" if scenario in reference else "ALTERED_CONDITION"


def sensor_location(sensor: str) -> dict:
    """Split a Z24 channel name (e.g. ``241T``, ``R2V``, ``DP1V``) into point and direction."""
    s = sensor.upper()
    direction = {"V": "vertical", "L": "longitudinal", "T": "transversal"}.get(s[-1:], "unknown")
    point = s[:-1] if s[-1:] in "VLT" else s
    if point.startswith("R"):
        group = "reference"
    elif point.startswith("DP"):
        group = "drive_point"
    elif point.isdigit():
        group = f"line_{point[0]}xx" if len(point) == 3 else f"line_{len(point)}d"
    else:
        group = "other"
    return {"point": point, "direction": direction, "group": group}
