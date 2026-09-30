"""Z24 environmental monitoring system (EMS) archives: hourly acceleration + temperature records.

Each archive ``WWDhh.zip`` holds eight acceleration channels (``WWDhhCC.aaa``, 100 Hz, 65 536
samples), a traffic log and two environmental scans (``PRE``/``POS``) with a timestamp line
``EnvScan started : <date>``.  Values 3276.7 and <= -100 are sensor-fault sentinels.
"""
from __future__ import annotations

import re
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np

SENTINEL_HIGH = 3000.0
SENTINEL_LOW = -100.0


def list_archives(root: str | Path) -> list[Path]:
    root = Path(root)
    return sorted(p for folder in ("Z24ems1", "Z24ems2", "Z24ems3") for p in (root / folder).glob("*.zip"))


def parse_env(text: str) -> dict:
    lines = text.splitlines()
    header = re.findall(r"([A-Z0-9]+)\s*\[[^\]]*\]", lines[0])
    rows = []
    for line in lines[1:]:
        parts = line.split()
        if header and len(parts) >= len(header):  # rows carry unnamed trailing columns
            try:
                rows.append([float(v) for v in parts[: len(header)]])
            except ValueError:
                pass
    stamp = None
    m = re.search(r"EnvScan started\s*:\s*(.+)", text)
    if m:
        try:
            stamp = datetime.strptime(m.group(1).strip(), "%a %b %d %H:%M:%S %Y")
        except ValueError:
            stamp = None
    arr = np.array(rows, dtype=float) if rows else np.empty((0, len(header)))
    arr[(arr >= SENTINEL_HIGH) | (arr <= SENTINEL_LOW)] = np.nan
    with np.errstate(all="ignore"):
        means = np.nanmean(arr, axis=0) if len(arr) else np.full(len(header), np.nan)
    # duplicate names in the header (e.g. TWS3) are disambiguated by position
    names, seen = [], {}
    for h in header:
        seen[h] = seen.get(h, 0) + 1
        names.append(h if seen[h] == 1 else f"{h}_{seen[h]}")
    return {"timestamp": stamp, "values": dict(zip(names, means))}


def read_archive(path: Path) -> dict:
    """Return timestamp, temperatures and the acceleration matrix of one hourly archive."""
    channels, env = {}, []
    with zipfile.ZipFile(path) as z:
        for name in sorted(z.namelist()):
            stem = Path(name).stem
            if name.lower().endswith(".env"):
                env.append(parse_env(z.read(name).decode("latin1")))
            elif name.lower().endswith(".aaa") and re.fullmatch(r"\d{2}[A-Z]\d{2}\d{2}", stem):
                lines = z.read(name).decode("latin1").splitlines()
                n, dt = int(float(lines[1])), float(lines[2])
                vals = np.array(lines[3:3 + n], dtype=float)
                if vals.size == n:
                    channels[stem[-2:]] = (vals, dt)
    stamps = [e["timestamp"] for e in env if e["timestamp"] is not None]
    temps: dict[str, float] = {}
    for key in set().union(*[e["values"] for e in env]) if env else []:
        vals = [e["values"].get(key, np.nan) for e in env]
        temps[key] = float(np.nanmean(vals)) if np.isfinite(vals).any() else float("nan")
    return {"archive": path.stem, "timestamp": min(stamps) if stamps else None, "temperatures": temps,
            "channels": channels}
