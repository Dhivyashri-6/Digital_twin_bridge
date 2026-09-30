"""Run bookkeeping: run ID, config snapshot, environment, seeds, runtime and output hashes.

Every experiment stage calls :func:`start_run` and :func:`finish_run`; the manifest is
written to ``results/runs/<run_id>/``.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .utils import PROJECT_ROOT, json_default

PACKAGES = ["numpy", "scipy", "pandas", "sklearn", "xgboost", "shap", "matplotlib", "yaml", "joblib", "pyarrow",
            "docx", "streamlit", "plotly"]


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def package_versions() -> dict[str, str]:
    out = {}
    for name in PACKAGES:
        try:
            out[name] = str(getattr(importlib.import_module(name), "__version__", "unknown"))
        except Exception:
            out[name] = "not installed"
    return out


def environment() -> dict:
    return {"python": sys.version, "platform": platform.platform(), "processor": platform.processor(),
            "cpu_count": os.cpu_count(), "packages": package_versions()}


def _peak_memory_mb() -> float | None:
    try:
        import psutil
        info = psutil.Process().memory_info()
        return round(getattr(info, "peak_wset", info.rss) / 2 ** 20, 1)
    except Exception:
        return None


class Run:
    def __init__(self, stage: str, config: dict, seeds: list[int] | dict | None = None, root: Path | None = None):
        config_text = yaml.safe_dump(config, sort_keys=True)
        config_hash = hashlib.sha256(config_text.encode()).hexdigest()[:8]
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.run_id = f"{stamp}_{stage}_{config_hash}"
        self.dir = (root or PROJECT_ROOT / "results" / "runs") / self.run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "config_snapshot.yaml").write_text(config_text, encoding="utf-8")
        self.stage, self.seeds, self.config_hash = stage, seeds, config_hash
        self.t0, self.cpu0 = time.perf_counter(), time.process_time()
        self.started = datetime.now(timezone.utc).isoformat()
        self.extra: dict = {}

    def finish(self, outputs: list[str | Path], extra: dict | None = None) -> dict:
        hashes = {}
        for p in outputs:
            p = Path(p)
            if p.is_file():
                rel = str(p.resolve().relative_to(PROJECT_ROOT)) if p.resolve().is_relative_to(PROJECT_ROOT) else str(p)
                hashes[rel] = {"sha256": sha256_file(p), "bytes": p.stat().st_size}
        manifest = {"run_id": self.run_id, "stage": self.stage, "started_utc": self.started,
                    "finished_utc": datetime.now(timezone.utc).isoformat(), "command": " ".join(sys.argv),
                    "wall_time_s": round(time.perf_counter() - self.t0, 2),
                    "main_process_cpu_time_s": round(time.process_time() - self.cpu0, 2),
                    "peak_memory_mb_main_process": _peak_memory_mb(), "config_hash": self.config_hash,
                    "seeds": self.seeds, "environment": environment(), "outputs": hashes, **self.extra, **(extra or {})}
        (self.dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=json_default), encoding="utf-8")
        latest = self.dir.parent / f"LATEST_{self.stage}.txt"
        latest.write_text(self.run_id, encoding="utf-8")
        return manifest


def start_run(stage: str, config: dict, seeds=None) -> Run:
    return Run(stage, config, seeds)
