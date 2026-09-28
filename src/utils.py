from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path else PROJECT_ROOT / "config.yaml"
    if not config_path.is_absolute():
        config_path = (PROJECT_ROOT / config_path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def ensure_output_dirs(config: dict[str, Any]) -> dict[str, Path]:
    paths = {k: resolve_project_path(v) for k, v in config.get("paths", {}).items()}
    paths.setdefault("processed", PROJECT_ROOT / "data" / "processed")
    paths.setdefault("models", PROJECT_ROOT / "models")
    paths.setdefault("results", PROJECT_ROOT / "results")
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    for child in ("figures", "figures/dataset_inspection", "metrics", "shap", "reports"):
        (paths["results"] / child).mkdir(parents=True, exist_ok=True)
    return paths


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=json_default), encoding="utf-8")
