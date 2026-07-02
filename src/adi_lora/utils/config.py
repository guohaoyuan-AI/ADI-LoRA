"""YAML config loading with minimal CLI override support."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

import yaml


def deep_update(base: dict, update: Mapping[str, Any]) -> dict:
    out = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_yaml(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def load_config(path: str | Path) -> dict:
    cfg = load_yaml(path)
    # Optional inheritance: base: path/to/base.yaml
    base_path = cfg.pop("base", None)
    if base_path:
        base_path = Path(path).parent / base_path if not Path(base_path).is_absolute() else Path(base_path)
        return deep_update(load_config(base_path), cfg)
    return cfg


def get_by_path(cfg: Mapping[str, Any], dotted: str, default: Any = None) -> Any:
    cur: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            return default
        cur = cur[part]
    return cur
