"""YAML config loading with CLI override merging.

Usage:
    cfg = load_config("config.yaml")
    cfg = apply_overrides(cfg, {"training.lr": 5e-4, "model.encoder": "gcn"})
    val = cfg["training"]["lr"]
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def apply_overrides(cfg: dict, overrides: dict[str, Any]) -> dict:
    """Apply dot-path overrides, e.g. {"training.lr": 1e-4} → cfg["training"]["lr"] = 1e-4."""
    cfg = copy.deepcopy(cfg)
    for key, value in overrides.items():
        parts = key.split(".")
        node = cfg
        for part in parts[:-1]:
            node = node[part]
        node[parts[-1]] = value
    return cfg


def get(cfg: dict, dot_path: str, default: Any = None) -> Any:
    """Safe dot-path accessor, e.g. get(cfg, "model.hidden_dim", 128)."""
    parts = dot_path.split(".")
    node = cfg
    for part in parts:
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node
