"""Configuration loading and merging utilities."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def merge_configs(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge override into base config."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def save_config(config: dict[str, Any], path: str | Path) -> None:
    """Save config dict to YAML."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def resolve_eval_every(config: dict[str, Any], section: str) -> int:
    """Mid-training validation interval in steps (0 = disabled)."""
    val_cfg = config.get("validation", {})
    section_cfg = config.get(section, {})
    if "eval_every" in section_cfg:
        return max(0, int(section_cfg["eval_every"]))
    if val_cfg.get("enabled", False):
        return int(val_cfg.get("eval_every", 0))
    return 0


def resolve_validation_batches(config: dict[str, Any]) -> int:
    return int(config.get("validation", {}).get("max_batches", 1))


def apply_variant(config: dict[str, Any], variant: str | None) -> dict[str, Any]:
    """Apply a named variant from config['variants'] if present."""
    if not variant:
        return config
    variants = config.get("variants", {})
    if variant not in variants:
        raise ValueError(f"Unknown variant '{variant}'. Available: {list(variants.keys())}")
    return merge_configs(config, variants[variant])
