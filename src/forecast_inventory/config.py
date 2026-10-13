"""Configuration loading (TOML, standard library only)."""

from __future__ import annotations

import copy
import tomllib
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.toml"


def _deep_update(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> dict:
    """Load the default config, optionally layered with a user file and a dict of overrides."""
    with open(DEFAULT_CONFIG, "rb") as fh:
        cfg = tomllib.load(fh)
    if path is not None and Path(path).resolve() != DEFAULT_CONFIG:
        with open(path, "rb") as fh:
            cfg = _deep_update(cfg, tomllib.load(fh))
    if overrides:
        cfg = _deep_update(cfg, overrides)
    return cfg


def resolve(path: str | Path) -> Path:
    """Resolve a config path relative to the project root."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p
