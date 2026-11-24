# Copyright © 2026 Intel Corporation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration loading.

`configs/default.toml` is the single source of defaults. A user file only
lists the keys it changes; it is merged over the defaults. Unknown keys and
wrong value types are rejected so a typo never silently falls back to a
default.
"""

from __future__ import annotations

import copy
import re
import tomllib
from pathlib import Path
from typing import Any

APP_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = APP_ROOT / "configs" / "default.toml"


class ConfigError(ValueError):
    pass


def _merge(base: dict, override: dict, path: str = "") -> None:
    for key, value in override.items():
        where = f"{path}{key}"
        if key not in base:
            raise ConfigError(f"unknown configuration key: {where}")
        current = base[key]
        if isinstance(current, dict):
            if not isinstance(value, dict):
                raise ConfigError(f"{where} must be a table")
            _merge(current, value, f"{where}.")
            continue
        if isinstance(current, bool) or isinstance(value, bool):
            ok = isinstance(current, bool) and isinstance(value, bool)
        elif isinstance(current, (int, float)):
            ok = isinstance(value, (int, float))
            if ok and isinstance(current, int) and not isinstance(value, int):
                ok = False
        else:
            ok = isinstance(value, type(current))
        if not ok:
            raise ConfigError(f"{where} must be {type(current).__name__}, got {type(value).__name__}")
        base[key] = float(value) if isinstance(current, float) else value


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> dict:
    """Return the merged configuration as a nested dict."""
    with DEFAULT_CONFIG.open("rb") as fh:
        cfg = tomllib.load(fh)
    if path is not None:
        with Path(path).open("rb") as fh:
            _merge(cfg, tomllib.load(fh))
    if overrides:
        _merge(cfg, copy.deepcopy(overrides))
    _validate(cfg)
    return cfg


def _validate(cfg: dict) -> None:
    run_id = cfg["run"]["run_id"]
    if run_id and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", run_id):
        raise ConfigError("run.run_id may only contain letters, digits, '.', '_' and '-' (max 64 characters)")
    if cfg["run"]["plates"] < 1:
        raise ConfigError("run.plates must be at least 1")
    plate = cfg["plate"]
    for key in ("negative_control_column", "positive_control_column"):
        if not 1 <= plate[key] <= plate["cols"]:
            raise ConfigError(f"plate.{key} must be between 1 and plate.cols")
    if plate["negative_control_column"] == plate["positive_control_column"]:
        raise ConfigError("negative and positive controls must be in different columns")
    if cfg["hit_scoring"]["device"] not in ("CPU", "GPU"):
        raise ConfigError("hit_scoring.device must be 'CPU' or 'GPU'")
    if cfg["hit_scoring"]["batch_size"] < 1:
        raise ConfigError("hit_scoring.batch_size must be at least 1")
    if cfg["ingestion"]["buffer_size"] < 1 or cfg["ingestion"]["stage_queue_plates"] < 1:
        raise ConfigError("ingestion.buffer_size and ingestion.stage_queue_plates must be at least 1")
    if not 1024 <= cfg["dashboard"]["port"] <= 65535:
        raise ConfigError("dashboard.port must be between 1024 and 65535")


def without_faults(cfg: dict) -> dict:
    """Return a copy of `cfg` with every fault scenario switched off."""
    clean = copy.deepcopy(cfg)
    for fault in clean["faults"].values():
        fault["enabled"] = False
    return clean
