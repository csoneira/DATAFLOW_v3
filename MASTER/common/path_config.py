"""
DATAFLOW_v3 Script Header v1
Script: MASTER/common/path_config.py
Purpose: Path config.
Owner: DATAFLOW_v3 contributors
Sign-off: csoneira <csoneira@ucm.es>
Last Updated: 2026-03-02
Runtime: python3
Usage: python3 MASTER/common/path_config.py [options]
Inputs: CLI args, config files, environment variables, and/or upstream files.
Outputs: Files, logs, plots, or stdout/stderr side effects.
Notes: Keep behavior configuration-driven and reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import os
import yaml


_DEFAULT_CONFIG_REL_PATH = Path("CONFIG") / "config_paths.yaml"


@dataclass(frozen=True)
class RepoPaths:
    home_path: Path
    repo_name: str
    repo_root: Path
    master_config_root: Path
    stations_root: Path
    digital_twin_root: Path
    operations_runtime_root: Path


def _discover_repo_root() -> Path:
    # MASTER/common/path_config.py -> DATAFLOW_v3
    return Path(__file__).resolve().parents[2]


def _resolve_config_path(config_path: str | Path | None = None) -> Path:
    raw_path = config_path or os.environ.get("DATAFLOW_PATHS_CONFIG")
    if raw_path:
        resolved = Path(raw_path).expanduser()
        if not resolved.is_absolute():
            resolved = _discover_repo_root() / resolved
        return resolved
    return _discover_repo_root() / _DEFAULT_CONFIG_REL_PATH


def _read_yaml_mapping(config_path: Path) -> Mapping[str, Any]:
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Path config must be a YAML mapping: {config_path}")
    return data


@lru_cache(maxsize=None)
def load_repo_paths(config_path: str | Path | None = None) -> RepoPaths:
    discovered_root = _discover_repo_root()
    config = _read_yaml_mapping(_resolve_config_path(config_path))

    configured_home = config.get("home_path")
    home_path = Path(str(configured_home)).expanduser() if configured_home else discovered_root.parent

    configured_repo_name = str(config.get("repo_name") or "").strip()
    repo_name = configured_repo_name or discovered_root.name

    configured_repo_root = config.get("repo_root")
    if configured_repo_root:
        repo_root = Path(str(configured_repo_root)).expanduser()
    elif configured_home or configured_repo_name:
        repo_root = home_path / repo_name
    else:
        repo_root = discovered_root

    return RepoPaths(
        home_path=home_path,
        repo_name=repo_name,
        repo_root=repo_root,
        master_config_root=repo_root / "MASTER" / "CONFIG_FILES",
        stations_root=repo_root / "STATIONS",
        digital_twin_root=repo_root / "MINGO_DIGITAL_TWIN",
        operations_runtime_root=repo_root / "OPERATIONS_RUNTIME",
    )


def get_home_path(config_path: str | Path | None = None) -> Path:
    return load_repo_paths(config_path).home_path


def get_repo_root(config_path: str | Path | None = None) -> Path:
    return load_repo_paths(config_path).repo_root


def get_master_config_root(config_path: str | Path | None = None) -> Path:
    return load_repo_paths(config_path).master_config_root


def resolve_home_path_from_config(
    config: Mapping[str, Any] | None = None,
    *,
    config_path: str | Path | None = None,
) -> Path:
    if isinstance(config, Mapping):
        value = config.get("home_path")
        if value:
            return Path(str(value)).expanduser()
    return get_home_path(config_path)


def resolve_master_config_root_from_config(
    config: Mapping[str, Any] | None = None,
    *,
    config_path: str | Path | None = None,
) -> Path:
    if isinstance(config, Mapping):
        value = config.get("config_files_directory")
        if value:
            return Path(str(value)).expanduser()
    return get_master_config_root(config_path)
