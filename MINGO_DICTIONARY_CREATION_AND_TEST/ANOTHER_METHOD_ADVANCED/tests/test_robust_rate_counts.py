from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def common_modules():
    return {
        "another_method": _load_module(
            "another_method_common",
            REPO_ROOT / "MINGO_DICTIONARY_CREATION_AND_TEST" / "ANOTHER_METHOD" / "common.py",
        ),
        "even_easier": _load_module(
            "even_easier_common",
            REPO_ROOT / "MINGO_DICTIONARY_CREATION_AND_TEST" / "AN_EVEN_EASIER_VARIATION" / "common.py",
        ),
    }


@pytest.mark.parametrize("module_key", ["another_method", "even_easier"])
def test_robust_rate_counts_reconstruct_from_denominator(common_modules, module_key: str) -> None:
    common = common_modules[module_key]
    dataframe = pd.DataFrame(
        {
            "eff1": [0.91],
            "eff2": [0.92],
            "eff3": [0.93],
            "eff4": [0.94],
            "rate_total_hz": [3.0],
            "rate_1234_hz": [2.0],
            "four_plane_robust_hz": [1.5],
            "count_rate_denominator_seconds": [20],
        }
    )
    config = {
        "trigger_type_selection": {
            "metadata_source": "robust_efficiency",
            "rate_family": "four_plane_robust_hz",
        }
    }

    derived, _ = common.derive_trigger_rate_features(dataframe, config)

    assert derived.loc[0, "four_plane_count"] == pytest.approx(40.0)
    assert derived.loc[0, "four_plane_robust_count"] == pytest.approx(30.0)
    assert derived.loc[0, "total_count"] == pytest.approx(60.0)
    assert derived.loc[0, "selected_rate_count"] == pytest.approx(30.0)


@pytest.mark.parametrize("module_key", ["another_method", "even_easier"])
def test_robust_rate_counts_prefer_explicit_count_columns(common_modules, module_key: str) -> None:
    common = common_modules[module_key]
    dataframe = pd.DataFrame(
        {
            "eff1": [0.91],
            "eff2": [0.92],
            "eff3": [0.93],
            "eff4": [0.94],
            "rate_total_hz": [3.0],
            "rate_1234_hz": [2.0],
            "four_plane_robust_hz": [1.5],
            "count_rate_denominator_seconds": [20],
            "four_plane_count": [41],
            "four_plane_robust_count": [29],
            "total_count": [61],
        }
    )
    config = {
        "trigger_type_selection": {
            "metadata_source": "robust_efficiency",
            "rate_family": "four_plane_robust_hz",
        }
    }

    derived, _ = common.derive_trigger_rate_features(dataframe, config)

    assert derived.loc[0, "four_plane_count"] == pytest.approx(41.0)
    assert derived.loc[0, "four_plane_robust_count"] == pytest.approx(29.0)
    assert derived.loc[0, "total_count"] == pytest.approx(61.0)
    assert derived.loc[0, "selected_rate_count"] == pytest.approx(29.0)
