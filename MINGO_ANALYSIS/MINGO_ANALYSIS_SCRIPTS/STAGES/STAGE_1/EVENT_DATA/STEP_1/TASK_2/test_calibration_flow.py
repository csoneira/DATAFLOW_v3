import ast
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml


SCRIPT_PATH = Path(__file__).with_name("script_2_clean_to_cal.py")
CALIBRATION_CONFIG_PATH = (
    Path(__file__).parents[5]
    / "CONFIG_FILES/STAGE_1/EVENT_DATA/STEP_1/TASK_2/config_calibration_subsample_task_2.yaml"
)


def _source() -> str:
    return SCRIPT_PATH.read_text(encoding="utf-8")


def test_canonical_calibration_dataframe_names() -> None:
    source = _source()
    assert "calibration_df = working_df.copy()" in source
    assert "qfb_aux_fit_df" not in source
    assert "calibration_work_df" not in source
    assert "calibration_work_df_qfb" not in source


def test_both_slewing_modes_apply_after_fitting() -> None:
    source = _source()
    polynomial_fit = source.index("per_channel_slewing_model, per_channel_slewing_diagnostics =")
    polynomial_calibration_apply = source.index(
        "apply_per_channel_polynomial_slewing_to_dataframe(\n"
        "                calibration_df,",
        polynomial_fit,
    )
    working_df_apply = source.index(
        "apply_per_channel_polynomial_slewing_to_dataframe(\n"
        "                working_df,",
        polynomial_calibration_apply,
    )
    assert polynomial_fit < polynomial_calibration_apply < working_df_apply
    assert '_task2_slewing_application_mode()' in source
    assert '"pair_event_solution", "per_channel_polynomial"' in source


def test_real_post_slewing_filter_is_wired_to_canonical_sample() -> None:
    source = _source()
    helper = source.index("def filter_task2_calibration_after_slewing(")
    filter_call = source.index('filter_strip_family_inplace(\n        calibration_df,\n        "T_sum"', helper)
    regularization_call = source.index('run_strip_zeroing_stage(\n        "post_slewing_t_sum"', filter_call)
    pipeline_call = source.index("filter_task2_calibration_after_slewing(\n            calibration_df,")
    assert helper < filter_call < regularization_call < pipeline_call
    assert 'filter_mode="real"' not in source[pipeline_call:pipeline_call + 500]
    assert '_slewing_filter_mode = "real"' in source[pipeline_call:pipeline_call + 500]


def test_qfb_and_qtdif_use_only_canonical_calibration_sample() -> None:
    source = _source()
    assert "qfb_aux_fit_df" not in source
    assert "Q_sum_fit = Q_sum_adj" in source
    assert "T_dif_fit = T_dif_adj" in source


def test_tsum_pair_residual_builder_uses_canonical_lowercase_components() -> None:
    source = _source()
    helper_start = source.index("def _build_tsum_pair_residuals(")
    helper_end = source.index("def _enrich_tsum_pair_residuals(", helper_start)
    helper_source = source[helper_start:helper_end]
    assert 'for family in ("qsum", "tsum", "tdif")' in helper_source
    assert 'if "T_sum" in c' not in helper_source
    assert 'if "Q_sum" in c' not in helper_source
    assert 'if "T_dif" in c' not in helper_source


def test_calibration_hard_filters_use_dedicated_config() -> None:
    source = _source()
    assert "config_calibration_subsample_task_2.yaml" in source
    for key in (
        "trigger_type_column",
        "allowed_trigger_types",
        "allowed_cluster_sizes_present_planes",
        "minimum_detector_charge",
        "minimum_plane_charge",
        "minimum_strip_charge",
        "initial_raw_tsum_left",
        "post_tdif_abs_max",
        "post_charge_side_qsum_left",
        "post_qfb_qtdif_qdif_abs_max",
        "post_tsum_left",
        "post_slewing_tsum_left",
    ):
        assert f'calibration_subsample_config["{key}"]' in source


def test_stage_logging_and_filter_switch_are_present() -> None:
    source = _source()
    assert "calibration_dataframe_filtering" in source
    for stage_name in (
        "initial_selection",
        "t_dif",
        "charge_side_q_sum",
        "q_fb_q_tdif",
        "post_charge_q_sum",
        "t_sum",
        "slewing",
    ):
        assert f'"{stage_name}"' in source


def _selection_function_namespace() -> dict[str, object]:
    selected_names = {
        "_task2_active_strip_matrix_from_raw",
        "_task2_plane_strip_qsum_from_raw",
        "build_task2_topology_charge_mask_from_raw",
    }
    tree = ast.parse(_source())
    selected_nodes = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in selected_names
    ]
    namespace: dict[str, object] = {"np": np, "pd": pd, "Iterable": Iterable}
    exec(compile(ast.Module(body=selected_nodes, type_ignores=[]), str(SCRIPT_PATH), "exec"), namespace)
    return namespace


def test_calibration_plane_requires_contiguous_nonzero_charge_strips() -> None:
    rows = 3
    data: dict[str, np.ndarray] = {}
    plane_patterns = {
        1: ([100.0, 100.0, 0.0, 0.0], [100.0, 0.0, 100.0, 0.0], [-1.0, 0.0, 0.0, 0.0]),
        2: ([100.0, 100.0, 0.0, 0.0], [100.0, 0.0, 100.0, 0.0], [-1.0, 0.0, 0.0, 0.0]),
        3: ([100.0, 100.0, 0.0, 0.0], [100.0, 0.0, 100.0, 0.0], [-1.0, 0.0, 0.0, 0.0]),
        4: ([0.0, 0.0, 0.0, 0.0],) * rows,
    }
    for plane in range(1, 5):
        for strip in range(1, 5):
            q_values = np.asarray([plane_patterns[plane][row][strip - 1] for row in range(rows)])
            data[f"p{plane}_s{strip}_ef_q"] = q_values
            data[f"p{plane}_s{strip}_eb_q"] = q_values
            data[f"p{plane}_s{strip}_ef_t"] = np.ones(rows)
            data[f"p{plane}_s{strip}_eb_t"] = np.ones(rows)

    config = yaml.safe_load(CALIBRATION_CONFIG_PATH.read_text(encoding="utf-8"))
    namespace = _selection_function_namespace()
    mask = namespace["build_task2_topology_charge_mask_from_raw"](
        pd.DataFrame(data),
        required_planes={1, 2, 3},
        missing_planes={4},
        detector_charge_threshold=-np.inf,
        plane_charge_threshold=-np.inf,
        strip_charge_threshold=-np.inf,
        allowed_cluster_sizes_present_planes=(1, 2, 3, 4),
        allowed_active_strip_topologies=tuple(config["allowed_active_strip_topologies"]),
        require_zero_cluster_size_missing_planes=False,
    )
    assert mask.tolist() == [True, False, True]


def test_removed_calibration_controls_and_qsum_guard() -> None:
    source = _source()
    config = yaml.safe_load(CALIBRATION_CONFIG_PATH.read_text(encoding="utf-8"))
    assert config["allowed_active_strip_topologies"] == [
        "1000", "0100", "0010", "0001", "1100",
        "0110", "0011", "1110", "0111", "1111",
    ]
    for removed_key in (
        "charge_share_filter_enabled",
        "charge_share_minimum_fraction",
        "minimum_calibration_rows_warning",
        "minimum_calibration_strip_observations_warning",
        "qfb_fit_require_positive_qsum",
    ):
        assert removed_key not in config
        assert removed_key not in source
    deferred_start = source.index("if defer_charge_calibration_until_post_time:")
    qsum_filter = source.index('filter_strip_family_inplace(\n            calibration_df,\n            "Q_sum"', deferred_start)
    guard = source.rfind("if task2_post_charge_side_qsum_filter_enabled:", deferred_start, qsum_filter)
    assert guard != -1
