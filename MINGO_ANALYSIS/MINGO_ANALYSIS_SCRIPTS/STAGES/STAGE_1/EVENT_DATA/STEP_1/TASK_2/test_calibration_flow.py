from pathlib import Path


SCRIPT_PATH = Path(__file__).with_name("script_2_clean_to_cal.py")


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
