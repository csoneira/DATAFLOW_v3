"""Shared helpers for the QUALITY_ASSURANCE workspace."""

from .column_rule_table import (
    ColumnRule,
    load_column_rule_table,
    matches_any_pattern,
    resolve_column_rule,
    rule_to_threshold_mapping,
    split_default_rule,
)
from .epochs import (
    attach_epoch_ids,
    load_all_online_run_dictionaries,
    load_online_run_dictionary,
    match_epoch,
    match_epoch_for_run_name,
    online_run_dictionary_path,
)
from .epoch_quality import (
    EvaluationColumnSpec,
    build_epoch_reference_table,
    build_scalar_value_frame,
    evaluate_scalar_frame,
)
from .registry import (
    DiscoveredMetadataFile,
    MetadataFamilySpec,
    discover_station_metadata,
    get_metadata_family,
    metadata_family_names,
    parse_metadata_filename,
)
from .thresholds import (
    ThresholdEvaluation,
    ThresholdRule,
    compute_bounds,
    evaluate_value,
    resolve_threshold_rule,
    select_threshold_rule,
)

__all__ = [
    "ColumnRule",
    "DiscoveredMetadataFile",
    "EvaluationColumnSpec",
    "MetadataFamilySpec",
    "ThresholdEvaluation",
    "ThresholdRule",
    "attach_epoch_ids",
    "build_epoch_reference_table",
    "build_scalar_value_frame",
    "compute_bounds",
    "discover_station_metadata",
    "evaluate_scalar_frame",
    "evaluate_value",
    "get_metadata_family",
    "load_column_rule_table",
    "load_all_online_run_dictionaries",
    "load_online_run_dictionary",
    "matches_any_pattern",
    "match_epoch",
    "match_epoch_for_run_name",
    "metadata_family_names",
    "online_run_dictionary_path",
    "parse_metadata_filename",
    "resolve_column_rule",
    "resolve_threshold_rule",
    "rule_to_threshold_mapping",
    "select_threshold_rule",
    "split_default_rule",
]
