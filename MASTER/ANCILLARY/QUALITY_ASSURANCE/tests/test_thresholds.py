from __future__ import annotations

from pathlib import Path
import sys
import unittest

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from MASTER.ANCILLARY.QUALITY_ASSURANCE.qa_core.thresholds import (  # noqa: E402
    ThresholdRule,
    compute_bounds,
    evaluate_value,
    resolve_threshold_rule,
    select_threshold_rule,
)


class ThresholdTests(unittest.TestCase):
    def test_compute_bounds_relative_pct(self) -> None:
        rule = ThresholdRule(tolerance_mode="relative_pct", tolerance_value=0.10)
        lower, upper = compute_bounds(100.0, rule)
        self.assertEqual(lower, 90.0)
        self.assertEqual(upper, 110.0)

    def test_compute_bounds_with_negative_center_keeps_order(self) -> None:
        rule = ThresholdRule(tolerance_mode="relative_pct", tolerance_value=0.10)
        lower, upper = compute_bounds(-100.0, rule)
        self.assertEqual(lower, -110.0)
        self.assertEqual(upper, -90.0)

    def test_compute_bounds_zscore_uses_scale(self) -> None:
        rule = ThresholdRule(center_method="mean", tolerance_mode="zscore", tolerance_value=3.0)
        lower, upper = compute_bounds(10.0, rule, scale=2.0)
        self.assertEqual(lower, 4.0)
        self.assertEqual(upper, 16.0)

    def test_resolve_threshold_rule_allows_overrides(self) -> None:
        rule = resolve_threshold_rule(
            {"tolerance_mode": "relative_pct", "tolerance_value": 0.10, "min_samples": 8},
            {"tolerance_value": 0.25, "min_samples": 3},
        )
        self.assertEqual(rule.tolerance_value, 0.25)
        self.assertEqual(rule.min_samples, 3)

    def test_select_threshold_rule_prefers_exact_before_glob(self) -> None:
        rule = select_threshold_rule(
            "data_purity_percentage",
            defaults={"tolerance_mode": "relative_pct", "tolerance_value": 0.10},
            column_rules={
                "data_*": {"tolerance_mode": "absolute", "tolerance_value": 5.0},
                "data_purity_percentage": {"tolerance_mode": "absolute", "tolerance_value": 2.0},
            },
        )
        self.assertEqual(rule.tolerance_mode, "absolute")
        self.assertEqual(rule.tolerance_value, 2.0)

    def test_evaluate_value_reports_failures(self) -> None:
        rule = ThresholdRule(tolerance_mode="absolute", tolerance_value=2.0)
        result = evaluate_value(15.0, 10.0, rule)
        self.assertEqual(result.status, "fail")
        self.assertEqual(result.reason, "above_upper_bound")

    def test_evaluate_value_zscore_passes_inside_sigma_band(self) -> None:
        rule = ThresholdRule(center_method="mean", tolerance_mode="zscore", tolerance_value=3.0)
        result = evaluate_value(11.5, 10.0, rule, scale=1.0)
        self.assertEqual(result.status, "pass")


if __name__ == "__main__":
    unittest.main()
