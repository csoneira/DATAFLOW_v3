from __future__ import annotations

from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from MASTER.ANCILLARY.QUALITY_ASSURANCE.qa_core.column_rule_table import (  # noqa: E402
    load_column_rule_table,
    matches_any_pattern,
    resolve_column_rule,
    rule_to_threshold_mapping,
)


class ColumnRuleTableTests(unittest.TestCase):
    def test_resolve_column_rule_uses_default_then_specific(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "rules.csv"
            csv_path.write_text(
                "\n".join(
                    [
                        "pattern,figure,plot_enabled,quality_enabled,center_method,tolerance_mode,tolerance_value,min_samples",
                        "*,unexpected,true,false,median,relative_pct,0.10,8",
                        "*_Q_B,q_b,true,true,,,0.15,12",
                    ]
                ),
                encoding="utf-8",
            )
            rules = load_column_rule_table(csv_path)
            q_b_rule = resolve_column_rule("P1_s1_Q_B", rules)
            default_rule = resolve_column_rule("some_new_column", rules)

            self.assertIsNotNone(q_b_rule)
            self.assertEqual(q_b_rule.figure, "q_b")
            self.assertTrue(q_b_rule.quality_enabled)
            self.assertEqual(q_b_rule.center_method, "median")
            self.assertEqual(q_b_rule.tolerance_mode, "relative_pct")
            self.assertEqual(q_b_rule.tolerance_value, 0.15)
            self.assertEqual(q_b_rule.min_samples, 12)

            self.assertIsNotNone(default_rule)
            self.assertEqual(default_rule.figure, "unexpected")
            self.assertFalse(default_rule.quality_enabled)

    def test_rule_to_threshold_mapping_omits_empty_fields(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "rules.csv"
            csv_path.write_text(
                "\n".join(
                    [
                        "pattern,figure,plot_enabled,quality_enabled,center_method,tolerance_mode,tolerance_value,min_samples",
                        "*,unexpected,true,false,median,relative_pct,0.10,8",
                    ]
                ),
                encoding="utf-8",
            )
            rules = load_column_rule_table(csv_path)
            rule = resolve_column_rule("P1_s1_Q_B", rules)
            mapping = rule_to_threshold_mapping(rule)
            self.assertEqual(mapping["center_method"], "median")
            self.assertEqual(mapping["tolerance_mode"], "relative_pct")
            self.assertEqual(mapping["tolerance_value"], 0.10)
            self.assertEqual(mapping["min_samples"], 8)

    def test_matches_any_pattern_supports_globs(self) -> None:
        self.assertTrue(matches_any_pattern("param_hash", ["param_hash", "qa_*"]))
        self.assertTrue(matches_any_pattern("qa_status", ["param_hash", "qa_*"]))
        self.assertFalse(matches_any_pattern("P1_s1_Q_B", ["param_hash", "qa_*"]))


if __name__ == "__main__":
    unittest.main()
