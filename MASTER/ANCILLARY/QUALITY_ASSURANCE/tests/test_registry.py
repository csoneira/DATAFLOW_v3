from __future__ import annotations

from pathlib import Path
import sys
import unittest

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from MASTER.ANCILLARY.QUALITY_ASSURANCE.qa_core.registry import (  # noqa: E402
    discover_station_metadata,
    get_metadata_family,
    metadata_family_names,
    parse_metadata_filename,
)


class RegistryTests(unittest.TestCase):
    def test_metadata_family_lookup_by_name_and_suffix(self) -> None:
        filtering = get_metadata_family("filtering")
        self.assertEqual(filtering.metadata_suffix, "filter")
        self.assertEqual(get_metadata_family("filter"), filtering)

    def test_parse_metadata_filename(self) -> None:
        task_id, suffix = parse_metadata_filename("task_2_metadata_trigger_type.csv")
        self.assertEqual(task_id, 2)
        self.assertEqual(suffix, "trigger_type")

    def test_metadata_family_names_include_expected_analysis_families(self) -> None:
        families = metadata_family_names(category="analysis")
        self.assertIn("calibration", families)
        self.assertIn("filtering", families)
        self.assertIn("trigger_types", families)

    def test_discover_station_metadata_uses_registry(self) -> None:
        discovered = discover_station_metadata(REPO_ROOT, 3, task_ids=[2])
        suffixes = {item.metadata_suffix for item in discovered}
        self.assertIn("calibration", suffixes)
        self.assertIn("filter", suffixes)
        self.assertIn("trigger_type", suffixes)
        self.assertTrue(all(item.registered for item in discovered if item.metadata_suffix in suffixes))


if __name__ == "__main__":
    unittest.main()
