from __future__ import annotations

from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import yaml

from src.select_files import discover_selected_files, load_selection_config, parse_filename_timestamp


class TestFilenameTimestamp(unittest.TestCase):
    def test_parse_filename_timestamp_uses_last_eleven_digits(self) -> None:
        timestamp = parse_filename_timestamp("corrected_mi0226065010914.parquet")
        self.assertEqual(timestamp, datetime(2026, 3, 6, 1, 9, 14))

    def test_parse_filename_timestamp_rejects_malformed_names(self) -> None:
        with self.assertRaises(ValueError):
            parse_filename_timestamp("not_a_timestamp.parquet")

    def test_discover_selected_files_filters_date_range(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_dir = root / "STATIONS" / "MINGO02" / "STAGE_1" / "EVENT_DATA" / "STEP_2" / "INPUT_FILES" / "COMPLETED"
            input_dir.mkdir(parents=True)

            selected_name = "corrected_mi0226065010914.parquet"
            outside_name = "corrected_mi0226066010914.parquet"
            invalid_name = "not_a_timestamp.parquet"
            for name in (selected_name, outside_name, invalid_name):
                (input_dir / name).write_text("", encoding="utf-8")

            config_path = root / "selection.yaml"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "selection": {
                            "station_base_dir": str(root / "STATIONS"),
                            "stations": ["MINGO02"],
                            "input_subdir": "STAGE_1/EVENT_DATA/STEP_2/INPUT_FILES/COMPLETED",
                            "start_datetime": "2026-03-06T00:00:00",
                            "end_datetime": "2026-03-06T23:59:59",
                            "output_dir": str(root / "outputs"),
                        }
                    }
                ),
                encoding="utf-8",
            )

            selection = load_selection_config(config_path)
            result = discover_selected_files(selection)

            self.assertEqual([item.path.name for item in result.selected_files], [selected_name])
            self.assertEqual([path.name for path in result.invalid_files], [invalid_name])

    def test_load_selection_config_accepts_plotting_rate_mode(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config_path = root / "selection.yaml"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "selection": {
                            "station_base_dir": str(root / "STATIONS"),
                            "stations": ["MINGO02"],
                            "input_subdir": "STAGE_1/EVENT_DATA/STEP_2/INPUT_FILES/COMPLETED",
                            "start_datetime": "2026-03-06T00:00:00",
                            "end_datetime": "2026-03-06T23:59:59",
                            "output_dir": str(root / "outputs"),
                        },
                        "plotting": {
                            "enabled": True,
                            "rate_mode": "zscores",
                        },
                    }
                ),
                encoding="utf-8",
            )

            selection = load_selection_config(config_path)

            self.assertEqual(selection.plotting.rate_mode, "zscores")

    def test_load_selection_config_rejects_unknown_plotting_rate_mode(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config_path = root / "selection.yaml"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "selection": {
                            "station_base_dir": str(root / "STATIONS"),
                            "stations": ["MINGO02"],
                            "input_subdir": "STAGE_1/EVENT_DATA/STEP_2/INPUT_FILES/COMPLETED",
                            "start_datetime": "2026-03-06T00:00:00",
                            "end_datetime": "2026-03-06T23:59:59",
                            "output_dir": str(root / "outputs"),
                        },
                        "plotting": {
                            "rate_mode": "not-a-mode",
                        },
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "plotting.rate_mode"):
                load_selection_config(config_path)
