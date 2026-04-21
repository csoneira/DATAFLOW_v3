from __future__ import annotations

from pathlib import Path
import sys
import unittest

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from MASTER.ANCILLARY.QUALITY_ASSURANCE.qa_core.epochs import (  # noqa: E402
    attach_epoch_ids,
    load_all_online_run_dictionaries,
    load_online_run_dictionary,
    match_epoch,
    match_epoch_for_run_name,
)


class EpochTests(unittest.TestCase):
    def test_load_online_run_dictionary_normalizes_station_one(self) -> None:
        df = load_online_run_dictionary(REPO_ROOT, 1)

        self.assertFalse(df.empty)
        self.assertTrue({"station_name", "conf_number", "start_timestamp", "end_timestamp", "epoch_id"} <= set(df.columns))
        self.assertTrue(df["station_name"].eq("MINGO01").all())
        self.assertGreaterEqual(df["conf_number"].max(), 15)

    def test_match_epoch_for_station_one_april_2025(self) -> None:
        df = load_online_run_dictionary(REPO_ROOT, 1)

        match = match_epoch(pd.Timestamp("2025-04-01 12:00:00"), df)
        self.assertIsNotNone(match)
        self.assertEqual(int(match["conf_number"]), 15)

    def test_match_epoch_for_run_name(self) -> None:
        df = load_online_run_dictionary(REPO_ROOT, 3)

        match = match_epoch_for_run_name("mi0324359180521", df, allow_nearest=True)
        self.assertIsNotNone(match)
        self.assertEqual(match["station_name"], "MINGO03")

    def test_attach_epoch_ids_adds_epoch_column(self) -> None:
        epochs = load_online_run_dictionary(REPO_ROOT, 1)
        frame = pd.DataFrame({"timestamp": [pd.Timestamp("2025-03-01 00:00:00"), pd.NaT]})

        out = attach_epoch_ids(frame, "timestamp", epochs)
        self.assertTrue(out["epoch_id"].iloc[0].startswith("MINGO01_conf_15"))
        self.assertTrue(pd.isna(out["epoch_id"].iloc[1]))

    def test_load_all_online_run_dictionaries_combines_multiple_stations(self) -> None:
        df = load_all_online_run_dictionaries(REPO_ROOT, stations=[1, 3])
        self.assertTrue({"MINGO01", "MINGO03"} <= set(df["station_name"]))


if __name__ == "__main__":
    unittest.main()
