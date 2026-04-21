from __future__ import annotations

from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

import yaml

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from MASTER.ANCILLARY.QUALITY_ASSURANCE.qa_core.task_setup import load_task_configs  # noqa: E402


class TaskSetupTests(unittest.TestCase):
    def test_load_task_configs_merges_optional_quality_layer(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            qa_root = Path(tmp_dir)
            step_dir = qa_root / "STEP_1_CALIBRATIONS"
            task_dir = step_dir / "TASK_2"
            common_dir = step_dir / "common"
            common_dir.mkdir(parents=True)
            task_dir.mkdir(parents=True)

            (qa_root / "config.yaml").write_text(
                yaml.safe_dump({"stations": [1], "stations_root": "STATIONS"}),
                encoding="utf-8",
            )
            (qa_root / "config_runtime.yaml").write_text("{}", encoding="utf-8")
            (common_dir / "config.yaml").write_text("{}", encoding="utf-8")
            (step_dir / "config.yaml").write_text("{}", encoding="utf-8")
            (task_dir / "config.yaml").write_text(
                yaml.safe_dump({"task_id": 2, "metadata_type": "calibration"}),
                encoding="utf-8",
            )
            (task_dir / "config_quality.yaml").write_text(
                yaml.safe_dump(
                    {
                        "quality": {
                            "enabled": True,
                            "rules": {
                                "defaults": {
                                    "tolerance_mode": "absolute",
                                    "tolerance_value": 2.5,
                                }
                            },
                        }
                    }
                ),
                encoding="utf-8",
            )

            config = load_task_configs(task_dir)
            self.assertTrue(config["quality"]["enabled"])
            self.assertEqual(config["quality"]["rules"]["defaults"]["tolerance_value"], 2.5)


if __name__ == "__main__":
    unittest.main()
