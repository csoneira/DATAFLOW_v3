# snapshot_logs_gui.py
"""
Graphical snapshot viewer for miniTRASGO cron logs.

Features
--------
* Four‑column fixed layout, identical to the original terminal layout.
* Colour‑coded job state:
    • Green   – file updated < 5 min ago.
    • Yellow  – updated < 1 h ago.
    • Red     – missing or stale ≥ 1 h.
* Auto‑refresh every 10 s (configurable).
* Station selector (combo box).  Default stations 1‑4.
* Adapts to any number of jobs; constant row height ensures perfect
  alignment regardless of QTextEdit wrapping.

Run
---
$ python3 snapshot_logs_gui.py            # default stations 1‑4
$ python3 snapshot_logs_gui.py --stations 2 3 --refresh 5

Dependencies: PyQt5 (or PySide6 ‑ see comment at bottom).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from datetime import datetime

try:
    from PyQt5.QtCore import QTimer, Qt
    from PyQt5.QtGui import QFont, QTextOption
    from PyQt5.QtWidgets import (
        QApplication,
        QComboBox,
        QFileDialog,
        QGridLayout,
        QLabel,
        QMainWindow,
        QPushButton,
        QTextEdit,
        QWidget,
    )
except ImportError as e:
    print("PyQt5 not found – install with 'pip install pyqt5' or switch to PySide6.")
    raise e

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
HOME = Path(os.environ.get("HOME", "~")).expanduser()
LOG_DIR = HOME / "DATAFLOW_v3/EXECUTION_LOGS/CRON_LOGS"
LINES = 3  # number of tail lines to show
JOBS = [
    "log_bring_reprocessing_files",
    "log_unpack_reprocessing_files",
    "log_bring_and_clean",
    "copernicus",
    "bring_data_and_config_files",
    "raw_to_list_events",
    "ev_accumulator",
    # "merge_large_table",
    # "corrector",
]
UPDATE_INTERVAL_MS = 10_000  # default auto‑refresh (10 s)

# Colour palette (Qt stylesheets)
GREEN = "#8bc34a"
YELLOW = "#ffeb3b"
RED = "#ef5350"
HEADER_BG = "#263238"
HEADER_FG = "#eceff1"


class LogBlock(QTextEdit):
    """Read‑only text area representing one job log."""

    def __init__(self, job: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.job = job
        self.setReadOnly(True)
        self.setWordWrapMode(QTextOption.NoWrap)
        self.setFont(QFont("Courier", 9))
        self.setFixedHeight(self.height_for_lines(LINES + 1))  # header + lines
        self.setStyleSheet("border: 1px solid #37474f; border-radius: 4px;")

    @staticmethod
    def height_for_lines(n: int) -> int:
        """Rough estimate of height needed for *n* text lines."""
        # 16 px per line is a good default for Courier 9pt.
        return n * 16 + 8

    def set_state(self, text: str, colour: str):
        self.setText(text)
        self.setStyleSheet(
            f"border: 1px solid #37474f; border-radius: 4px; background: {colour}33;"
        )


class SnapshotWindow(QMainWindow):
    def __init__(self, stations: list[int], refresh_s: int):
        super().__init__()
        self.stations = stations
        self.refresh_s = refresh_s
        self.current_station = stations[0]
        self.blocks: dict[str, LogBlock] = {}

        self.setWindowTitle("miniTRASGO Log Snapshot")
        self.resize(1200, 700)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QGridLayout(central)
        layout.setHorizontalSpacing(12)
        layout.setVerticalSpacing(8)

        # Station selector + refresh button
        self.station_box = QComboBox()
        for st in stations:
            self.station_box.addItem(f"Station {st}", userData=st)
        self.station_box.currentIndexChanged.connect(self.change_station)

        self.open_dir_btn = QPushButton("Choose log dir …")
        self.open_dir_btn.clicked.connect(self.choose_dir)

        header_lbl = QLabel(
            datetime.now().strftime("Snapshot – %Y‑%m‑%d %H:%M:%S")
        )
        header_lbl.setStyleSheet(
            f"background: {HEADER_BG}; color: {HEADER_FG}; padding: 4px;"
        )
        header_lbl.setAlignment(Qt.AlignCenter)

        layout.addWidget(header_lbl, 0, 0, 1, 4)
        layout.addWidget(self.station_box, 1, 0)
        layout.addWidget(self.open_dir_btn, 1, 1)

        # Create LogBlocks in a 4‑column grid
        for idx, job in enumerate(JOBS):
            block = LogBlock(job)
            self.blocks[job] = block
            row = 2 + idx // 4
            col = idx % 4
            layout.addWidget(block, row, col)

        # Timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(refresh_s * 1000)
        self.refresh()  # initial fill

    # ---------------------------------------------------------------------
    # Slots / helpers
    # ---------------------------------------------------------------------
    def choose_dir(self):
        global LOG_DIR
        new_dir = QFileDialog.getExistingDirectory(
            self, "Select LOG_DIR", str(LOG_DIR)
        )
        if new_dir:
            LOG_DIR = Path(new_dir)
            self.refresh()

    def change_station(self, idx: int):
        self.current_station = self.station_box.itemData(idx)
        self.refresh()

    def refresh(self):
        now = time.time()
        for job, block in self.blocks.items():
            path = LOG_DIR / f"{job}_{self.current_station}.log"
            if path.is_file():
                age = now - path.stat().st_mtime
                if age < 300:
                    colour = GREEN
                elif age < 3600:
                    colour = YELLOW
                else:
                    colour = RED
                header = f"[{job}] (updated {time.strftime('%H:%M:%S', time.localtime(path.stat().st_mtime))})"
                tail = tail_lines(path, LINES)
                block.set_state(f"{header}\n{tail}", colour)
            else:
                block.set_state(f"[{job}] (missing)\n", RED)


# ----------------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------------

def tail_lines(path: Path, n: int) -> str:
    """Return the last *n* lines of *path* as a string."""
    try:
        with path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            end = f.tell()
            size = min(4096, end)
            f.seek(-size, os.SEEK_END)
            data = f.read().splitlines()
            return "\n".join(line.decode("utf‑8", "replace") for line in data[-n:])
    except Exception as e:
        return f"<error reading log: {e}>"


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------

def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="GUI snapshot of miniTRASGO logs")
    parser.add_argument("--stations", "-s", nargs="*", type=int, default=[1, 2, 3, 4])
    parser.add_argument("--refresh", "-r", type=int, default=UPDATE_INTERVAL_MS // 1000,
                        help="Refresh interval in seconds (default 10)")
    args = parser.parse_args(argv)

    app = QApplication(sys.argv)
    win = SnapshotWindow(args.stations, args.refresh)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

# ----------------------------------------------------------------------------
# PySide6 compatibility
# ----------------------------------------------------------------------------
# If you prefer PySide6, replace the PyQt5 imports with:
#   from PySide6.QtCore import QTimer, Qt
#   from PySide6.QtGui import QFont, QTextOption
#   from PySide6.QtWidgets import (QApplication, QComboBox, QFileDialog,
#                                  QGridLayout, QLabel, QMainWindow, QPushButton,
#                                  QTextEdit, QWidget)
