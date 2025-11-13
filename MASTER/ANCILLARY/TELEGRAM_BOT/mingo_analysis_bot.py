#!/usr/bin/env python3
"""Telegram bot that serves analysis PDFs on demand."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional

import telebot


BASE_DIR = Path(__file__).resolve().parents[3]
TELEGRAM_DIR = Path(__file__).resolve().parent
TOKEN_PATH = TELEGRAM_DIR / "API_TOKEN.txt"

PIPELINE_PLOTS_DIR = (
    BASE_DIR / "MASTER" / "ANCILLARY" / "PLOTTERS" / "PIPELINE_TRACK_FILES" / "PLOTS"
)
PIPELINE_DEFAULT_PDF = PIPELINE_PLOTS_DIR / "station_output_file_counts.pdf"
PIPELINE_COMPLETE_PDF = PIPELINE_PLOTS_DIR / "station_output_file_counts_complete.pdf"

PDF_TARGETS = {
    "high_voltage_summary": {
        "path": BASE_DIR
        / "MASTER"
        / "ANCILLARY"
        / "PLOTTERS"
        / "HIGH_VOLTAGE"
        / "PLOTS"
        / "high_voltage_network_summary.pdf",
        "description": "High voltage network summary",
    },
    "execution_report_realtime": {
        "path": BASE_DIR
        / "MASTER"
        / "ANCILLARY"
        / "PLOTTERS"
        / "METADATA"
        / "EXECUTION"
        / "PLOTS"
        / "execution_metadata_report_real_time.pdf",
        "description": "Execution metadata report (real time)",
    },
    "execution_report_zoomed": {
        "path": BASE_DIR
        / "MASTER"
        / "ANCILLARY"
        / "PLOTTERS"
        / "METADATA"
        / "EXECUTION"
        / "PLOTS"
        / "execution_metadata_report_zoomed.pdf",
        "description": "Execution metadata report (zoomed)",
    },
    "execution_report_full": {
        "path": BASE_DIR
        / "MASTER"
        / "ANCILLARY"
        / "PLOTTERS"
        / "METADATA"
        / "EXECUTION"
        / "PLOTS"
        / "execution_metadata_report.pdf",
        "description": "Execution metadata report (full)",
    },
    "real_time_hv_exec": {
        "path": BASE_DIR
        / "MASTER"
        / "ANCILLARY"
        / "PLOTTERS"
        / "REAL_TIME_HV_AND_EXEC"
        / "PLOTS"
        / "real_time_hv_and_execution.pdf",
        "description": "Real-time HV & execution PDF",
    },
    "online_file_count": {
        "path": BASE_DIR
        / "MASTER"
        / "ANCILLARY"
        / "PLOTTERS"
        / "ONLINE_FILE_COUNT"
        / "PLOTS"
        / "online_file_count_report.pdf",
        "description": "Online file count report",
    },
}

HELP_TEXT = (
    "Welcome to mingo_analysis_bot!\n\n"
    "Available commands:\n"
    "/start or /help - show this message\n"
    "/high_voltage_summary - send the latest high voltage network summary\n"
    "/execution_report_realtime - execution metadata (real time)\n"
    "/execution_report_zoomed - execution metadata (zoomed)\n"
    "/execution_report_full - execution metadata (full report)\n"
    "/real_time_hv_exec - combined real-time HV and execution PDF\n"
    "/online_file_count - online file count report\n"
    "/pipeline_latest - send the most recent pipeline tracker PDF\n"
    "/pipeline_complete - send the complete pipeline tracker PDF\n"
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def load_token(token_path: Path) -> str:
    try:
        token = token_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError as exc:
        raise RuntimeError(f"Token file not found: {token_path}") from exc
    if not token:
        raise RuntimeError(f"Token file {token_path} is empty.")
    return token


bot = telebot.TeleBot(load_token(TOKEN_PATH))


def send_pdf(chat_id: int, pdf_path: Path, caption: str) -> None:
    if not pdf_path.exists():
        bot.send_message(chat_id, f"File not found: {pdf_path}")
        return

    try:
        with pdf_path.open("rb") as handle:
            bot.send_document(chat_id, handle, caption=caption)
    except Exception as exc:  # pragma: no cover - defensive for runtime failures
        logging.exception("Failed to send %s", pdf_path)
        bot.send_message(chat_id, f"Unable to send file due to error: {exc}")


def register_static_pdf_command(command: str, path: Path, description: str) -> None:
    @bot.message_handler(commands=[command])
    def handler(message, pdf_path=path, caption=description):  # type: ignore[misc]
        send_pdf(message.chat.id, pdf_path, caption)


for command, payload in PDF_TARGETS.items():
    register_static_pdf_command(command, payload["path"], payload["description"])


@bot.message_handler(commands=["start", "help"])
def send_welcome(message):
    bot.send_message(message.chat.id, HELP_TEXT)


def pipeline_pdfs() -> List[Path]:
    if not PIPELINE_PLOTS_DIR.exists():
        return []
    return sorted(PIPELINE_PLOTS_DIR.glob("*.pdf"))


def most_recent_pipeline_pdf() -> Optional[Path]:
    pdfs = pipeline_pdfs()
    if not pdfs:
        return None
    return max(pdfs, key=lambda path: path.stat().st_mtime)


@bot.message_handler(commands=["pipeline_latest"])
def handle_pipeline_latest(message):
    latest = most_recent_pipeline_pdf()
    if not latest:
        bot.send_message(
            message.chat.id,
            f"No PDF files were found in {PIPELINE_PLOTS_DIR}",
        )
        return

    caption = f"Latest pipeline file tracker\nGenerated: {datetime.fromtimestamp(latest.stat().st_mtime):%Y-%m-%d %H:%M}"
    send_pdf(message.chat.id, latest, caption)


@bot.message_handler(commands=["pipeline_complete"])
def handle_pipeline_complete(message):
    if not PIPELINE_COMPLETE_PDF.exists():
        bot.send_message(
            message.chat.id,
            "Complete tracker PDF not found. Run file_tracker_plotter.py with --complete first."
            f"\nExpected path: {PIPELINE_COMPLETE_PDF}",
        )
        return

    mtime = datetime.fromtimestamp(PIPELINE_COMPLETE_PDF.stat().st_mtime)
    caption = (
        "Complete pipeline tracker (all directories)\n"
        f"Generated: {mtime:%Y-%m-%d %H:%M}"
    )
    send_pdf(message.chat.id, PIPELINE_COMPLETE_PDF, caption)


@bot.message_handler(func=lambda message: True, content_types=["text"])
def handle_unknown_text(message):
    """Fallback handler: show the start/help message for any unrecognized text."""
    if message.text.strip() in ("", None):
        bot.send_message(message.chat.id, HELP_TEXT)
        return

    known_commands = {f"/{cmd}" for cmd in PDF_TARGETS.keys()}
    known_commands.update(
        {
            "/start",
            "/help",
            "/pipeline",
            "/pipeline_latest",
            "/pipeline_list",
            "/pipeline_complete",
        }
    )

    if message.text.split()[0] in known_commands:
        # Known command but likely malformed (e.g., missing args)
        bot.send_message(message.chat.id, HELP_TEXT)
        return

    bot.send_message(
        message.chat.id,
        "I only respond to the documented commands. Here is what I can do:\n\n"
        f"{HELP_TEXT}",
    )

bot.infinity_polling()

# def main() -> None:
#     logging.info("Starting mingo_analysis_bot as @mingo_analysis_bot")
#     bot.infinity_polling(timeout=60, long_polling_timeout=20)


# if __name__ == "__main__":
#     main()
