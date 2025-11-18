#!/usr/bin/env python3
"""Telegram bot that serves analysis PDFs on demand."""

from __future__ import annotations

import logging
import subprocess
from datetime import datetime
from pathlib import Path
import telebot


BASE_DIR = Path(__file__).resolve().parents[3]
TELEGRAM_DIR = Path(__file__).resolve().parent
TOKEN_PATH = TELEGRAM_DIR / "API_TOKEN.txt"

PDF_TARGETS = {
    "fill_factor_timeseries": {
        "path": BASE_DIR
        / "MASTER"
        / "ANCILLARY"
        / "PLOTTERS"
        / "FILL_FACTOR"
        / "PLOTS"
        / "fill_factor_timeseries.pdf",
        "description": "Fill-factor coverage report",
    },
    "execution_report_realtime_hv": {
        "path": BASE_DIR
        / "MASTER"
        / "ANCILLARY"
        / "PLOTTERS"
        / "METADATA"
        / "EXECUTION"
        / "PLOTS"
        / "execution_metadata_report_real_time_hv.pdf",
        "description": "Execution metadata report (real-time HV)",
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

CLEANER_SCRIPT = (
    BASE_DIR / "MASTER" / "ANCILLARY" / "CLEANERS" / "clean_dataflow.sh"
)
MAX_MESSAGE_CHARS = 3500

HELP_TEXT = (
    "===========================================\n"
    "      mingo_analysis_bot - Command Guide\n"
    "===========================================\n\n"
    "General Commands:\n"
    "  /start or /help - Display this guide.\n\n"
    "PDF Reports:\n"
    "  /fill_factor_timeseries - Fill-factor coverage PDF.\n"
    "  /execution_report_realtime_hv - Execution metadata (real-time HV overlay).\n"
    "  /execution_report_zoomed - Execution metadata (zoomed view).\n"
    "  /online_file_count - Online file count report.\n\n"
    "Maintenance Tools:\n"
    "  /clean_dataflow_status - Run clean_dataflow.sh to show disk usage.\n"
    "  /clean_dataflow_force - Run clean_dataflow.sh --force for cleanup.\n"
    "==========================================="
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


def send_pdf(chat_id: int, pdf_path: Path, description: str) -> None:
    if not pdf_path.exists():
        bot.send_message(chat_id, f"File not found: {pdf_path}")
        return

    creation = datetime.fromtimestamp(pdf_path.stat().st_mtime)
    caption = f"{description}\nCreated: {creation:%Y-%m-%d %H:%M}"

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


def truncate_message(text: str) -> str:
    if len(text) <= MAX_MESSAGE_CHARS:
        return text
    return text[: MAX_MESSAGE_CHARS - 3] + "..."


def run_cleaner(force: bool = False) -> str:
    if not CLEANER_SCRIPT.exists():
        raise RuntimeError(f"Cleaner script not found: {CLEANER_SCRIPT}")

    command = [str(CLEANER_SCRIPT)]
    if force:
        command.append("--force")

    logging.info("Calling %s", " ".join(command))

    try:
        result = subprocess.run(
            command,
            cwd=str(CLEANER_SCRIPT.parent),
            capture_output=True,
            text=True,
            check=False,
            timeout=600,
        )
    except Exception as exc:  # pragma: no cover - defensive for runtime failures
        raise RuntimeError(f"Failed to execute clean_dataflow.sh: {exc}") from exc

    output = "".join(
        part
        for part in (
            result.stdout or "",
            "\n" if result.stdout and result.stderr else "",
            result.stderr or "",
        )
    ).strip() or "<no output>"

    mode = "--force" if force else "(disk usage check)"
    message = (
        f"clean_dataflow.sh {mode} finished with exit code {result.returncode}.\n"
        f"{output}"
    )
    return truncate_message(message)


def _handle_cleaner_command(message, force: bool) -> None:
    try:
        response = run_cleaner(force)
    except Exception as exc:  # pragma: no cover
        logging.exception("clean_dataflow.sh command failed")
        bot.send_message(message.chat.id, f"Unable to run clean_dataflow.sh: {exc}")
        return

    bot.send_message(message.chat.id, response)


@bot.message_handler(commands=["clean_dataflow_status"])
def handle_clean_dataflow_status(message):  # type: ignore[misc]
    _handle_cleaner_command(message, force=False)


@bot.message_handler(commands=["clean_dataflow_force"])
def handle_clean_dataflow_force(message):  # type: ignore[misc]
    _handle_cleaner_command(message, force=True)


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
            "/clean_dataflow_status",
            "/clean_dataflow_force",
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
