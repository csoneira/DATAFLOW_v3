#!/usr/bin/env python3
"""
DATAFLOW_v3 Script Header v1
Script: OPERATIONS/NOTIFICATIONS/TELEGRAM_BOT/mingo_analysis_bot.py
Purpose: Telegram bot that serves analysis PDFs on demand.
Owner: DATAFLOW_v3 contributors
Sign-off: csoneira <csoneira@ucm.es>
Last Updated: 2026-03-02
Runtime: python3
Usage: python3 OPERATIONS/NOTIFICATIONS/TELEGRAM_BOT/mingo_analysis_bot.py [options]
Inputs: CLI args, config files, environment variables, and/or upstream files.
Outputs: Files, logs, plots, or stdout/stderr side effects.
Notes: Keep behavior configuration-driven and reproducible.
"""

from __future__ import annotations

import logging
import os
import shlex
import subprocess
from datetime import datetime
from pathlib import Path
import telebot


BASE_DIR = Path(__file__).resolve().parents[3]
TELEGRAM_DIR = Path(__file__).resolve().parent
TOKEN_PATH = TELEGRAM_DIR / "API_TOKEN.txt"
RESTART_SCRIPT = TELEGRAM_DIR / "kill_bot_and_restart.sh"
TERMINAL_PASSWORD_PATH = TELEGRAM_DIR / "TERMINAL_PASSWORD.txt"

PDF_TARGETS = {
    "definitive_execution_report": {
        "path": BASE_DIR
        / "MASTER"
        / "ANCILLARY"
        / "PLOTTERS"
        / "DEFINITIVE_EXECUTION"
        / "PLOTS"
        / "definitive_execution_map_report.pdf",
        "description": "Definitive execution map report",
    },
    "filter_metadata_report": {
        "path": BASE_DIR
        / "MASTER"
        / "ANCILLARY"
        / "PLOTTERS"
        / "METADATA"
        / "FILTER"
        / "PLOTS"
        / "filter_metadata_report.pdf",
        "description": "Filter metadata report",
    },
    "rates_metadata_report": {
        "path": BASE_DIR
        / "MASTER"
        / "ANCILLARY"
        / "PLOTTERS"
        / "RATES"
        / "PLOTS"
        / "rates_metadata_report.pdf",
        "description": "Rates metadata report",
    },
    "execution_metadata_report": {
        "path": BASE_DIR
        / "MASTER"
        / "ANCILLARY"
        / "PLOTTERS"
        / "METADATA"
        / "EXECUTION"
        / "PLOTS"
        / "execution_metadata_report.pdf",
        "description": "Execution metadata report",
    },
    "efficiency_metadata_report": {
        "path": BASE_DIR
        / "MASTER"
        / "ANCILLARY"
        / "PLOTTERS"
        / "METADATA"
        / "EFFICIENCIES"
        / "PLOTS"
        / "efficiency_metadata_report.pdf",
        "description": "Efficiency metadata report",
    },
    "simulated_data_evolution_report": {
        "path": BASE_DIR
        / "MASTER"
        / "ANCILLARY"
        / "PLOTTERS"
        / "SIMULATED_DATA_EVOLUTION"
        / "PLOTS"
        / "simulated_data_evolution_report.pdf",
        "description": "Simulated data evolution report",
    },
    "param_mesh_summary": {
        "path": BASE_DIR
        / "MINGO_DIGITAL_TWIN"
        / "PLOTTERS"
        / "EXECUTION"
        / "MESH"
        / "param_mesh_summary.pdf",
        "description": "Param mesh summary PDF",
    },
    "simulation_execution_time_hist": {
        "path": BASE_DIR
        / "MINGO_DIGITAL_TWIN"
        / "PLOTTERS"
        / "EXECUTION"
        / "SIMULATION_TIME"
        / "simulation_execution_time_hist.pdf",
        "description": "Simulation execution time histogram",
    },
    "backpressure_monitor": {
        "path": BASE_DIR
        / "MINGO_DIGITAL_TWIN"
        / "PLOTTERS"
        / "EXECUTION"
        / "BACKPRESSURE"
        / "backpressure_monitor.pdf",
        "description": "STEP_0 backpressure monitor PDF",
    },
}

CLEANER_SCRIPT = (
    BASE_DIR / "OPERATIONS" / "MAINTENANCE" / "CLEANERS" / "clean_dataflow.sh"
)
MAX_MESSAGE_CHARS = 3500
TERMINAL_COMMAND_TIMEOUT_SEC = 300

HELP_TEXT = (
    "===========================================\n"
    "      mingo_analysis_bot - Command Guide\n"
    "===========================================\n\n"
    "General Commands:\n"
    "  /start or /help - Display this guide.\n\n"
    "SIMULATION PDF Reports:\n"
    "  /param_mesh_summary - Param mesh summary PDF.\n"
    "  /simulation_execution_time_hist - Simulation execution time histogram PDF.\n"
    "  /backpressure_monitor - Send latest STEP_0 backpressure monitor PDF.\n\n"
    "ANALYSIS PDF Reports:\n"
    "  /definitive_execution_report - Definitive execution map PDF.\n"
    "  /filter_metadata_report - Filter metadata PDF.\n"
    "  /rates_metadata_report - Rates metadata PDF.\n"
    "  /execution_metadata_report - Execution metadata PDF.\n"
    "  /efficiency_metadata_report - Efficiency metadata PDF.\n"
    "  /simulated_data_evolution_report - Simulated data evolution report PDF.\n\n"
    "Maintenance Tools:\n"
    "  /clean_dataflow_status - Run clean_dataflow.sh --compact to show disk usage.\n"
    "  /clean_dataflow_force - Run clean_dataflow.sh --force --compact (temps, plots, completed; never metadata).\n\n"
    "Terminal:\n"
    "  /terminal <password> - Enable terminal mode for this chat.\n"
    "  /exit - Disable terminal mode.\n\n"
    "Operation Tools:\n"
    "  /restart_bot - Restart this Telegram bot.\n"
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
TERMINAL_ACTIVE_CHATS: set[int] = set()
TERMINAL_CHAT_CWDS: dict[int, Path] = {}


def load_terminal_password(password_path: Path) -> str | None:
    if not password_path.exists():
        logging.warning("Terminal mode disabled: password file not found at %s", password_path)
        return None

    password = password_path.read_text(encoding="utf-8").strip()
    if not password:
        logging.warning("Terminal mode disabled: password file %s is empty", password_path)
        return None
    return password


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
        if maybe_handle_terminal_message(message):
            return
        send_pdf(message.chat.id, pdf_path, caption)


for command, payload in PDF_TARGETS.items():
    register_static_pdf_command(command, payload["path"], payload["description"])


@bot.message_handler(commands=["start", "help"])
def send_welcome(message):
    if maybe_handle_terminal_message(message):
        return
    bot.send_message(message.chat.id, HELP_TEXT)


def truncate_message(text: str) -> str:
    if len(text) <= MAX_MESSAGE_CHARS:
        return text
    return text[: MAX_MESSAGE_CHARS - 3] + "..."


def run_cleaner(force: bool = False) -> str:
    if not CLEANER_SCRIPT.exists():
        raise RuntimeError(f"Cleaner script not found: {CLEANER_SCRIPT}")

    command = [str(CLEANER_SCRIPT), "--compact"]
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

    mode = "--force --compact" if force else "--compact (disk usage check)"
    message = (
        f"clean_dataflow.sh {mode} finished with exit code {result.returncode}.\n"
        f"{output}"
    )
    return truncate_message(message)


def get_chat_cwd(chat_id: int) -> Path:
    cwd = TERMINAL_CHAT_CWDS.get(chat_id, BASE_DIR)
    if not cwd.exists() or not cwd.is_dir():
        cwd = BASE_DIR
        TERMINAL_CHAT_CWDS[chat_id] = cwd
    return cwd


def maybe_handle_cd_command(chat_id: int, command_text: str) -> str | None:
    try:
        tokens = shlex.split(command_text)
    except ValueError as exc:
        return truncate_message(f"$ {command_text}\nInvalid command syntax: {exc}")

    if not tokens or tokens[0] != "cd":
        return None

    if len(tokens) > 2:
        return truncate_message(f"$ {command_text}\nUsage: cd <path>")

    current_cwd = get_chat_cwd(chat_id)
    target_raw = tokens[1] if len(tokens) == 2 else "~"
    target_expanded = os.path.expandvars(os.path.expanduser(target_raw))
    target = Path(target_expanded)
    if not target.is_absolute():
        target = (current_cwd / target).resolve()
    else:
        target = target.resolve()

    if not target.exists():
        return truncate_message(
            f"$ {command_text}\nExit code: 1\n/bin/bash: line 1: cd: {target}: No such file or directory"
        )

    if not target.is_dir():
        return truncate_message(
            f"$ {command_text}\nExit code: 1\n/bin/bash: line 1: cd: {target}: Not a directory"
        )

    TERMINAL_CHAT_CWDS[chat_id] = target
    return truncate_message(f"$ {command_text}\nExit code: 0\nCurrent directory: {target}")


def run_terminal_command(command_text: str, cwd: Path) -> str:
    try:
        result = subprocess.run(
            command_text,
            cwd=str(cwd),
            shell=True,
            executable="/bin/bash",
            capture_output=True,
            text=True,
            check=False,
            timeout=TERMINAL_COMMAND_TIMEOUT_SEC,
        )
    except subprocess.TimeoutExpired as exc:
        partial = "".join(
            part
            for part in (
                exc.stdout or "",
                "\n" if exc.stdout and exc.stderr else "",
                exc.stderr or "",
            )
        ).strip()
        body = partial or "<no output>"
        return truncate_message(
            f"$ {command_text}\nTimed out after {TERMINAL_COMMAND_TIMEOUT_SEC}s.\n{body}"
        )
    except Exception as exc:  # pragma: no cover - defensive for runtime failures
        return truncate_message(f"$ {command_text}\nExecution failed: {exc}")

    output = "".join(
        part
        for part in (
            result.stdout or "",
            "\n" if result.stdout and result.stderr else "",
            result.stderr or "",
        )
    ).strip() or "<no output>"
    return truncate_message(
        f"$ {command_text}\nExit code: {result.returncode}\n{output}"
    )


def trigger_restart() -> None:
    if not RESTART_SCRIPT.exists():
        raise RuntimeError(f"Restart script not found: {RESTART_SCRIPT}")
    subprocess.Popen(
        ["/bin/bash", str(RESTART_SCRIPT)],
        cwd=str(RESTART_SCRIPT.parent),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def maybe_handle_terminal_message(message) -> bool:
    chat_id = message.chat.id
    if chat_id not in TERMINAL_ACTIVE_CHATS:
        return False

    text = (message.text or "").strip()
    if not text:
        bot.send_message(chat_id, "Terminal mode is active. Send a command or use /exit.")
        return True

    if text == "/exit":
        TERMINAL_ACTIVE_CHATS.discard(chat_id)
        TERMINAL_CHAT_CWDS.pop(chat_id, None)
        bot.send_message(chat_id, "Terminal mode disabled.")
        return True

    cd_response = maybe_handle_cd_command(chat_id, text)
    if cd_response is not None:
        bot.send_message(chat_id, cd_response)
        return True

    bot.send_message(chat_id, run_terminal_command(text, get_chat_cwd(chat_id)))
    return True


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
    if maybe_handle_terminal_message(message):
        return
    _handle_cleaner_command(message, force=False)


@bot.message_handler(commands=["clean_dataflow_force"])
def handle_clean_dataflow_force(message):  # type: ignore[misc]
    if maybe_handle_terminal_message(message):
        return
    _handle_cleaner_command(message, force=True)


@bot.message_handler(commands=["terminal"])
def handle_terminal(message):  # type: ignore[misc]
    chat_id = message.chat.id
    terminal_password = load_terminal_password(TERMINAL_PASSWORD_PATH)
    if terminal_password is None:
        bot.send_message(chat_id, "Terminal mode is disabled because no password is configured.")
        return

    parts = (message.text or "").split(maxsplit=1)
    if len(parts) < 2 or not parts[1].strip():
        bot.send_message(chat_id, "Usage: /terminal <password>")
        return

    provided = parts[1].strip()
    if provided != terminal_password:
        bot.send_message(chat_id, "Wrong password.")
        return

    TERMINAL_ACTIVE_CHATS.add(chat_id)
    TERMINAL_CHAT_CWDS[chat_id] = BASE_DIR
    bot.send_message(
        chat_id,
        f"Terminal mode enabled. Current directory: {BASE_DIR}\n"
        "Any text will run as a shell command. Use /exit to leave.",
    )


@bot.message_handler(commands=["exit"])
def handle_exit(message):  # type: ignore[misc]
    chat_id = message.chat.id
    if chat_id in TERMINAL_ACTIVE_CHATS:
        TERMINAL_ACTIVE_CHATS.discard(chat_id)
        TERMINAL_CHAT_CWDS.pop(chat_id, None)
        bot.send_message(chat_id, "Terminal mode disabled.")
        return
    bot.send_message(chat_id, "Terminal mode is not active.")


@bot.message_handler(commands=["restart_bot"])
def handle_restart_bot(message):  # type: ignore[misc]
    if maybe_handle_terminal_message(message):
        return
    chat_id = message.chat.id
    bot.send_message(chat_id, "Restart command received. Attempting to restart bot...")
    try:
        trigger_restart()
    except Exception as exc:  # pragma: no cover
        logging.exception("Failed to trigger restart")
        bot.send_message(chat_id, f"Unable to restart bot: {exc}")
        return
    # The restart script will terminate this process shortly after this handler returns.


@bot.message_handler(func=lambda message: True, content_types=["text"])
def handle_unknown_text(message):
    """Fallback handler: show the start/help message for any unrecognized text."""
    if maybe_handle_terminal_message(message):
        return

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
            "/terminal",
            "/exit",
            "/restart_bot",
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
