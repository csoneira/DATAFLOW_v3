from pathlib import Path

root = Path("MINGO_ANALYSIS/MINGO_ANALYSIS_SCRIPTS/STAGES/STAGE_1/EVENT_DATA/STEP_1")
for task in range(6):
    task_dir = root / f"TASK_{task}"
    for path in task_dir.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        had_final_newline = text.endswith("\n")
        cleaned = "\n".join(line.rstrip() for line in text.splitlines())
        if had_final_newline:
            cleaned += "\n"
        path.write_text(cleaned, encoding="utf-8")
