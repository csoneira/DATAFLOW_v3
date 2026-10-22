from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path("MINGO_ANALYSIS/MINGO_ANALYSIS_SCRIPTS")
STEP = ROOT / "STAGES/STAGE_1/EVENT_DATA/STEP_1"

SCRIPT_NAMES = {
    0: "script_0_acq_to_raw.py",
    1: "script_1_raw_to_clean.py",
    2: "script_2_clean_to_cal.py",
    3: "script_3_cal_to_list.py",
    4: "script_4_list_to_fit.py",
    5: "script_5_fit_to_post.py",
}


def comment_source(source: str) -> str:
    return "\n".join(("# " + line.rstrip()) if line.strip() else "#" for line in source.rstrip("\n").splitlines()) + "\n"


def append_legacy(task: int, sections: list[tuple[str, str]]) -> None:
    if not sections:
        return
    path = STEP / f"TASK_{task}/task_{task}_legacy.py"
    existing = path.read_text(encoding="utf-8").rstrip() + "\n\n" if path.exists() else (
        f"# LEGACY CODE ARCHIVE FOR TASK {task}\n# Not imported by the active pipeline.\n\n"
    )
    chunks = [existing]
    for reason, block in sections:
        chunks.extend([
            "# " + "=" * 76 + "\n",
            f"# ARCHIVED FROM: {SCRIPT_NAMES[task]}\n",
            f"# REASON: {reason}\n",
            "# " + "=" * 76 + "\n",
            comment_source(block),
            "\n",
        ])
    path.write_text("".join(chunks), encoding="utf-8")


def apply_replacements(source: str, replacements: list[tuple[int, int, str]]) -> str:
    lines = source.splitlines(keepends=True)
    for start, end, replacement in sorted(replacements, reverse=True):
        lines[start - 1:end] = [replacement] if replacement else []
    return "".join(lines)


def uses_name(node: ast.AST, name: str) -> bool:
    return any(isinstance(item, ast.Name) and item.id == name for item in ast.walk(node))


def remove_runtime_pdf_branches(task: int) -> None:
    path = STEP / f"TASK_{task}/{SCRIPT_NAMES[task]}"
    source = path.read_text(encoding="utf-8")

    # No local code is allowed to re-enable PDF output in the runtime task.
    source = re.sub(
        r"(?m)^(\s*)create_pdf\s*=\s*True(?:\s*#.*)?$",
        r"\1create_pdf = False  # Runtime plots are strictly PNG-only.",
        source,
    )

    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    candidates: list[ast.If] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If) or node.end_lineno is None or node.orelse:
            continue
        block = "".join(lines[node.lineno - 1:node.end_lineno])
        test_source = ast.get_source_segment(source, node.test) or ""
        pdf_gate = uses_name(node.test, "create_pdf") or "create_pdf" in test_source
        direct_pdf_gate = "globals().get(\"create_pdf\"" in test_source or "globals().get('create_pdf'" in test_source
        if (pdf_gate or direct_pdf_gate) and "pdf" in block.lower():
            candidates.append(node)

    candidates.sort(key=lambda node: (node.lineno, -(node.end_lineno or node.lineno)))
    selected: list[ast.If] = []
    for node in candidates:
        if any(
            outer.lineno <= node.lineno
            and (node.end_lineno or node.lineno) <= (outer.end_lineno or outer.lineno)
            for outer in selected
        ):
            continue
        selected.append(node)

    replacements: list[tuple[int, int, str]] = []
    archived: list[tuple[str, str]] = []
    for node in selected:
        indent = lines[node.lineno - 1][: len(lines[node.lineno - 1]) - len(lines[node.lineno - 1].lstrip())]
        block = "".join(lines[node.lineno - 1:node.end_lineno])
        replacements.append((
            node.lineno,
            node.end_lineno,
            f"{indent}# Runtime PDF branch removed: runtime plots are PNG-only.\n{indent}pass\n",
        ))
        archived.append(("runtime PDF branch disabled by the PNG-only runtime plotting contract", block))

    if replacements:
        source = apply_replacements(source, replacements)

    path.write_text(source, encoding="utf-8")
    append_legacy(task, archived)
    print(f"Task {task}: removed {len(archived)} runtime PDF branch(es)")


def remove_unreferenced_plot_helpers(task: int) -> None:
    path = STEP / f"TASK_{task}/{SCRIPT_NAMES[task]}"
    source = path.read_text(encoding="utf-8")

    # Repeat because deleting one helper can make another helper unused.
    archived: list[tuple[str, str]] = []
    while True:
        tree = ast.parse(source)
        lines = source.splitlines(keepends=True)
        top_functions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.end_lineno]
        referenced_names = {
            node.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        }
        removable = []
        for function in top_functions:
            name = function.name
            block = "".join(lines[function.lineno - 1:function.end_lineno])
            is_plot_helper = (
                name.startswith("plot_")
                or name.startswith("_plot_")
                or "PdfPages(" in block
                or "pdf_save_rasterized_page" in block
            )
            if is_plot_helper and name not in referenced_names:
                removable.append(function)
        if not removable:
            break
        replacements = []
        for function in removable:
            block = "".join(lines[function.lineno - 1:function.end_lineno])
            replacements.append((function.lineno, function.end_lineno, ""))
            archived.append((f"unreferenced plotting helper after runtime/FINAL separation: {function.name}", block))
        source = apply_replacements(source, replacements)

    path.write_text(source, encoding="utf-8")
    append_legacy(task, archived)
    if archived:
        print(f"Task {task}: archived {len(archived)} now-unreferenced plotting helper(s)")


def patch_task3_catalog_loader() -> None:
    path = STEP / "TASK_3/script_3_cal_to_list.py"
    source = path.read_text(encoding="utf-8")
    old = '''    missing_aliases = [alias for alias in TASK3_PLOT_ALIASES if alias not in catalog]\n    for alias in missing_aliases:\n        warnings.warn(\n            f"Task 3 plot alias {alias!r} is not listed in {catalog_path.name}; defaulting to 'usual'.",\n            RuntimeWarning,\n        )\n        catalog[alias] = "usual"\n'''
    new = '''    missing_aliases = [alias for alias in TASK3_PLOT_ALIASES if alias not in catalog]\n    for alias in missing_aliases:\n        # Known aliases missing from the runtime catalog are intentionally not\n        # runtime plots; FINAL aliases live in config_final_plots_task_3.yaml.\n        catalog[alias] = "none"\n'''
    if old not in source:
        raise RuntimeError("Task 3 plot-loader default block not found")
    path.write_text(source.replace(old, new, 1), encoding="utf-8")


def normalize_trailing_whitespace() -> None:
    for task in range(6):
        for path in (STEP / f"TASK_{task}").glob("*.py"):
            text = path.read_text(encoding="utf-8")
            final_newline = text.endswith("\n")
            cleaned = "\n".join(line.rstrip() for line in text.splitlines())
            if final_newline:
                cleaned += "\n"
            path.write_text(cleaned, encoding="utf-8")


patch_task3_catalog_loader()
for task_number in range(6):
    remove_runtime_pdf_branches(task_number)
for task_number in range(6):
    remove_unreferenced_plot_helpers(task_number)
normalize_trailing_whitespace()
