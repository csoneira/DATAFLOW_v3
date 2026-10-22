from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path("MINGO_ANALYSIS/MINGO_ANALYSIS_SCRIPTS/STAGES/STAGE_1/EVENT_DATA/STEP_1")
SCRIPTS = {
    0: "script_0_acq_to_raw.py",
    1: "script_1_raw_to_clean.py",
    2: "script_2_clean_to_cal.py",
    3: "script_3_cal_to_list.py",
    4: "script_4_list_to_fit.py",
    5: "script_5_fit_to_post.py",
}
PDF_FUNCTION_NAMES = {
    "_build_temp_pdf_path",
    "_write_task0_pdf_from_pngs",
    "close_direct_pdf_writer",
}
PDF_GLOBAL_PREFIXES = ("_direct_pdf_",)


def comment_source(source: str) -> str:
    return "\n".join(("# " + line.rstrip()) if line.strip() else "#" for line in source.rstrip("\n").splitlines()) + "\n"


def append_legacy(task: int, sections: list[tuple[str, str]]) -> None:
    if not sections:
        return
    path = ROOT / f"TASK_{task}/task_{task}_legacy.py"
    existing = path.read_text(encoding="utf-8").rstrip() + "\n\n" if path.exists() else (
        f"# LEGACY CODE ARCHIVE FOR TASK {task}\n# Not imported by the active pipeline.\n\n"
    )
    chunks = [existing]
    for reason, source in sections:
        chunks.extend([
            "# " + "=" * 76 + "\n",
            f"# ARCHIVED FROM: {SCRIPTS[task]}\n",
            f"# REASON: {reason}\n",
            "# " + "=" * 76 + "\n",
            comment_source(source),
            "\n",
        ])
    path.write_text("".join(chunks), encoding="utf-8")


def apply_replacements(source: str, replacements: list[tuple[int, int, str]]) -> str:
    lines = source.splitlines(keepends=True)
    for start, end, replacement in sorted(replacements, reverse=True):
        lines[start - 1:end] = [replacement] if replacement else []
    return "".join(lines)


def assigned_names(node: ast.AST) -> set[str]:
    targets: list[ast.AST] = []
    if isinstance(node, ast.Assign):
        targets.extend(node.targets)
    elif isinstance(node, ast.AnnAssign):
        targets.append(node.target)
    result: set[str] = set()
    for target in targets:
        if isinstance(target, ast.Name):
            result.add(target.id)
    return result


def remove_dead_pdf_nodes(task: int, source: str) -> tuple[str, list[tuple[str, str]]]:
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    replacements: list[tuple[int, int, str]] = []
    archived: list[tuple[str, str]] = []

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.end_lineno:
            if node.name in PDF_FUNCTION_NAMES or "pdf" in node.name.lower():
                block = "".join(lines[node.lineno - 1:node.end_lineno])
                replacements.append((node.lineno, node.end_lineno, ""))
                archived.append((f"dead runtime PDF helper after PNG-only split: {node.name}", block))
                continue
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and getattr(node, "end_lineno", None):
            names = assigned_names(node)
            if any(name.startswith(PDF_GLOBAL_PREFIXES) for name in names):
                block = "".join(lines[node.lineno - 1:node.end_lineno])
                replacements.append((node.lineno, node.end_lineno, ""))
                archived.append(("dead runtime PDF state after PNG-only split", block))
                continue
        if isinstance(node, ast.Expr) and node.end_lineno:
            text = ast.get_source_segment(source, node) or ""
            if "atexit.register(close_direct_pdf_writer" in text:
                block = "".join(lines[node.lineno - 1:node.end_lineno])
                replacements.append((node.lineno, node.end_lineno, ""))
                archived.append(("dead runtime PDF atexit registration after PNG-only split", block))

    return apply_replacements(source, replacements), archived


def clean_save_plot_figure(source: str) -> str:
    # Remove the small remnants that were left inside PNG save helpers when the
    # direct-PDF branch itself was removed. These are not functional anymore.
    source = source.replace(
        '    """Save a figure to PNG or directly append it to the task PDF."""\n',
        '    """Save one runtime plot as PNG."""\n',
    )
    source = source.replace(
        '    """Save plot to PNG or directly to the task PDF when PDF mode is active."""\n',
        '    """Save one runtime plot as PNG."""\n',
    )
    filtered: list[str] = []
    skip_next_pass = False
    for line in source.splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith("global _direct_pdf_"):
            continue
        if stripped.startswith("direct_pdf_path = globals().get("):
            continue
        if stripped == "# Runtime PDF branch removed: runtime plots are PNG-only.":
            skip_next_pass = True
            continue
        if skip_next_pass and stripped == "pass":
            skip_next_pass = False
            continue
        skip_next_pass = False
        filtered.append(line)
    return "".join(filtered)


def remove_unused_targeted_imports(source: str) -> str:
    # Repeat until stable: after PDF helpers disappear, the names below are safe
    # to drop only if AST says no Load reference remains.
    candidates = {"PdfPages", "Image", "pdf_save_rasterized_page", "collect_saved_plot_paths", "atexit"}
    while True:
        tree = ast.parse(source)
        loaded = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)}
        removable = candidates - loaded
        if not removable:
            return source
        lines = source.splitlines(keepends=True)
        replacements: list[tuple[int, int, str]] = []
        changed = False
        for node in tree.body:
            if isinstance(node, ast.Import) and node.end_lineno:
                kept = [alias for alias in node.names if (alias.asname or alias.name.split(".")[0]) not in removable]
                if len(kept) == len(node.names):
                    continue
                if kept:
                    replacement = "import " + ", ".join(
                        f"{alias.name} as {alias.asname}" if alias.asname else alias.name for alias in kept
                    ) + "\n"
                else:
                    replacement = ""
                replacements.append((node.lineno, node.end_lineno, replacement))
                changed = True
            elif isinstance(node, ast.ImportFrom) and node.end_lineno:
                imported_names = [(alias.asname or alias.name, alias) for alias in node.names]
                if not any(name in removable for name, _alias in imported_names):
                    continue
                kept = [alias for name, alias in imported_names if name not in removable]
                if kept:
                    module = "." * node.level + (node.module or "")
                    replacement = "from " + module + " import " + ", ".join(
                        f"{alias.name} as {alias.asname}" if alias.asname else alias.name for alias in kept
                    ) + "\n"
                else:
                    replacement = ""
                replacements.append((node.lineno, node.end_lineno, replacement))
                changed = True
        if not changed:
            return source
        source = apply_replacements(source, replacements)


def normalize(source: str) -> str:
    final_newline = source.endswith("\n")
    source = "\n".join(line.rstrip() for line in source.splitlines())
    return source + ("\n" if final_newline else "")


for task, script_name in SCRIPTS.items():
    path = ROOT / f"TASK_{task}/{script_name}"
    source = path.read_text(encoding="utf-8")
    source, archived = remove_dead_pdf_nodes(task, source)
    source = clean_save_plot_figure(source)
    source = remove_unused_targeted_imports(source)
    source = normalize(source)
    ast.parse(source, filename=str(path))
    path.write_text(source, encoding="utf-8")
    append_legacy(task, archived)
    print(f"Task {task}: archived {len(archived)} dead runtime-PDF node(s)")
