---
title: Script Header Standard
description: Canonical header/docstring contract for executable scripts across DATAFLOW_v3.
last_updated: 2026-03-02
status: active
---

# Script Header Standard

This document defines the standard header format for executable scripts in DATAFLOW_v3.

## Why this exists

- Make every script self-describing.
- Improve maintainability and onboarding.
- Preserve professional, auditable engineering practice.
- Keep code quality presentation consistent for external review.

## Required marker

Every script header must include this marker exactly:

- `DATAFLOW_v3 Script Header v1`

This allows automated tooling to detect/verify standard-compliant headers.

## Required fields

All script headers must include:

1. `Script`: repository-relative path
2. `Purpose`: concise one-line behavior summary
3. `Owner`: team or subsystem owner
4. `Sign-off`: maintainer signature (`name <email>`)
5. `Last Updated`: `YYYY-MM-DD`
6. `Runtime`: language/runtime (`python3`, `bash`, `octave/matlab`, `html`)
7. `Usage`: canonical invocation pattern
8. `Inputs`: key input channels (CLI/config/files/env)
9. `Outputs`: key outputs/artifacts
10. `Notes`: short safety/reproducibility note

## Python template

```python
#!/usr/bin/env python3
"""
DATAFLOW_v3 Script Header v1
Script: path/to/script.py
Purpose: One-line purpose.
Owner: DATAFLOW_v3 contributors
Sign-off: <name> <email>
Last Updated: YYYY-MM-DD
Runtime: python3
Usage: python3 path/to/script.py [options]
Inputs: CLI args, config files, environment variables, and/or upstream files.
Outputs: Files, logs, plots, or stdout/stderr side effects.
Notes: Keep behavior configuration-driven and reproducible.
"""
```

## Bash template

```bash
#!/usr/bin/env bash
# =============================================================================
# DATAFLOW_v3 Script Header v1
# Script: path/to/script.sh
# Purpose: One-line purpose.
# Owner: DATAFLOW_v3 contributors
# Sign-off: <name> <email>
# Last Updated: YYYY-MM-DD
# Runtime: bash
# Usage: bash path/to/script.sh [options]
# Inputs: CLI args, config files, environment variables, and/or upstream files.
# Outputs: Files, logs, or process-level side effects.
# Notes: Keep behavior configuration-driven and reproducible.
# =============================================================================
```

## Octave/MATLAB template

```matlab
% =============================================================================
% DATAFLOW_v3 Script Header v1
% Script: path/to/script.m
% Purpose: One-line purpose.
% Owner: DATAFLOW_v3 contributors
% Sign-off: <name> <email>
% Last Updated: YYYY-MM-DD
% Runtime: octave/matlab
% Usage: Run from MATLAB/Octave entrypoint with expected args/context.
% Inputs: Variables, config files, environment, and/or upstream files.
% Outputs: Variables, files, plots, or logs.
% Notes: Keep behavior configuration-driven and reproducible.
% =============================================================================
```

## HTML template

```html
<!--
=============================================================================
DATAFLOW_v3 Script Header v1
Script: path/to/page.html
Purpose: One-line purpose.
Owner: DATAFLOW_v3 contributors
Sign-off: <name> <email>
Last Updated: YYYY-MM-DD
Runtime: html
Usage: Open/render in browser or embed in docs/UI workflow.
Inputs: Static assets, linked resources, template variables (if any).
Outputs: Rendered page/view.
Notes: Keep references stable and document external dependencies.
=============================================================================
-->
```

## Rules

- Do not modify runtime logic when standardizing headers.
- Preserve shebang and Python encoding declaration placement.
- Keep existing meaningful purpose text when available.
- Use repository-relative paths.
- Update `Last Updated` whenever script behavior changes.

## Validation guideline

Before merging header updates:

- Confirm no logic lines changed.
- Confirm script remains syntactically valid.
- Confirm marker presence in standardized files.

