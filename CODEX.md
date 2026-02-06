**Purpose And Scope**
- Define how we work on this data analysis repository with a focus on correctness, reproducibility, and small refactors.
- Keep changes minimal, deterministic, and easy to audit.

**Typical Tasks**
- Data analysis and summary checks.
- Refactors and cleanup that preserve outputs.
- Debugging failed runs or unexpected results.
- Plotting and figure updates.
- CI or pipeline fixes when existing checks fail.

**Hard Rules**
- Reproducibility: results must be reproducible from committed code and config without manual steps.
- Determinism: set and document random seeds when randomness is used.
- Configuration: use existing config files and parameters rather than hardcoding values.
- Data paths: reference data through configured paths; never embed absolute user-specific paths.
- Outputs: write derived outputs to designated output locations; never overwrite raw inputs.
- Logging: preserve existing logs and log formats; add minimal logging only when needed.
- “Never do” rules:
- Never modify or delete raw data files.
- Never silently change filtering, thresholds, or units without explicitly noting it.
- Never mix data from different runs without clear provenance.
- Never introduce new tools or workflows not already present in the repo.

**How I Want You To Work**
- Propose a plan before coding when changes span multiple files, affect data correctness, or alter outputs.
- Implement directly for small, local fixes or one-file edits.
- Keep diffs minimal and scoped; avoid formatting-only churn.
- State assumptions and risks explicitly when inputs or behavior are unclear.

**Verification And Acceptance**
- “Done” means code changes plus a clear statement of how correctness was validated.
- Validate by running existing scripts/tests or by providing a concise, reproducible manual check.
- If checks fail, report the failure, likely cause, and next steps; do not mask or bypass failures.

**Accumulated Knowledge**
- Add new rules here as short bullets when we learn a specific constraint or mistake to avoid.
- Keep each entry concrete and testable.
