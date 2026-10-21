# DATAFLOW_v3 → mingo-computing: Repository Rename and Path-Decoupling Migration Plan

**Status:** planning and migration document.  
**Target repository name:** `mingo-computing`  
**Current repository:** `csoneira/DATAFLOW_v3`  
**Current production checkout:** `/home/mingo/DATAFLOW_v3`  
**Target production checkout:** `/home/mingo/mingo-computing`  
**Primary rule:** **the codebase must cease to depend on the literal checkout name `DATAFLOW_v3` before either the GitHub repository or the production checkout is renamed.**

This document is intentionally conservative. `DATAFLOW_v3` is not just a GitHub label today. It appears in active runtime paths, setup/recovery logic, cron-related tooling, simulation orchestration, analysis scripts, YAML configuration, shell launchers, documentation, historical notebooks, tests, and internal naming. A safe migration therefore consists of two conceptually separate operations:

1. **Make the software location-independent.**
2. **Only after that is proven, rename the public repository and production checkout.**

The migration must not be implemented as a global `sed`/search-and-replace operation.

---

## 1. Objective

Rename the repository from:

```text
csoneira/DATAFLOW_v3
```

to:

```text
csoneira/mingo-computing
```

and, after the codebase is proven relocation-safe, rename the canonical analysis-PC checkout from:

```text
/home/mingo/DATAFLOW_v3
```

to:

```text
/home/mingo/mingo-computing
```

without breaking:

- real-data ingestion;
- analysis/reconstruction stages;
- reprocessing;
- calibration and QA;
- simulation and the digital twin;
- simulation orchestration and maintenance;
- cron execution;
- production gating;
- process/lock management;
- Telegram monitoring/control tooling;
- log handling;
- file-flow tracking;
- remote data transfer;
- station-script deployment;
- remote log backup;
- analysis-PC recovery/setup;
- SSH integration;
- tmux monitoring;
- documentation deployment;
- CI tests;
- reproducible environment creation.

The final state should make the repository basename an implementation detail. A future clone should be capable of living at, for example,

```text
~/mingo-computing
~/src/mingo-computing
/tmp/mingo-computing-relocation-test
```

without scientific or operational code assuming a particular basename.

---

## 2. Non-goals

This migration is **not** a general internal rebranding exercise.

The following changes are specifically **not required** for the repository rename and should not be mixed into the first migration unless a concrete technical dependency demands them:

- renaming the environment variable `DATAFLOW_ROOT`;
- renaming every function, script, tmux socket, log label, or internal object containing `dataflow`;
- renaming `dataflow.sh`;
- renaming `dataflow_bashrc.sh`;
- renaming `start-dataflow-tmux`;
- renaming the `dataflow` tmux socket;
- renaming old scientific notebook text that accurately records historical paths;
- rewriting old migration fixtures whose purpose is to recognize legacy `DATAFLOW_v3` installations;
- renaming the station deployment path `/home/rpcuser/station_automation_scripts`;
- restructuring the repository;
- changing scientific algorithms, file formats, detector conventions, data products, or simulation physics.

`DATAFLOW_ROOT` is an internal interface. It can remain after the repository becomes `mingo-computing`. Removing a hard-coded **path dependency** and renaming every historical/internal symbol are different tasks.

---

## 3. Safety invariants

The following invariants apply throughout the migration.

1. **No GitHub rename and no local `mv DATAFLOW_v3 mingo-computing` until the relocation test passes.**
2. **No global replacement of `DATAFLOW_v3`.** Every occurrence must first be classified.
3. **Production cron must be gated closed before the production checkout is moved.** Closing the gate does not terminate already-running jobs, so active processes must also be inspected.
4. **The existing repository history must remain intact.** This is a rename, not a new repository.
5. **The `.venv` must be treated as disposable after moving the checkout.** A Python virtual environment may contain absolute interpreter/shebang paths and should be recreated at the target location.
6. **Old-path recognition must survive long enough to remove old machine configuration.** In particular, migration logic that recognizes `DATAFLOW_v3` in an existing crontab is intentional compatibility code and should not be blindly removed.
7. **Historical notebooks should normally remain historical.** If a notebook says a file was at `/home/mingo/DATAFLOW_v3/...` at the time, changing it may falsify provenance rather than improve the migration.
8. **Secrets must never be printed while auditing external machine configuration.** Search filenames and configuration references, not credential contents.
9. **Station-side automation must not be disrupted merely to rename the central repository.** Its deployed destination is already independent of the central checkout basename.
10. **Every phase must leave a recoverable state and should be committed separately.**

---

## 4. Audit basis and present architecture

A GitHub code search performed before writing this plan found **438 tracked files matching the literal string `DATAFLOW_v3`** on the current default branch. This is a file-level search count, not an occurrence count. A large fraction are script headers (`DATAFLOW_v3 Script Header v1` / `Owner: DATAFLOW_v3 contributors`), documentation, comments, or historical notebooks. These are not equivalent to runtime path dependencies.

The repository already contains the beginning of a correct location-independent architecture:

### 4.1 Shell root discovery already exists

`CONFIG/CONTENTS/environment/dataflow.sh` derives `DATAFLOW_ROOT` from its own location rather than from `$HOME/DATAFLOW_v3`:

```bash
export DATAFLOW_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd -P)"
```

This is the preferred pattern for scripts that can source the central environment.

### 4.2 Python root discovery already exists

`MINGO_ANALYSIS/MINGO_ANALYSIS_SCRIPTS/common/path_config.py` contains repository-root discovery based on `__file__`, and exposes `get_repo_root()` and related helpers.

However, `CONFIG/config_paths.yaml` currently overrides the dynamically discovered basename with:

```yaml
home_path: "~"
repo_name: DATAFLOW_v3
```

That override defeats relocation and must be redesigned or removed before the rename.

### 4.3 Cron is mostly already relocation-friendly

`CONFIG/CONTENTS/cron/analysis_pc.crontab` uses placeholders and environment variables such as:

```text
DATAFLOW_ROOT=@REPO_ROOT@
PYTHON_BIN=@REPO_ROOT@/.venv/bin/python3
MASTER_STAGE_ROOT=@REPO_ROOT@/MINGO_ANALYSIS/MINGO_ANALYSIS_SCRIPTS/STAGES
```

`CONFIG/SETUP/recovery.py` renders `@REPO_ROOT@` from the checkout in which it is executing. This is good architecture and should be preserved.

### 4.4 The setup system currently enforces the old basename

`CONFIG/SETUP/recovery.py` currently contains an explicit production guard equivalent to:

```python
if args.apply and ROOT != HOME / "DATAFLOW_v3":
    raise RuntimeError(...)
```

and `doctor()` also checks that the checkout is `$HOME/DATAFLOW_v3`.

This is currently a deliberate safety restriction because legacy paths still exist. The restriction must only be relaxed **after** the legacy runtime paths have been removed and relocation regression tests exist.

---

## 5. Classification of `DATAFLOW_v3` references

Every search result must be assigned to one of the following classes before editing.

### Class A: execution-critical basename dependency

A literal old name participates in construction of an input, output, executable, lock, log, configuration, environment, import, or runtime path.

Examples:

```bash
BASE_DIR="$HOME/DATAFLOW_v3"
```

```python
ROOT = Path.home() / "DATAFLOW_v3"
```

```python
os.path.expanduser("~/DATAFLOW_v3/...")
```

These references **must be eliminated before rename**.

### Class B: active configuration dependency

A checked-in YAML/CONF/configuration field resolves to `~/DATAFLOW_v3/...` or otherwise selects the repository by basename.

These references **must be eliminated or made repository-relative before rename**.

### Class C: machine-integration dependency

The reference is not necessarily stored in Git after setup, but has been written into the machine, for example:

- `~/.bashrc`;
- `~/.local/bin/start-dataflow-tmux`;
- installed user crontab;
- tmux launcher configuration;
- systemd user/system units, if any;
- local helper scripts or symlinks outside the repository.

These must be inventoried before the move and regenerated/updated after the move.

### Class D: current public branding or repository URL

Examples:

- clone instructions;
- README title;
- MkDocs `site_name`;
- GitHub repository URL;
- GitHub Pages project URL.

These should be updated **at the actual rename stage**, not used as a substitute for path decoupling.

### Class E: intentional legacy compatibility

Examples include a test fixture that contains an old cron line so that migration code can recognize and remove it.

These references may need to **remain `DATAFLOW_v3` after the rename**.

### Class F: historical/provenance text

Research notebooks and dated records containing `/home/mingo/DATAFLOW_v3/...` may describe the historical state correctly. They should generally remain untouched unless they are intended to be live instructions or live links.

### Class G: cosmetic/internal vocabulary

Examples:

- `DATAFLOW_v3 Script Header v1`;
- `Owner: DATAFLOW_v3 contributors`;
- `DATAFLOW_ROOT`;
- `dataflow.sh`;
- the `dataflow` tmux socket;
- human-readable log labels.

These do **not** block the rename. They can be cleaned up later as a separate refactor.

---

## 6. Known execution/configuration candidates found by the pre-migration audit

The following list is a **starting inventory, not the final authority**. Codex must repeat the search locally with `git grep`/`rg`, because local search is more complete than remote snippets and can distinguish tracked code from generated files.

### 6.1 Setup, recovery, and machine integration

Inspect carefully:

- `CONFIG/SETUP/recovery.py`
  - explicit `$HOME/DATAFLOW_v3` production restriction;
  - doctor/canonical-checkout check;
  - legacy crontab recognition;
  - `SAFE_DATAFLOW_v3` handling;
  - generated `.bashrc`, launcher, cron, SSH, and runtime paths.
- `CONFIG/config_paths.yaml`
  - `repo_name: DATAFLOW_v3` currently forces a basename.
- `CONFIG/add_to_tmux.info`
  - contains `$HOME/DATAFLOW_v3/.../start-dataflow-tmux`.
- `CONFIG/CONTENTS/environment/dataflow.sh`
  - currently **good**: derives the root from its own location. Preserve this behavior.
- `CONFIG/CONTENTS/shell/dataflow_bashrc.sh`
  - currently **good**: sources environment configuration relative to itself.
- `CONFIG/CONTENTS/cron/analysis_pc.crontab`
  - mostly good through `@REPO_ROOT@` / `$DATAFLOW_ROOT`;
  - contains `SAFE_DATAFLOW_v3` backup naming, which is a separate migration decision.
- `CONFIG/SETUP/tests/test_recovery.py`
  - contains at least one deliberate legacy `DATAFLOW_v3` cron fixture; do not destroy the compatibility test.
- `CONFIG/SETUP/README.md`
  - clone and setup instructions require branding/path updates at the rename stage.

### 6.2 Digital-twin/orchestration shell code

Known candidates include:

- `MINGO_DIGITAL_TWIN/ORCHESTRATOR/core/run_step_final_cron.sh`
- `MINGO_DIGITAL_TWIN/ORCHESTRATOR/core/run_step.sh`
- `MINGO_DIGITAL_TWIN/ORCHESTRATOR/core/run_main_simulation_cycle.sh`
- `MINGO_DIGITAL_TWIN/ORCHESTRATOR/maintenance/stop_execution.sh`
- `MINGO_DIGITAL_TWIN/ANCILLARY/restart_simulation_pipeline_totally.sh`
- `MINGO_DIGITAL_TWIN/CONFIG_FILES/ORCHESTRATOR/sim_main_pipeline_frequency.conf` (at least a documented/default old path)

Known examples include:

```bash
ROOT_DIR="${DATAFLOW_ROOT:-${HOME}/DATAFLOW_v3}"
```

```bash
BASE_DIR="$HOME/DATAFLOW_v3"
```

and a hard-coded final-step lock under `$HOME/DATAFLOW_v3` even in code that otherwise derives its runtime root dynamically.

### 6.3 Operations shell code

Known candidates include:

- `OPERATIONS/OPERATIONS_SCRIPTS/ORCHESTRATION/CRON_RUN_LOGGED/cron_run_logged.sh`
- `OPERATIONS/OPERATIONS_SCRIPTS/ORCHESTRATION/RESOURCE_GATE/resource_gate.sh`
- `OPERATIONS/OPERATIONS_SCRIPTS/ORCHESTRATION/ENSURE_CRON_LOG_PATHS/ensure_cron_log_paths.sh`
- `OPERATIONS/OPERATIONS_SCRIPTS/ORCHESTRATION/WATCHDOG_PROCESS_COUNTS/watchdog_process_counts.sh`
- `OPERATIONS/OPERATIONS_SCRIPTS/MAINTENANCE/ERASE_ALL_CRON_LOGS/erase_cron_logs.sh`
- `OPERATIONS/OPERATIONS_SCRIPTS/DATA_MAINTENANCE/REMOTE_LOG_BACKUP/run_log_backup.sh`
- `OPERATIONS/OPERATIONS_SCRIPTS/DATA_MAINTENANCE/COMPLETED_TO_UNPROCESSED/move_completed_to_unprocessed.sh`
- `OPERATIONS/OPERATIONS_SCRIPTS/DATA_MAINTENANCE/FILE_FLOW_TRACKER/watch_file_flow.sh`
- `OPERATIONS/OPERATIONS_SCRIPTS/OBSERVABILITY/EXECUTION_TOP_OFFENDERS/execution_top_offenders.sh`
- `OPERATIONS/OPERATIONS_SCRIPTS/NOTIFICATIONS/TELEGRAM_BOT/persistent_telegram_bot_check.sh`
- `OPERATIONS/OPERATIONS_SCRIPTS/NOTIFICATIONS/TELEGRAM_BOT/kill_bot_and_restart.sh`
- `OPERATIONS/OPERATIONS_SCRIPTS/MAINTENANCE/CLEANERS/interstep_simulation_cleaner.sh`

### 6.4 Analysis/reprocessing shell code

Known candidates include:

- `MINGO_ANALYSIS/MINGO_ANALYSIS_SCRIPTS/ANCILLARY/PLOTTERS/create_metadata_plots.sh`
- `MINGO_ANALYSIS/MINGO_ANALYSIS_SCRIPTS/STAGES/STAGE_0/NEW_FILES/bring_data_and_config_files.sh`
- `MINGO_ANALYSIS/MINGO_ANALYSIS_SCRIPTS/STAGES/STAGE_1/LOG_DATA/STEP_1/lab_logs_bring_and_clean.sh`
- `MINGO_ANALYSIS/MINGO_ANALYSIS_SCRIPTS/STAGES/STAGE_1/EVENT_DATA/STEP_1/guide_raw_to_corrected.sh`
- `MINGO_ANALYSIS/MINGO_ANALYSIS_SCRIPTS/STAGES/STAGE_2/EVENT_DATA/STEP_1_ACCUMULATION/guide_corrected_to_accumulated.sh`
- `MINGO_ANALYSIS/MINGO_ANALYSIS_SCRIPTS/STAGES/STAGE_2/EVENT_DATA/STEP_2_DAILY_EVENT_DATA/guide_accumulated_to_joined.sh`
- `MINGO_ANALYSIS/MINGO_ANALYSIS_SCRIPTS/STAGES/STAGE_0/REPROCESSING/STEP_0/prepare_reprocessing_metadata.sh`
- `MINGO_ANALYSIS/MINGO_ANALYSIS_SCRIPTS/STAGES/STAGE_0/REPROCESSING/STEP_1/bring_reprocessing_files.sh`
- `MINGO_ANALYSIS/MINGO_ANALYSIS_SCRIPTS/STAGES/STAGE_0/REPROCESSING/STEP_2/unpack_reprocessing_files.sh`
- `MINGO_ANALYSIS/MINGO_ANALYSIS_SCRIPTS/STAGES/STAGE_0/REPROCESSING/UNPACKER_ZERO_STAGE_FILES/bin/unpack.sh`

Some of these may only contain fallback or embedded Python snippets. Inspect each result rather than assuming the whole file is wrong.

### 6.5 Python code constructing paths from `Path.home()` or `~/DATAFLOW_v3`

Known candidates include:

- `MINGO_DIGITAL_TWIN/ORCHESTRATOR/maintenance/prune_step_final_params.py`
- `MINGO_DIGITAL_TWIN/ORCHESTRATOR/maintenance/sanitize_sim_runs.py`
- `MINGO_DIGITAL_TWIN/ORCHESTRATOR/maintenance/prune_completed_param_mesh_rows.py`
- `MINGO_DIGITAL_TWIN/ORCHESTRATOR/maintenance/repair_param_mesh_step_ids.py`
- `MINGO_DIGITAL_TWIN/ORCHESTRATOR/diagnostics/size_and_expected_report.py`
- `OPERATIONS/OPERATIONS_SCRIPTS/ORCHESTRATION/SOLVE_STALE_LOCKS/solve_stale_locks.py`
- `OPERATIONS/OPERATIONS_SCRIPTS/DATA_MAINTENANCE/REMOTE_LOG_BACKUP/prune_log_backup_history.py`
- `MINGO_ANALYSIS/MINGO_ANALYSIS_SCRIPTS/common/execution_logger.py`
- `MINGO_ANALYSIS/MINGO_ANALYSIS_SCRIPTS/common/step1_shared.py`
- `MINGO_ANALYSIS/MINGO_ANALYSIS_SCRIPTS/common/reprocessing_qa_retry.py`
- `MINGO_ANALYSIS/MINGO_ANALYSIS_SCRIPTS/STAGES/STAGE_2/EVENT_DATA/STEP_2_DAILY_EVENT_DATA/TASK_1/accumulated_distributor.py`
- `MINGO_ANALYSIS/MINGO_ANALYSIS_SCRIPTS/STAGES/STAGE_2/EVENT_DATA/STEP_2_DAILY_EVENT_DATA/TASK_2/distributed_joiner.py`
- `MINGO_ANALYSIS/MINGO_ANALYSIS_SCRIPTS/STAGES/STAGE_0/SIMULATION/ingest_simulated_station_data.py`
- `MINGO_ANALYSIS/MINGO_ANALYSIS_SCRIPTS/STAGES/STAGE_1/EVENT_DATA/STEP_1/TASK_1/script_1_raw_to_clean.py`

Plotter/fallback candidates found by search include:

- `MINGO_ANALYSIS/MINGO_ANALYSIS_SCRIPTS/ANCILLARY/PLOTTERS/METADATA/NAIVE_EFFICIENCY/naive_efficiency_metadata_plotter.py`
- `MINGO_ANALYSIS/MINGO_ANALYSIS_SCRIPTS/ANCILLARY/PLOTTERS/RATES/rates_metadata_plotter.py`
- `MINGO_ANALYSIS/MINGO_ANALYSIS_SCRIPTS/ANCILLARY/PLOTTERS/TASK_4/task4_chi2_four_plane_plotter.py`
- `MINGO_ANALYSIS/MINGO_ANALYSIS_SCRIPTS/ANCILLARY/PLOTTERS/METADATA/FILTER/filter_metadata_plotter.py`
- `MINGO_ANALYSIS/MINGO_ANALYSIS_SCRIPTS/ANCILLARY/PLOTTERS/METADATA/EFFICIENCIES/efficiency_metadata_plotter.py`
- `MINGO_ANALYSIS/MINGO_ANALYSIS_SCRIPTS/ANCILLARY/PLOTTERS/METADATA/EXECUTION/execution_metadata_plotter.py`
- `MINGO_ANALYSIS/MINGO_ANALYSIS_SCRIPTS/ANCILLARY/PLOTTERS/METADATA/TRIGGER_RATES/trigger_rate_metadata_plotter.py`
- `MINGO_ANALYSIS/MINGO_ANALYSIS_SCRIPTS/ANCILLARY/PLOTTERS/DEFINITIVE_EXECUTION/definitive_execution_plotter.py`
- `MINGO_ANALYSIS/MINGO_ANALYSIS_SCRIPTS/ANCILLARY/PLOTTERS/METADATA/NOISE_CONTROL/noise_control_metadata_plotter.py`
- `MINGO_ANALYSIS/MINGO_ANALYSIS_SCRIPTS/ANCILLARY/PLOTTERS/METADATA/EFFICIENCIES_THREE_TO_FOUR/efficiencies_three_to_four_plotter.py`
- `MINGO_ANALYSIS/MINGO_ANALYSIS_SCRIPTS/ANCILLARY/PLOTTERS/SIMULATED_DATA_EVOLUTION/simulated_data_evolution_plotter.py`

Several of these already search parent directories and use `$HOME/DATAFLOW_v3` only as a final fallback. The correct migration is to make their successful path discovery authoritative, or to use the shared path helper, rather than merely changing that fallback to `$HOME/mingo-computing`.

### 6.6 Active YAML/configuration paths

Known active candidates include:

- `MINGO_ANALYSIS/MINGO_ANALYSIS_SCRIPTS/CONFIG_FILES/STAGE_0/REPROCESSING/config_reprocessing.yaml`
  - contains a `qa_retry.source_csv` under `~/DATAFLOW_v3/...`.
- `MINGO_ANALYSIS/MINGO_ANALYSIS_SCRIPTS/CONFIG_FILES/STAGE_1/LOG_DATA/STEP_2/config_step_2.yaml`
  - contains `backup_restore.local_log_backup_root: ~/DATAFLOW_v3/...`.

Codex must search **all** YAML/JSON/TOML/CONF/INI-like files, not only these examples.

### 6.7 Absolute `/home/mingo/DATAFLOW_v3` candidates outside the main pipeline

Remote search also surfaced absolute old-checkout paths in ancillary/comparison/dictionary code. These must be inspected and classified as executable, live documentation, or historical material. Relevant paths include:

- `MINGO_DIGITAL_TWIN/ANCILLARY/CALCULATIONS/calculate_limits.py`
- `MINGO_SIMULATION_VS_SIMULATION/compare_sidecar_with_simulated_data.py`
- `FOR_MINGO_SYSTEMS/station_automation_scripts/bot_scripts/minitrasgo_bot.py`
- `MINGO_ANALYSIS/MINGO_ANALYSIS_SCRIPTS/ANCILLARY/SIMULATION_VS_DATA_AND_VALIDATION/common/file_pairing.py`
- `MINGO_ANALYSIS/MINGO_ANALYSIS_SCRIPTS/ANCILLARY/SIMULATION_VS_DATA_AND_VALIDATION/TASK_4/FILEvFILE/ANGULAR_DISTRIBUTION/angular_check.py`
- `MINGO_ANALYSIS/MINGO_ANALYSIS_SCRIPTS/ANCILLARY/SIMULATION_VS_DATA_AND_VALIDATION/TASK_5/METADATA_COMPARISON/TRIGGER_RATES/trigger_rate_similarity.py`
- `MINGO_DICTIONARY_CREATION_AND_TEST/ONLY_LUT/trigger_rate_vs_flux.py`
- `MINGO_DICTIONARY_CREATION_AND_TEST/ONLY_LUT_FINAL/creator.py`
- `MINGO_DICTIONARY_CREATION_AND_TEST/ONLY_LUT_FOUR_THREE_TWO_TOTAL/lut_creator.py`
- `MINGO_DICTIONARY_CREATION_AND_TEST/DEFINITIVE_METHOD/simple_common.py`
- `MINGO_DICTIONARY_CREATION_AND_TEST/ONLY_LUT_FOUR_PLANE/four_plane_trigger_vs_flux.py`
- `MINGO_DICTIONARY_CREATION_AND_TEST/AN_EVEN_EASIER_VARIATION_ADVANCED/step_3_build_rate_to_flux_calibration.py`

Documentation under the same search includes `GATES_TUTORIAL.md`, several dictionary-method READMEs, digital-twin runbooks, QA documentation, and dated notebooks. These require classification, not automatic replacement.

### 6.8 Public repository identity and documentation

Known current-name references include:

- root `README.md`;
- `CONFIG/SETUP/README.md`;
- `DOCUMENTATION/README.md`;
- `DOCUMENTATION/mkdocs.yml`.

At the time of this audit, `DOCUMENTATION/mkdocs.yml` identifies:

```yaml
site_name: DATAFLOW_v3 Documentation
site_url: https://csoneira.github.io/DATAFLOW_v3/
repo_name: csoneira/DATAFLOW_v3
repo_url: https://github.com/csoneira/DATAFLOW_v3
```

These are **branding/URL changes**, not preconditions for path independence. Update them during the actual public rename stage.

### 6.9 `SAFE_DATAFLOW_v3`

Known references exist in:

- `CONFIG/safe_dataflow_rsync_excludes.list`;
- `CONFIG/CONTENTS/cron/analysis_pc.crontab`;
- `CONFIG/SETUP/recovery.py`;
- `OPERATIONS/OPERATIONS_SCRIPTS/MAINTENANCE/CLEANERS/top_large_dirs.sh`;
- `OPERATIONS/OPERATIONS_SCRIPTS/MAINTENANCE/CLEANERS/clean_dataflow.sh`.

This backup-directory name is not the same problem as a hard-coded repository checkout. It is an external state/storage name. The first migration should preferably **leave `SAFE_DATAFLOW_v3` intact** so that repository relocation and backup-directory migration are not combined. It can be renamed later in a dedicated operational migration if desired.

---

## 7. Required local exhaustive audit before editing

Remote search is useful for planning, but the implementation must begin by auditing the actual local checkout.

From `/home/mingo/DATAFLOW_v3`, record the baseline:

```bash
pwd
git status --short
git branch --show-current
git rev-parse HEAD
git remote -v
```

Do not proceed with a dirty working tree without first understanding every change.

### 7.1 Tracked-file inventory

Run:

```bash
git grep -nI 'DATAFLOW_v3' > /tmp/dataflow-v3-all-tracked.txt || true
git grep -nI 'SAFE_DATAFLOW_v3' > /tmp/dataflow-v3-safe-backup.txt || true
git grep -nI 'github.com/csoneira/DATAFLOW_v3' > /tmp/dataflow-v3-github-url.txt || true
git grep -nI 'csoneira.github.io/DATAFLOW_v3' > /tmp/dataflow-v3-pages-url.txt || true
```

Then search specifically for path-building forms:

```bash
git grep -nIE '(\$HOME|\$\{HOME\}|~)/DATAFLOW_v3|/home/[^/]+/DATAFLOW_v3' \
  > /tmp/dataflow-v3-hardcoded-home-paths.txt || true

git grep -nIE 'Path\.home\(\).*DATAFLOW_v3|expanduser\([^\n]*DATAFLOW_v3' \
  > /tmp/dataflow-v3-python-home-paths.txt || true

git grep -nIE 'repo_name:[[:space:]]*DATAFLOW_v3|DATAFLOW_ROOT[^\n]*DATAFLOW_v3' \
  > /tmp/dataflow-v3-config-and-fallbacks.txt || true
```

Also use ripgrep across the checkout so that relevant ignored/local operational files are not silently missed, while excluding `.git` itself:

```bash
rg -n --hidden --no-messages \
  --glob '!.git/**' \
  'DATAFLOW_v3|SAFE_DATAFLOW_v3|csoneira/DATAFLOW_v3|csoneira\.github\.io/DATAFLOW_v3' \
  . > /tmp/dataflow-v3-rg-all.txt || true
```

### 7.2 Produce a classification table before modifying code

Codex should create or maintain a temporary table with at least:

```text
file | line | exact pattern | class A-G | runtime? | proposed action | test covering action
```

The implementation should not begin with bulk replacement. First identify which references are actually path semantics.

---

## 8. Target path-resolution architecture

The migration should converge on the following architecture.

### 8.1 Shell

Preferred order:

1. If the script is part of the configured production environment, source the central environment file using a path derived from the script location.
2. Use the resulting `DATAFLOW_ROOT` everywhere.
3. For self-contained scripts that cannot source it, derive the repository root from `${BASH_SOURCE[0]}` and the repository structure.
4. Do **not** use `$HOME/<repository-name>` as a default.

Good:

```bash
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# derive or source the repository root from a structural location
```

Bad:

```bash
ROOT="$HOME/DATAFLOW_v3"
```

Also bad after the migration:

```bash
ROOT="$HOME/mingo-computing"
```

The objective is not to replace one hard-coded basename with another.

### 8.2 Python

For analysis code that already depends on the common package, prefer the existing path API:

```python
from MINGO_ANALYSIS.MINGO_ANALYSIS_SCRIPTS.common.path_config import get_repo_root

repo_root = get_repo_root()
```

For digital-twin/operations components where importing the analysis package would create an undesirable cross-component dependency, use a component-local root-discovery helper based on `Path(__file__).resolve()` and stable repository markers/directories.

Do not create multiple new ad-hoc `Path(...).parents[N]` implementations without justification. Reuse an existing helper where the dependency direction is sensible.

### 8.3 Configuration files

Checked-in configuration should preferably contain:

- repository-relative paths; or
- values expanded relative to a known configured root; or
- component-relative paths interpreted by the loader.

It should not contain:

```yaml
something: ~/DATAFLOW_v3/...
```

If a loader currently assumes arbitrary absolute strings, modify the loader carefully so a repository-relative representation is unambiguous and backward compatible.

### 8.4 Setup/recovery

`CONFIG/SETUP/recovery.py` should continue deriving:

```python
ROOT = Path(__file__).resolve().parents[2]
```

or an equally structural equivalent.

The safety check should validate properties that actually matter, such as:

- not running as root;
- real user `$HOME` consistency;
- shell-safe path if legacy unquoted code still exists;
- expected repository structure/sentinels;
- required files present;
- writable runtime directories where appropriate.

It should **not** validate that `ROOT.name == "DATAFLOW_v3"`.

### 8.5 Legacy recognition

Functions responsible for removing old machine state should continue to recognize the old basename. For example, `split_cron()` currently recognizes old `DATAFLOW_v3` entries. That is migration logic, not a location dependency.

Add comments/tests explicitly identifying these remaining literals as intentional legacy compatibility.

---

## 9. Phase 0: freeze and baseline

**No rename occurs in this phase.**

1. Confirm branch, HEAD, remotes, and working-tree state.
2. Record existing cron:

   ```bash
   crontab -l > /tmp/crontab.before-mingo-computing-rename.txt
   ```

3. Record managed tmux/process state without modifying it.
4. Run the exhaustive search in Section 7.
5. Run the existing tests and record baseline failures separately from migration-caused failures:

   ```bash
   python3 -m unittest discover -s CONFIG/SETUP/tests -v
   ```

   and, using the repository environment where required:

   ```bash
   .venv/bin/python -m pytest -q MINGO_DIGITAL_TWIN/MASTER_STEPS/tests
   ```

6. Run:

   ```bash
   bash CONFIG/SETUP/setup.sh --dry-run
   bash CONFIG/SETUP/setup.sh --check
   ```

7. Do not “fix” unrelated pre-existing scientific/test failures as part of this rename. Document them and distinguish them from rename regressions.

**Gate 0:** baseline recorded and every literal-path candidate classified.

---

## 10. Phase 1: introduce/standardize location-independent root resolution

Still **no repository rename and no local-directory rename**.

1. Preserve `CONFIG/CONTENTS/environment/dataflow.sh` as the shell source of truth.
2. Preserve/enhance `common/path_config.py` as the analysis-side Python source of truth.
3. Remove `CONFIG/config_paths.yaml` behavior that forces `repo_name: DATAFLOW_v3`. Prefer discovered checkout root by default.
4. Establish a clean root-discovery mechanism for components that cannot import the analysis helper.
5. Add regression tests for root discovery before converting large numbers of callers.

A useful test should prove that path resolution uses the file/check-out location rather than `$HOME/DATAFLOW_v3`.

**Gate 1:** central path resolution is independently testable and does not encode the old basename.

---

## 11. Phase 2: remove execution-critical old-basename dependencies

Still **no repository rename and no local-directory rename**.

Process Class A and B references methodically.

### 11.1 Setup and machine configuration

- Remove the basename restriction in `recovery.py` only after root-resolution tests exist.
- Update `doctor()` to validate the checkout structurally, not by name.
- Preserve old-name matching in legacy cleanup logic.
- Update `CONFIG/add_to_tmux.info` to use a location-independent mechanism, or deprecate it if the generated launcher is the canonical path.

### 11.2 Shell runtime

For every active shell script that contains `$HOME/DATAFLOW_v3`, `${HOME}/DATAFLOW_v3`, or an equivalent literal:

- source/derive the root;
- use `DATAFLOW_ROOT` or an equivalent structural root;
- remove the basename fallback;
- keep CLI overrides when they are meaningful.

### 11.3 Python runtime

Replace all active constructs such as:

```python
Path.home() / "DATAFLOW_v3"
Path(os.path.expanduser("~/DATAFLOW_v3/..."))
```

with the appropriate shared/structural root.

### 11.4 Active configuration

Replace repository-absolute paths in active YAML/configuration with repository-relative or root-expanded equivalents and update loaders/tests as required.

### 11.5 Ancillary tools

Do not ignore dictionary/comparison/calculation scripts merely because they are not cron-critical. If they are intended to run from this repository and their old path is executable configuration rather than historical documentation, make them relocation-safe too.

**Gate 2:** no active executable/configuration path requires the basename `DATAFLOW_v3`.

---

## 12. Phase 3: add a permanent anti-regression test

Before renaming anything, add a test that prevents reintroduction of old-basename runtime dependencies.

The test should scan relevant executable/configuration source files for forbidden forms such as:

```text
$HOME/DATAFLOW_v3
${HOME}/DATAFLOW_v3
~/DATAFLOW_v3
/home/<user>/DATAFLOW_v3
Path.home() / "DATAFLOW_v3"
expanduser("~/DATAFLOW_v3...")
repo_name: DATAFLOW_v3     # when used as active path configuration
```

The test must use an **explicit allowlist** for intentional cases, for example:

- legacy migration fixtures in `CONFIG/SETUP/tests/test_recovery.py`;
- legacy-recognition logic in `recovery.py`;
- historical notebook directories if they are scanned at all;
- migration documentation such as this file.

Do not write a test asserting that the string `DATAFLOW_v3` is absent everywhere. That would incorrectly reject historical documentation, headers, and compatibility code.

**Gate 3:** the anti-regression test passes and clearly documents every remaining allowed old-name literal.

---

## 13. Phase 4: prove relocation before the real rename

This phase is the decisive safety proof.

Create a detached temporary worktree whose directory is **not named `DATAFLOW_v3`**:

```bash
TMP_RENAME_TEST="$(mktemp -d /tmp/mingo-computing-relocation-test.XXXXXX)"
git worktree add --detach "$TMP_RENAME_TEST" HEAD
```

Do **not** create a `DATAFLOW_v3` symlink for this test. The point is to prove that the old basename is unnecessary.

From the temporary checkout:

1. run setup dry-run;
2. run setup/recovery unit tests;
3. run path/config unit tests;
4. run digital-twin unit tests using an available interpreter with the required dependencies;
5. run `bash -n` on modified shell scripts;
6. run lightweight `--help`, configuration, or dry-run modes where available;
7. verify that all computed paths resolve inside the temporary checkout rather than `/home/mingo/DATAFLOW_v3`.

At minimum:

```bash
cd "$TMP_RENAME_TEST"
bash CONFIG/SETUP/setup.sh --dry-run
python3 -m unittest discover -s CONFIG/SETUP/tests -v
```

Use the current repository interpreter explicitly for dependency-rich tests if necessary, but ensure imports/code under test come from the temporary worktree.

After testing:

```bash
cd /home/mingo/DATAFLOW_v3
git worktree remove "$TMP_RENAME_TEST"
```

**Gate 4, mandatory:** a checkout whose basename is not `DATAFLOW_v3` passes the relevant test suite **without any compatibility symlink**.

If Gate 4 fails, the public/local rename is forbidden. Fix the dependency and repeat.

---

## 14. Phase 5: audit references outside the Git repository

The repository setup writes absolute paths into machine state, so repository tests alone are insufficient.

Perform read-only searches of likely integration locations. Avoid reading/printing secret contents.

Inspect at least:

```text
~/.bashrc
~/.profile
~/.bash_profile
~/.local/bin/
~/.config/systemd/user/
~/.tmux.conf
installed user crontab
/etc/systemd/system/      (read-only where permitted)
/etc/cron.d/              (read-only where permitted)
```

The setup system is known to write a repository-specific source path into `.bashrc` and a repository-specific launcher into `~/.local/bin/start-dataflow-tmux`; these will need regeneration after the move.

For SSH configuration, search only configuration files and filenames. Do not print private-key material.

Also search for consumers of the old GitHub URL in local automation or other maintained repositories if practical.

**Gate 5:** all external machine references are inventoried and a post-move regeneration/update path is known.

---

## 15. Phase 6: prepare the public identity changes, but do not execute them early

Only after Gates 0-5 pass, prepare the branding diff for:

- root README title: `MINGO Computing`;
- repository references: `csoneira/mingo-computing`;
- clone instructions: `~/mingo-computing`;
- `DOCUMENTATION/mkdocs.yml`:

  ```yaml
  site_name: MINGO Computing Documentation
  site_url: https://csoneira.github.io/mingo-computing/
  repo_name: csoneira/mingo-computing
  repo_url: https://github.com/csoneira/mingo-computing
  ```

- current operational documentation that is intended as live instruction.

Do not rewrite dated notebooks merely for appearance.

GitHub documentation states that ordinary repository web/Git operations are redirected after a repository rename, but **project-site (GitHub Pages) URLs are an exception** and need explicit attention. GitHub also warns not to later create a new repository using the old name if redirects are to be preserved.

Official references:

- GitHub repository rename documentation: <https://docs.github.com/en/repositories/creating-and-managing-repositories/renaming-a-repository>
- GitHub CLI `gh repo rename`: <https://cli.github.com/manual/gh_repo_rename>

---

## 16. Phase 7: controlled production rename

This is the first phase in which the actual names change.

### 16.1 Preconditions

Before continuing:

- working tree clean;
- path-independence changes committed;
- tests passing or baseline-known exceptions documented;
- relocation worktree test passing;
- latest changes pushed;
- external machine references inventoried;
- a known rollback commit/HEAD recorded.

### 16.2 Close the production gate

`CONFIG/SETUP/run-job.sh` refuses to execute managed jobs if:

```text
OPERATIONS/OPERATIONS_RUNTIME/SETUP/production-enabled
```

is absent.

Close the gate in a reversible way, for example by moving the marker to a temporary pause name rather than deleting its contents blindly. Then verify managed cron invocations no longer start work.

Closing the gate does **not** terminate jobs already running. Inspect active analysis/simulation/transfer processes and allow them to finish or stop them using the repository's existing controlled mechanisms. Do not use broad process killing without confirming ownership.

Record another copy of `crontab -l` immediately before the move.

### 16.3 Rename GitHub repository

Using the authenticated GitHub CLI, the current command is:

```bash
gh repo rename -R csoneira/DATAFLOW_v3 mingo-computing
```

Use interactive confirmation unless there is a strong reason not to.

Then explicitly update the local remote even though GitHub redirects old Git remote traffic:

```bash
git remote set-url origin https://github.com/csoneira/mingo-computing.git
git remote -v
```

### 16.4 Rename the local checkout

From the parent directory:

```bash
cd /home/mingo
mv DATAFLOW_v3 mingo-computing
cd /home/mingo/mingo-computing
```

At this point the repository must already be capable of functioning at this path. The move is not where path dependencies are fixed.

### 16.5 Optional temporary compatibility symlink

Only as an operational safety net after path independence has already been proven, an optional temporary symlink may be created:

```bash
ln -s /home/mingo/mingo-computing /home/mingo/DATAFLOW_v3
```

This symlink must **not** be used to make Gate 4 pass, because that would conceal remaining coupling. It exists only as rollback/compatibility protection for an external reference that the machine audit may have missed.

Plan to remove it after validation.

---

## 17. Phase 8: rebuild path-sensitive machine state at the new location

### 17.1 Recreate `.venv`

Do not trust the moved virtual environment as the final environment. Preserve it temporarily only if useful for diagnostics, then create a fresh `.venv` at `/home/mingo/mingo-computing/.venv` using the repository's pinned setup process.

The important property is that generated console scripts/shebangs reference the new environment.

### 17.2 Rerun setup from the new checkout

From:

```text
/home/mingo/mingo-computing
```

run first:

```bash
bash CONFIG/SETUP/setup.sh --dry-run
bash CONFIG/SETUP/setup.sh --check
```

Then, after reviewing the result:

```bash
bash CONFIG/SETUP/setup.sh --apply
```

This should regenerate/update managed external state using the new absolute root, including:

- `.bashrc` managed source block;
- `~/.local/bin/start-dataflow-tmux`;
- rendered/installed managed cron;
- repository runtime setup state;
- other setup-managed paths.

Because legacy cron cleanup still recognizes old `DATAFLOW_v3` entries, old managed paths should be replaced rather than duplicated.

### 17.3 Reopen production only through the normal setup gate

Do not manually invent a new gate state if `setup.sh --apply` is the canonical mechanism for declaring the environment ready. Let setup/doctor prove prerequisites and open the production gate according to its normal rules.

---

## 18. Phase 9: post-rename validation

Validate in this order.

### 18.1 Repository identity

```bash
pwd
git status --short
git remote -v
git rev-parse HEAD
```

Expected root:

```text
/home/mingo/mingo-computing
```

Expected origin:

```text
https://github.com/csoneira/mingo-computing.git
```

### 18.2 Path leakage

Run the same pre-migration searches again.

Any remaining old-path reference must be one of:

- intentional legacy compatibility;
- historical documentation/provenance;
- this migration document;
- optional legacy backup naming (`SAFE_DATAFLOW_v3`), if deliberately retained;
- cosmetic/header terminology.

No active executable/configuration path may depend on `/home/mingo/DATAFLOW_v3` or `~/DATAFLOW_v3`.

### 18.3 Setup and tests

Run:

```bash
bash CONFIG/SETUP/setup.sh --check
python -m unittest discover -s CONFIG/SETUP/tests -v
python -m pytest -q MINGO_DIGITAL_TWIN/MASTER_STEPS/tests
```

using the rebuilt `.venv` where appropriate.

Run shell syntax validation for every modified shell script.

### 18.4 Cron

Inspect:

```bash
crontab -l
```

Confirm:

- rendered `DATAFLOW_ROOT` points to `/home/mingo/mingo-computing`;
- no active cron command invokes `/home/mingo/DATAFLOW_v3`;
- unrelated user cron jobs were preserved;
- no duplicate managed block exists;
- expected disabled simulation entries remain disabled if that was their prior state.

### 18.5 External integration

Verify:

- `.bashrc` source path;
- `start-dataflow-tmux` launcher;
- dedicated tmux monitor;
- SSH config integration;
- Telegram bot startup/config check;
- file-flow tracker;
- remote log backup dry-run/read-only behavior;
- station deployment rsync dry-run;
- analysis/reprocessing dry-run or smallest safe real unit;
- simulation quick/unit validation;
- documentation build.

### 18.6 GitHub/Pages

Verify:

- new GitHub repository URL;
- Actions workflows after a push;
- GitHub Pages deployment at the new project-site URL;
- MkDocs repository links;
- no maintained documentation still instructs users to clone `DATAFLOW_v3`.

---

## 19. Phase 10: remove the compatibility symlink and prove it is unnecessary

If `/home/mingo/DATAFLOW_v3` was temporarily created as a symlink, remove it only after the system has run successfully from the new checkout.

Then rerun:

- setup check;
- anti-regression path tests;
- cron inspection;
- Telegram/config check;
- representative analysis job;
- representative simulation validation;
- remote sync/log backup checks.

**Final technical gate:** the production system works with **no filesystem object at `/home/mingo/DATAFLOW_v3`**.

This is the point at which basename decoupling is actually demonstrated in production.

---

## 20. What should remain named `DATAFLOW` initially

To reduce blast radius, retain these internal interfaces unless there is a separate reason to change them:

```text
DATAFLOW_ROOT
dataflow.sh
dataflow_bashrc.sh
start-dataflow-tmux
tmux -L dataflow
DATAFLOW-related internal log labels
```

Their names do not imply a filesystem path and do not prevent the repository from being called `mingo-computing`.

Likewise, keep `SAFE_DATAFLOW_v3` during the first migration unless its rename is separately planned. It is stateful backup storage, so changing it introduces data-migration and cleanup semantics that are unnecessary for the repository rename.

A later cosmetic/internal cleanup can decide whether `DATAFLOW_ROOT` should become, for example, `MINGO_COMPUTING_ROOT`, but that should be a separate commit/project after the production rename is stable.

---

## 21. References that should probably NOT be rewritten

### 21.1 Legacy cron fixture

A test like:

```python
old = '* * * * * crontab $HOME/DATAFLOW_v3/CONFIG/add_to_crontab.info\n'
```

is deliberately testing old-state cleanup. Preserve the old string and annotate it as a legacy fixture.

### 21.2 Legacy cleanup logic

If `recovery.py` recognizes `DATAFLOW_v3` to remove pre-migration managed cron, that recognition must survive the migration.

### 21.3 Historical notebooks

Dated notebooks containing old absolute paths normally represent provenance. Do not mass-edit them unless the path is explicitly presented as a current command that must remain executable.

### 21.4 This file

This migration plan must discuss the old name extensively and must therefore be on the anti-regression allowlist.

---

## 22. Commit strategy

Use small, reviewable commits. A reasonable sequence is:

1. `refactor(paths): make repository root discovery checkout-name independent`
2. `refactor(paths): remove hardcoded DATAFLOW_v3 runtime paths`
3. `test(paths): add relocated-checkout and legacy-path regression coverage`
4. `docs(rename): prepare MINGO Computing public identity`
5. actual GitHub/local rename (not itself a Git content change)
6. `docs(rename): finalize repository URLs and post-rename instructions`

Do not combine unrelated scientific changes with this migration.

Before each commit, review:

```bash
git diff --check
git diff
```

and run the tests relevant to the files modified.

---

## 23. Rollback strategy

The rename has two separable rollback dimensions.

### 23.1 Code rollback

All path-decoupling changes are ordinary Git commits and can be reverted normally if required.

### 23.2 Operational rollback after local move

If a serious production problem appears immediately after the move:

1. keep the production gate closed;
2. stop/wait for managed jobs;
3. preserve diagnostics;
4. move the checkout back to `/home/mingo/DATAFLOW_v3` if necessary;
5. restore/regenerate machine integration from the last known-good commit;
6. restore the old origin URL if the GitHub repository was also renamed back.

Do not solve rollback by copying partial directory trees between the two names. Move the single Git checkout as a unit.

Because GitHub retains repository history across a rename, the GitHub rename itself does not create a history fork.

---

## 24. Definition of done

The migration is complete only when all of the following are true:

- [ ] GitHub repository is `csoneira/mingo-computing`.
- [ ] Production checkout is `/home/mingo/mingo-computing`.
- [ ] No compatibility symlink `/home/mingo/DATAFLOW_v3` is required.
- [ ] No active executable code constructs a path using the old basename.
- [ ] No active configuration requires `~/DATAFLOW_v3`.
- [ ] `setup.sh --check` passes to the same or better level than the pre-migration baseline.
- [ ] Setup/recovery unit tests pass.
- [ ] Digital-twin unit tests pass to baseline expectations.
- [ ] Relocated-checkout regression test exists and passes.
- [ ] Anti-regression scan/test prevents new hard-coded old-basename dependencies.
- [ ] Managed cron uses the new absolute checkout root.
- [ ] Unrelated cron jobs are preserved.
- [ ] `.bashrc` and generated local launchers point to the new checkout.
- [ ] Fresh `.venv` exists at the new path.
- [ ] Telegram/config monitoring works.
- [ ] File-flow/operations monitoring works.
- [ ] Remote log backup and station deployment checks work.
- [ ] Representative real-data processing works.
- [ ] Representative simulation validation works.
- [ ] GitHub Actions run successfully under the new repository name.
- [ ] GitHub Pages is served under the new project-site URL.
- [ ] Current README/setup documentation uses `MINGO Computing` / `mingo-computing`.
- [ ] Remaining `DATAFLOW_v3` literals are explicitly classified as historical, legacy compatibility, migration documentation, backup-state naming, or cosmetic internal vocabulary.

---

## 25. Codex execution rules

When this document is handed to Codex, the following rules are mandatory:

1. Read this entire document before modifying files.
2. Inspect the actual local repository state; do not assume this audit is exhaustive.
3. Repeat all search/audit commands locally and build the classification before edits.
4. Do not rename the GitHub repository or local directory until the relocated-checkout test passes.
5. Do not perform global `DATAFLOW_v3 -> mingo-computing` replacement.
6. Do not rename `DATAFLOW_ROOT` in this migration.
7. Preserve legacy-path fixtures and migration recognition where they are intentional.
8. Preserve dated scientific provenance unless a document is clearly current operational instruction.
9. Avoid unrelated refactors.
10. Keep production safe: no uncontrolled killing, no secret output, no destructive deletion of scientific data.
11. Commit the path-decoupling work before executing the public/local rename.
12. If an ambiguous old-name reference is encountered, classify it and record the reasoning rather than guessing.
13. Continue autonomously through safe code/test phases; stop before an irreversible or externally visible action only when user authorization is genuinely required by the environment/tooling.
14. Report exactly which gates passed and which remain.

The central engineering criterion is simple:

> **`mingo-computing` must work because the code discovers where it is installed, not because every old literal was mechanically replaced with a new literal.**

---

## 26. Implementation audit discoveries

The fresh implementation audit found several relocation dependencies that need to
be included in the pre-rename verification, in addition to the files listed in
the original inventory:

- `UNPACKER_ZERO_STAGE_FILES/software/conf/initConf.m` builds the Octave
  software root from `$HOME/DATAFLOW_v3`; resolve it from `mfilename('fullpath')`.
- Stage-0 and Stage-1 shell wrappers contain inline Python blocks that separately
  construct `Path.home() / "DATAFLOW_v3"`. They need the wrapper's dynamically
  resolved `DATAFLOW_ROOT` passed through the environment.
- `OPERATIONS/.../CLEANERS/top_large_dirs.sh` searches for a sibling directory
  literally named `DATAFLOW_v3`. It should locate the checkout from its script
  path while leaving the historical `SAFE_DATAFLOW_v3` backup root intact.
- The terminal-password ignore entry was an absolute path under the old checkout;
  make the ignore rule repository-relative.
- `CONFIG/add_to_tmux.info` is a compatibility entry point and should invoke the
  setup-managed `~/.local/bin/start-dataflow-tmux` launcher rather than a script
  under the old checkout path.
- Some Python tools were finding the checkout only because the ignored runtime
  station-data directory existed. Root discovery should use checked-in markers
  such as `CONFIG/config_paths.yaml` instead.

These are implementation notes, not exceptions to the relocated-checkout gate.

## 27. Production move findings (2026-10-21)

The pre-move machine audit found one active helper outside the repository-managed
`.bashrc` block: a `watch_file_flow` function with an absolute
`/home/mingo/DATAFLOW_v3` path. It was changed to use the dynamically populated
`DATAFLOW_ROOT`. The setup-managed source block itself was regenerated from the
new checkout as expected.

The crontab outside its managed block contains old-checkout paths only in
commented examples. These were retained as inactive user history; setup rendered
the managed schedule for `/home/mingo/mingo-computing`, and the unmanaged crontab
content was verified byte-for-byte unchanged.

During the move window, an already-running bot process recreated a small
`OPERATIONS/OPERATIONS_RUNTIME/CRON_LOGS/bot_analysis.log` under the old path.
That log was preserved under the new checkout's runtime log directory, the bot
was relaunched against the rebuilt virtual environment, and the residual old
directory was removed. The stale monitor panes were stopped and the managed
`pipeline_monitor` session was recreated from the new checkout.

After the move, setup applied successfully and reported 148 passes, 3 warnings,
and no failures. The warnings concern the missing QA summary, unavailable
`mingo04` SSH, and an undocumented second backup-PC endpoint. GitHub Actions
passed for both simulation checks and documentation deployment at the renamed
repository. The first HTTP probe exposed that the docs source had no root
`index.md`, so MkDocs deployed only a 404 page. A minimal home page linking to
the existing operations guides and notebook archive was added. The subsequent
documentation deployment passed and
`https://csoneira.github.io/mingo-computing/` returned HTTP 200 with the
`MINGO Computing Documentation` title.
