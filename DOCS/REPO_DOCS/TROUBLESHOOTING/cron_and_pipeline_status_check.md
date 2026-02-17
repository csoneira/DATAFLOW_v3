## Cron + pipeline status check (2026-02-12 08:09 UTC)

### Overall status
- `cron.service` is active and running (`since 2026-02-05 11:01:47 WET`).
- Core cron logs are updating this morning (main analysis, simulation, bot, stale-lock solver, watchdog state).
- Telegram bot watchdog is healthy (`Bot already running` every minute up to `2026-02-12T08:09:01+00:00`).
- Stale-lock solver is healthy (`No stale locks detected` up to `2026-02-12 08:05:02 WET`).

### Simulation status (special focus)
- Simulation is currently running continuously:
  - `SIMULATION/RUN/cron_mingo_digital_twin_continuous.log` advanced to `2026-02-12T08:08:24Z`.
  - Live simulation processes are active (`run_step.sh`, `step_final_daq_to_station_dat.py`, `step_2_generated_to_crossing.py`).
- `STEP_0` is running each minute but still in partial-progress mode:
  - Latest: `mesh completion 67.3% (2156/3202 rows done)` at `2026-02-12 08:09:22`.
  - Still reporting `Skip append: mesh not fully done`.
- `STEP_FINAL` is actively running again:
  - Latest writes around `08:09`, including `SIM_RUNs selected: 1`.
  - Frequent expected warnings when requested rows exceed available source rows.

### Important issues found (simulation is active, but NOT error-free)
- `run_step` failure in current morning window:
  - `2026-02-12T08:00:44Z`: `step=2 status=failed rc=1`, `FileNotFoundError` for missing chunk file in `STEP_1_TO_2`.
  - The loop recovered (`all steps completed`, restart at `08:01:52`, then continued).
- `STEP_FINAL` hard failure in current morning window:
  - `2026-02-12 08:03:03`: traceback with `FileNotFoundError: No SIM_RUN_* directories found in .../STEP_10_TO_FINAL`.
  - It recovered by `08:06:03` and resumed producing output.
- Additional runtime pressure signal:
  - `2026-02-12 07:55:22`: `RESOURCE_GATE` skipped one `STEP_0` execution due high swap (`99%`).

### Monitoring notes
- `watchdog_process_counts.log` is updating normally (latest `08:05` report).
- `watchdog_process_counts.cron.log` is currently empty (size `0`, mtime `07:29:33`), unlike yesterday’s error-filled state.

### Conclusion
- The system is running this morning, and simulation is currently continuing.
- It is **still not error-free**: at least one `run_step` failure and one `STEP_FINAL` crash occurred in the latest window, with automatic recovery afterward.
- For your requirement ("continuous and with no errors"), continuity is currently met, but the zero-error condition is not met yet.

### Applied fixes (2026-02-12 08:10 UTC)
- `STEP_FINAL` now handles temporary empty `STEP_10_TO_FINAL` input without crashing:
  - File: `MINGO_DIGITAL_TWIN/MASTER_STEPS/STEP_FINAL/step_final_daq_to_station_dat.py`
  - Behavior change: catches `No SIM_RUN_* directories found` at selection stage, logs warning, exits cleanly.
- `STEP_2` now tolerates transient missing/unreadable chunk files instead of hard-failing:
  - File: `MINGO_DIGITAL_TWIN/MASTER_STEPS/STEP_2/step_2_generated_to_crossing.py`
  - Behavior change: unreadable/empty manifests are skipped; missing chunk files are skipped with warnings.
- Shared chunk reader now skips transiently missing chunk files across steps:
  - File: `MINGO_DIGITAL_TWIN/MASTER_STEPS/STEP_SHARED/sim_utils.py`
  - Behavior change: `iter_input_frames` logs warning and continues when a listed chunk disappears before read.
