# Postmortem & Runbook — missing STEP_3 INTERSTEPS (example: `SIM_RUN_087_006`)

## Executive summary ✅
- Symptom: STEP_3 produced no new `INTERSTEPS/STEP_3_TO_4` outputs for param-mesh rows that were *pending* in `param_mesh.csv` (example: `step_1_id=087, step_2_id=006`).
- Root cause: param_mesh had pending rows that lacked the required upstream STEP_2 `SIM_RUN` directories; the scheduler respected those dependencies and (with `RUN_STEP_STRICT_LINE_CLOSURE=1`) did **not** open new STEP_1 lines to generate upstream inputs.
- Impact: STEP_3 appeared idle for those param rows while the pipeline continued processing other, already-satisfied prefixes.

---

## Key evidence & exact reproduction (what I inspected)
- Param row(s) pending (example excerpt from `INTERSTEPS/STEP_0_TO_1/param_mesh.csv`):

```csv
# pending rows (done==0) for prefix 087
0,087,006,280,001,001,001,001,001,001,001,...
0,087,006,276,001,001,001,001,001,001,001,...
0,087,006,277,001,001,001,001,001,001,001,...
```

- Upstream directory missing (before remediation):
  - `ls INTERSTEPS/STEP_2_TO_3/` → contained `SIM_RUN_087_004/` but **no** `SIM_RUN_087_006/`.
- Scheduler state that prevented automatic upstream creation:
  - `/tmp/mingo_digital_twin_run_step_work_cache.csv` — step 3 row showed `has_work=0 produced_dirs=0 expected_dirs=3`.
  - `/tmp/mingo_digital_twin_run_step_state.csv` — `active_open_step1_lines=1` and `step1_new_lines_allowed=0` (strict closure).
- run_step logs (examples):
  - `step=2 status=progress dirs=0->1 elapsed_s=141.718` and `all steps completed in 143s` — runner loop was active but not creating the missing upstream inputs.

---

## Root-cause breakdown (concise)
1. Pipeline dependency: STEP_3 requires a STEP_2 `SIM_RUN_<step1>_<step2>` directory for each param_mesh row. If that directory is missing, STEP_3 has nothing to process for that param row.
2. The `param_mesh.csv` row was correctly marked `done==0` (pending) but the matching STEP_2 SIM_RUN was not present → manual or upstream generation needed.
3. Scheduler gating: `RUN_STEP_STRICT_LINE_CLOSURE=1` prevented automatically opening new STEP_1 lines to produce the upstream input; therefore **no automatic remediation** happened.

> Net effect: pending param_mesh row + missing upstream SIM_RUN + strict scheduler policy = no STEP_3 output for that row.

---

## How I validated (commands used)
- Inspect param mesh:
  - `grep -n "^0,087,006" INTERSTEPS/STEP_0_TO_1/param_mesh.csv`
- Confirm missing upstream:
  - `ls -1 INTERSTEPS/STEP_2_TO_3/`  # verify absence of SIM_RUN_087_006
- Check scheduler state:
  - `cat /tmp/mingo_digital_twin_run_step_work_cache.csv`
  - `cat /tmp/mingo_digital_twin_run_step_state.csv`
- Manual remediation used to validate behaviour (example commands I ran to reproduce and confirm outputs):
  - STEP_2 (create upstream SIM_RUN):
    ```bash
    python3 MASTER_STEPS/STEP_2/step_2_generated_to_crossing.py \
      --config MASTER_STEPS/STEP_2/config_step_2_physics_087_006.yaml \
      --runtime-config MASTER_STEPS/STEP_2/config_step_2_runtime_087_006.yaml --no-plots
    ```
  - STEP_3 (consume new input):
    ```bash
    python3 MASTER_STEPS/STEP_3/step_3_crossing_to_hit.py \
      --runtime-config MASTER_STEPS/STEP_3/config_step_3_runtime_087_006.yaml --no-plots
    ```
  - Expected result: `INTERSTEPS/STEP_2_TO_3/SIM_RUN_087_006/` and `INTERSTEPS/STEP_3_TO_4/SIM_RUN_087_006_<id>/` appear.

---

## Immediate (manual) remediation checklist — do this when you see the symptom
1. Confirm the param_mesh rows are `done==0` and list their `(step_1_id, step_2_id)` pairs.
2. If the upstream `SIM_RUN` is missing, either:
   - Manually run the appropriate `STEP_2` (example command above) for that `(step_1, step_2)`, then run `STEP_3`; OR
   - Set `RUN_STEP_STRICT_LINE_CLOSURE=0` temporarily and let the scheduler open STEP_1 lines (use with caution).
3. Re-run the scheduler (or the single steps) and verify `INTERSTEPS/STEP_3_TO_4` contains the expected `SIM_RUN_*`.
4. If param_mesh `done` flags are inconsistent (rows incorrectly marked `done==1`), do not change automatically — inspect and correct param_mesh with careful source control.

---

## Permanent fixes (prioritized, recommended) — implement these to *prevent recurrence*
1. Param-mesh ↔ INTERSTEPS consistency check (high priority)
   - Add a preflight check in `run_step.sh` (or `STEP_SHARED/sim_utils.py`) that scans `param_mesh.csv` for `done==0` rows and verifies the required upstream `SIM_RUN_<s1>_<s2>` exists in the corresponding `INTERSTEPS` directory.
   - If missing, emit a clear `WARN/CRITICAL` log and set a `work_cache` flag so the scheduler knows upstream work is required.
   - File candidates to edit: `MINGO_DIGITAL_TWIN/run_step.sh` + `MASTER_STEPS/STEP_SHARED/sim_utils.py`.

2. Add an automated *consistency monitor* (medium priority)
   - Small script `tools/check_param_mesh_consistency.py` run by cron/CI that reports mismatches and writes to `CRON_LOGS/` and optionally posts an alert.

3. Expose safe auto-bootstrap policy (optional / opt‑in)
   - Add configuration `RUN_STEP_AUTO_BOOTSTRAP_UPSTREAM=0|1`. If enabled, `run_step` will open STEP_1 lines to produce missing upstream `SIM_RUN`s when safe.

4. Improve `STEP_2` input-selection logic (medium)
   - Make `input_sim_run=random` prefer param_mesh rows with `done==0` and/or provide an explicit CLI flag `--param-row-id` for deterministic runs.

5. CI + unit tests (high priority)
   - Add tests asserting: for any `param_mesh` row with `done==0`, either the matching upstream `SIM_RUN` exists or `run_step` will enqueue upstream work (test the preflight checker).
   - Add an integration test that simulates `step1_new_lines_allowed=0` and verifies the consistency checker raises a visible alert.

6. Runbook & operator docs (immediate)
   - Add an entry under `MINGO_DIGITAL_TWIN/DOCS/` (this file) and link it from the on-call runbook.

---

## Example detection script (one-liner)
Use this to find pending param rows that lack upstream STEP_2 sim runs:

```bash
awk -F, '$1==0 {printf "%03d %03d\n", $2, $3}' INTERSTEPS/STEP_0_TO_1/param_mesh.csv \
  | while read s1 s2; do
    sim_dir="INTERSTEPS/STEP_2_TO_3/SIM_RUN_${s1}_${s2}"
    [[ -d "$sim_dir" ]] || echo "MISSING_UPSTREAM: $s1,$s2"
  done
```

---

## Suggested code patch (pseudo / exact-where-to-add)
- Add function `check_param_mesh_upstream()` to `MASTER_STEPS/STEP_SHARED/sim_utils.py` that returns missing pairs.
- Call it from `MINGO_DIGITAL_TWIN/run_step.sh` after `refresh_step_work_cache()` and before gating `step1_new_lines_allowed()` checks; if missing upstreams exist, add a `work_cache` row or log.

Suggested unit test: `tests/test_param_mesh_consistency.py` — assert missing pairs are detected and that `run_step` logs the mismatch.

---

## Validation checklist (after fixes)
- [ ] Unit tests added and passing.
- [ ] `tools/check_param_mesh_consistency.py` returns zero mismatches on a clean pipeline.
- [ ] On a simulated broken state (create a `param_mesh` row and remove upstream `SIM_RUN`), the new preflight raises a WARN/CRITICAL and the CI test fails.
- [ ] Run `run_step -c` for one loop and verify `work_cache` flags upstream work or auto-bootstraps when configured.

---

## Suggested Codex prompt (ready to paste)
> "Add a param_mesh ↔ INTERSTEPS consistency checker: implement `check_param_mesh_upstream()` in `MASTER_STEPS/STEP_SHARED/sim_utils.py`, call it from `MINGO_DIGITAL_TWIN/run_step.sh` as a preflight so `run_step` logs missing upstream SIM_RUNs for `param_mesh` rows with `done==0`. Add a unit test `tests/test_param_mesh_consistency.py` and a small `tools/check_param_mesh_consistency.py` script. Make the change opt‑out via `RUN_STEP_AUTO_BOOTSTRAP_UPSTREAM` and include documentation in `MINGO_DIGITAL_TWIN/DOCS/`." 

---

## Appendix — relevant paths
- Pipeline entrypoint / scheduler: `MINGO_DIGITAL_TWIN/run_step.sh`
- Param mesh: `MINGO_DIGITAL_TWIN/INTERSTEPS/STEP_0_TO_1/param_mesh.csv`
- STEP_1 outputs: `MINGO_DIGITAL_TWIN/INTERSTEPS/STEP_1_TO_2/` (e.g., `SIM_RUN_087`)
- STEP_2 inputs/outputs: `MINGO_DIGITAL_TWIN/INTERSTEPS/STEP_2_TO_3/` (missing `SIM_RUN_087_006` before remediation)
- STEP_3 outputs: `MINGO_DIGITAL_TWIN/INTERSTEPS/STEP_3_TO_4/`
- Scheduler temporary state: `/tmp/mingo_digital_twin_run_step_work_cache.csv`, `/tmp/mingo_digital_twin_run_step_state.csv`
- Helpful scripts to run manually: `MASTER_STEPS/STEP_2/step_2_generated_to_crossing.py`, `MASTER_STEPS/STEP_3/step_3_crossing_to_hit.py`

---

If you want, I can (pick one):
- implement the consistency checker + unit test now, or
- add the cron monitor script, or
- create a CI test and PR with the changes above.

Pick which preventive change to implement next and I will proceed (I will not change anything until you confirm).