# To-do — INFERENCE_DICTIONARY_VALIDATION

## Pending tasks

### Step 1.1 — Data collection
- [ ] **Expand task collection beyond task 1.** Currently collecting only
      task 1 metadata (`task_1_metadata_specific.csv`).  The simulation runs
      through tasks 1–5 and each task has its own metadata CSV.  The config
      already has `task_ids` but only `[1]` is set.  To collect all tasks,
      set `"task_ids": [1, 2, 3, 4, 5]` and ensure the column prefix
      handling adapts to each task's trigger type (raw_tt, clean_tt, etc.).

### Step 2.1 — Estimation
- [ ] **Explore alternative distance metrics.**  The `l2_zscore` metric
      works well, but `chi2` and `poisson` may perform better for specific
      parameter ranges.  Run a comparison and document the results.
- [ ] **Explore alternative feature column subsets.**  Currently using all
      `raw_tt_*_rate_hz` columns.  Investigate whether a reduced set
      (e.g. only 4-fold and 3-fold topologies) improves robustness.

### Step 3.1 — Uncertainty
- [ ] **Validate LUT monotonicity.**  The uncertainty should decrease with
      increasing event count.  Add an automated check and warning.
- [ ] **Cross-validation.**  Implement a K-fold or leave-one-out strategy
      to assess the stability of the LUT.

### General
- [ ] **Integration with real data.**  Once the pipeline is validated on
      simulations, adapt it for use with real detector data by importing the
      `estimate_parameters` module with a production dictionary.
- [ ] **Documentation.**  Write a user guide for running the pipeline end
      to end, including config options.
