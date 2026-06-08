# To do

- [ ] Clean the repository. This means:
  - [X] Remove not useful code and directories.
  - [ ] Clean all the AI rubbish.
  - [ ] Create a good global scheme of the pipeline and automation, including control scripts.

- [ ] Change pipeline dynamics so that the processed listed files are not erased, but kept as a reference and data to actually make a lot of test over. Does not make sense to lose a lot of info accumulating too soon, before we have clear how some stuff should be gated. To do that, we need to track the size and coverage,as well as version of files for the directory. Also we need a good way to store efficiently.

- [ ] Modify /home/mingo/DATAFLOW_v3/MINGO_ANALYSIS/MINGO_ANALYSIS_SCRIPTS/CONFIG_FILES/STAGE_0/REPROCESSING/STEP_2/config_step_2.yaml and /home/mingo/DATAFLOW_v3/MINGO_ANALYSIS/MINGO_ANALYSIS_SCRIPTS/CONFIG_FILES/STAGE_0/REPROCESSING/STEP_1/config_step_1.yaml so they are more logical, because currently that variable is pretty clumsy, and also implement the possibility of defining them in /home/mingo/DATAFLOW_v3/MINGO_ANALYSIS/MINGO_ANALYSIS_SCRIPTS/CONFIG_FILES/STAGE_0/REPROCESSING/config_reprocessing.yaml, having that priority if not empty (similarly to what happens with the STAGE_1 configs)

- [ ] I want the plotting parts of the task scripts externalized, or even as a secondary script, for some cases, not in between the analysis code, or at least not extensively.

- RENAME:
  - /DATAFLOW_v3/MINGO_DIGITAL_TWIN/SIMULATED_DATA --> /DATAFLOW_v3/MINGO_DIGITAL_TWIN/OUTPUT_SIMULATED_DATA


- [ ] Clean the /home/mingo/DATAFLOW_v3/MINGO_DIGITAL_TWIN/ORCHESTRATOR of redudancies, imprecissions and complexities. Maybe not as many scripts are needed.


