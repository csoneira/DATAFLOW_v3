# To do

- [ ] Clean the repository. This means:
  - [X] Remove not useful code and directories.
  - [ ] Clean all the AI rubbish from the documentation.
  - [ ] Create a good global scheme of the pipeline and automation, including control scripts.

- [ ] Change pipeline dynamics so that the processed listed files are not erased, but kept as a reference and data to actually make a lot of test over. Does not make sense to lose a lot of info accumulating too soon, before we have clear how some stuff should be gated. To do that, we need to track the size and coverage,as well as version of files for the directory. Also we need a good way to store efficiently.

- [~] Modify /home/mingo/DATAFLOW_v3/MINGO_ANALYSIS/MINGO_ANALYSIS_SCRIPTS/CONFIG_FILES/STAGE_0/REPROCESSING/STEP_2/config_step_2.yaml and /home/mingo/DATAFLOW_v3/MINGO_ANALYSIS/MINGO_ANALYSIS_SCRIPTS/CONFIG_FILES/STAGE_0/REPROCESSING/STEP_1/config_step_1.yaml so they are more logical, because currently that variable is pretty clumsy, and also implement the possibility of defining them in /home/mingo/DATAFLOW_v3/MINGO_ANALYSIS/MINGO_ANALYSIS_SCRIPTS/CONFIG_FILES/STAGE_0/REPROCESSING/config_reprocessing.yaml, having that priority if not empty (similarly to what happens with the STAGE_1 configs). So for example instead of being true or false per each station, it could have a list of stations to process and another list of stations to reprocess, even if completed, similarly to the stations variable in /home/mingo/DATAFLOW_v3/MINGO_ANALYSIS/MINGO_ANALYSIS_SCRIPTS/CONFIG_FILES/config_selection.yaml


- [ ] I want the plotting parts of the task scripts externalized, or even as a secondary script, for some cases, not in between the analysis code, or at least not extensively.




- [X] Clean the /home/mingo/DATAFLOW_v3/MINGO_DIGITAL_TWIN/ORCHESTRATOR of redudancies, imprecissions and complexities. Maybe not as many scripts are needed. I think now its clearly suboptimal.

- [X] Check that all the scripts that are executed from crontab produce outputs into the /home/mingo/DATAFLOW_v3/OPERATIONS/OPERATIONS_RUNTIME/CRON_LOGS, so that only seeing that i can see if everything is ok or something gives errors. Try, especially in the task scripts, that there are no silent errors, but always gives the reason to stop. And at the same time, make sure that no file is created there if its not being filled: i dont want empty files.







---



# To do list


- put the efficiencies in the csvs as 4 columns, and not a string of four values, just as it is done with the z positions.


## Urgent
- [ ] Put the ideas of the scripts of /home/mingo/DATAFLOW_v3/NOT_ESSENTIAL/tools in the VALIDATION folder.
- [ ] Move the plotting codes inside of the scripts in the /home/mingo/DATAFLOW_v3/MINGO_DIGITAL_TWIN/MASTER_STEPS to PLOTTERS oo have them there displayed right. VALIDATION should be only for validation plots, PLOTTERS is for cool plots that we want to have for the paper and conferences and to explain the method, etc.

## Not so

- [ ] Validate the method using different sample sizes with different statistics. Build the dictionary with a lot of data, then try to estimate the parameters using different sample sizes to see how stable the estimation is.
- [ ] Make possible to generate more events for a certain set of parameters to make richer the statistics.
- [ ] Validation of the simulation.
- [X] Change the name of the json and csv in the SIMULATED_DATA so it does not say STEP_13, which does not exist.
- [ ] Add the energy value to the muons according to the spectrum used in the simulation. And its velocity calculated with it.
- [ ] Add the Moliére Multiple Coulomb scattering.
- [ ] Add the lead above the detector.
- [ ] Validate the avalanche charge with GARFIELD, maybe.
- [ ] Instead of putting the rate in the last steps, put the timestamp already in the first step, depending on the flux, as I had done at some point. Then we could easily relate the final rate with an original rate.
- [ ] Add the efficiency dependence with the angle of incidence. Maybe we could try to implement it from the intrinsic efficiency we have already put, using the oblique trace, and try to get from there a dependence and see if we replicate the experimental data.
- [ ] Add the dark rate of the detector.
- [ ] Add the crosstalk.
- [ ] Add the one side measurements.
- [ ] Put the Townsend coeff from the HV, pressure and Temperature, if possible.