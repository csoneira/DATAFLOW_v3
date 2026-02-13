# To do list

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