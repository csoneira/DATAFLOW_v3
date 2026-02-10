# To do:

- [ ] Relax the filters so the filtered events per simulated data is the 100% of the incoming, that is: all data (or almost) is valid. Else it means that the filtering is not done right.
- [X] Add a new metadata csv to the execution pipeline called status for the STEP_1 scripts that will do the following: initialize a row with the basename of the file and the execution date, then set a number which will indicate the % of completion of the execution. In the VERY beginning it will be set to 0, in the end, before exiting and as a very last step, it will be set to 1, and some calls can be introduced inside of the code to update that number with a value between 0 and 1 to indicate a filling percentage. Let's start with 0, 0.25, 0.5, 0.75, 1.
- [X] Now create a GUI tool that displays the current status for the last files executed in a timeline. I would like to see in real time the latest modifications to see if everything is running. Note that i am running from a remote computer, so it should work with the -X11 forwarding.
- [ ] change the name /home/mingo/DATAFLOW_v3/INFERENCE_DICTIONARY_VALIDATION by a more proper name to reflect what we are doing there


Note that:

- Simulated data is useful to *validate* the software and the reconstruction and so on. And even to define uncertainties and systematics.
- The simulation needs validation.
- The inference method needs validation.
- The analysis software needs validation.




- [X] Please ensure that the task scripts have a well-organized and consistent header that can be shared across all five scripts to avoid redundancy. Focus on removing repeated code, but do not add any additional functionality or attempt to modularize the introduction. I only want you to eliminate the redundant code.

- [ ] Can you do a viability study to put the scripts inside of /home/mingo/DATAFLOW_v3/MASTER/STAGE_1/EVENT_DATA/STEP_1 in GPU? Using CuPy instead of NumPy. Do not change anything, I only want you to estimate your action in the scripts, if it will be possible, if you see bottlenecks, or some issues that should be solved before doing the transition to GPU.

- [X] /home/mingo/DATAFLOW_v3/MINGO_DIGITAL_TWIN/PLOTTERS/plot_param_mesh.py should use the info in /home/mingo/DATAFLOW_v3/MINGO_DIGITAL_TWIN/SIMULATED_DATA/step_final_simulation_params.csv which will be marked as "completed", like in green or something, and the info in /home/mingo/DATAFLOW_v3/MINGO_DIGITAL_TWIN/INTERSTEPS/STEP_0_TO_1/param_mesh.csv which, if not in the step_final_simulation_params.csv, will be marked as "in process".

- [X] The thing is that this is a MAJOR CONCERN i have respect to my pipeline: the purity should be close to 100% for the mi00, because it's noiseless data, but still i see that the purity is not so high. The key is not so much the fact that the task 4 removes rows, but the fact that during the tasks 1 to 4 some values are set to 0, and in the end those 0-settings are giving the issue with the row removal. So the key is, in the end, the filters that are colelcted in /home/mingo/DATAFLOW_v3/STATIONS/MINGO00/STAGE_1/EVENT_DATA/STEP_1/TASK_*/METADATA/task_*_metadata_filter.csv. The thing is, then: some of my filters are too strict. Let's start and focus on task 4 by now, then let's move for the others. I need to have super clear which are the filters that are being applied to the task 4. They are mostly defined in /home/mingo/DATAFLOW_v3/MASTER/CONFIG_FILES/config_parameters.csv, but i think that i prefer to have 5 config_parameters, one per each task, so i have very clear where the filters apply.

- [~] Now i have a very important task for you. I am studying the filters. So, what i want is that you add some plots which will be called "debug plots" which are going to show, just before the filterings in any part of the task scripts, the histograms of the columns that will be filtered, just before filtering. I want to see visually which is the shape of the data before applying the filters. And i would like a plot per each one of the columns and per each one of the parameters. Actually you could join into one plot all the histograms that suffer a certain filter in a certain part of the code. The values of the filters should be displayed in the x axis of the histograms. Make sure to treat those plots as "debug plots", so they dont require the "create_plots", but only a "create_debug_plots" variable.