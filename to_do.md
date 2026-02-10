# To do:

- [ ] Relax the filters so the filtered events per simulated data is the 100% of the incoming, that is: all data (or almost) is valid. Else it means that the filtering is not done right.
- [X] Add a new metadata csv to the execution pipeline called status for the STEP_1 scripts that will do the following: initialize a row with the basename of the file and the execution date, then set a number which will indicate the % of completion of the execution. In the VERY beginning it will be set to 0, in the end, before exiting and as a very last step, it will be set to 1, and some calls can be introduced inside of the code to update that number with a value between 0 and 1 to indicate a filling percentage. Let's start with 0, 0.25, 0.5, 0.75, 1.
- [X] Now create a GUI tool that displays the current status for the last files executed in a timeline. I would like to see in real time the latest modifications to see if everything is running. Note that i am running from a remote computer, so it should work with the -X11 forwarding.
- [ ] change the name /home/mingo/DATAFLOW_v3/MASTER_VS_SIMULATION by a more proper name to reflect what we are doing there
- 