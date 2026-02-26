# Measuring
Here some information about the mearurement procedure.

## Starting the storage of the data of the event
The following script has to be executed to start the data collection: `home/rpcuser/trbsoft/userscripts/trb/startRun.sh`. Now the data is saved into binary `.hld` files, saved at `/media/externalDisk/hlds/`. We will discuss the dataflow in a dedicated section. Press `CTRL + C` to stop the run.

## Some considerations
- Always check the gas flow is on when ramping up the HV.
- Remove from the surrounding every device that could interfere with mingo, for example mobile phones.
- Check the DAQ is on.
- Check the Threshold information: on the `Thresholds` window there is a script called `./setThresholds.sh` which puts the threshold to -40 mV (nominal value); the `./startDAQ` includes this line, but we have to be sure that nothing changes the thresholds automatically, since we are usually executing `./startRun.sh` and not `./startDAQ`, which is started once per system reboot. To check the value of the thresholds, just type: ?????
- To check the number of events of a `.hld` file the line `daq_anal <.hld file>` can be executed; then it can just be stopped. This is mostly useful to know if a file is empty, because in that case the .hlds are ignored.
- Also check that the .hlds have enough events, because in other case some .mats can be loaded, such as Rate.mat, but for example Eff.mat will not.
- When the HV is turned off, the gas flux changes for some minutes: this is due to the fact that the HV bends the RPCs so turning it off will make the structure to recover its original shape.
- The rates can be accepted or edges. Accepted means that the it triggered and the detector had enough time to get the charge signal. Edges means that the trigger was done so close to other signal that the system, even if it can count as a rate since it is detected by the TRB, cannot actually store the information of time and charge. Hence, it will only contribute to the rate, but not to the other analysis.
