# Data: general considerations
In this section we discuss the fundamental parts of the preliminary data treatment until it is ready to extract and analyze.

## Data retrieval

If needed, files from miniTRASGO can be transferred to a local machine using `scp`.

    scp <user>@<remote machine>:<remote file> <local directory>
  
For example:

    scp rpcuser@192.168.3.216:gate/system/devices/RCP01/data/dcData/data/2023-07-13-EffMap.mat ~/

## About the clocks and units of the timeStamps
Take into account that the internal clock of the mingo PC is set in UTC (Coordinated Universal Time).

In MATLAB, the date format, for example 739077.6528125, represents a serial date number. MATLAB uses serial date numbers to represent dates as the number of days since January 0, 0000 (a fictitious date). To convert the serial date number to a readable date, you can use the `datestr` function in MATLAB. We can obtain a maximum precision of miliseconds with there data format, even though the actual value is known up to nanoseconds.

## Detector Control System and the logs
The slow control is controlling the flow-meters, the environment meter (bus 0 for the outside information, laboratory conditions, and bus 1 for that inside the DAQ, inside the box) and the HV through an I2C Hub that is connected to the minGO PC. This Hub allows the communication with the computer, back and forth. Actually the programs that retrieve data from these devices are called from the `crontab` periodically, so its name and path can be seen there. The operation is quite similar to that of the TRB: the devices drop a data file that is read and stored as a `.log` text file in `home/rpcuser/logs`. A `.log` for that day is updated each time it is loaded; when the day passed that file is moved to the `done` directory where all the previous ones are stored.

Every device has its own LookUp Table (LUT).

Information about general variables involving temperature, pressure, humidity... obtained from the I2C hub. There is a `.log`, though, that does not come from the DCS but from the TRB: the Rate, whose information is configurated in a python script `~/gate/python/log_CTSrates_multiProcessing.py` instead of in a LookUp Table in the `~/gate/system/lookUpTables`. It is obtained thanks to `trbnetd`, a daemon.

### `sensors_bus0_YYYY-MM-DD.log`
Environment information as measured by the external (outside the box) climate detector. The row format is as follows:

    <YYYY-MM-DD>T<HH:mm:ss>; nan nan nan nan <T in ºC> <HR in %> <P in mbar>
<!-- tsk -->
    2023-07-21T23:45:03; nan nan nan nan 24.7 54.5 1007.8
### `sensors_bus1_YYYY-MM-DD.log`
Environment information as measured by the internal (inside the box) climate detector. The format is as follows:

    <YYYY-MM-DD>T<HH:mm:ss>; nan nan nan nan <T in ºC> <HR in %> <P in mbar>
<!-- tsk --> 
    2023-07-21T23:45:03; nan nan nan nan 24.7 54.5 1007.8
### `Flow0_YYYY-MM-DD.log`
Flow coming out of the four RPCs. The format is as follows, all the flux is in AU. The zeroes are considered, in the same order as they appear in the `.log`, in 492, 513, 501 and 518.

    <YYYY-MM-DD> <HH:mm:ss> <T1 flux> <T2 flux> <T3 flux> <T4 flux>
<!-- tsk -->
    2023-08-01 13:35:04 805 802 860 667
### `hv0_YYYY-MM-DD.log`
Intensities in micro A, voltages in kV unless other unit is specified.

    <YYYY-MM-DD>T<HH:mm:ss> <MAC addres (6 values)> <IHVp> <IHVn> <VHVn> <VHVp> <VHVs> <Vpwr in V> <Vset> <Ilim> <Vset in DAC> <Ilim in DAC> <Some info about IO (6 values)>
<!-- tsk -->
    2023-08-01T13:30:03 80 1F 12 59 F5 21 0.066 0.070 5.202 5.202 0.000 1.284 5.202 1.006 5.200 1.000 1 1 0 0 1 1 1
### `rates_YYYY-MM-DD.log`
This essentially contains the information from the CTS web based system.

    <YYYY-MM-DD>T<HH:mm:ss> <Trigger asserted> <Trigger rising edges> <Trigger accepted> <Trigger multiplexer (TM) 0> <TM 1> <TM 2> <TM 3> <Coincidence Module (CM) 0> <CM 1> <CM 2> <CM 3>
<!-- tsk -->
    2023-08-01T13:45:51; 9.0 7.9 7.9 56.5 83.9 91.0 97.0 5.0 2.6 2.8 4.7 
## The LookUp Tables
They are stored in the `~/gate/system/lookUpTables`. It is a pending objective to write them in the same format: excel. They give information on how the programs that retrieve information store them, among other tasks.
