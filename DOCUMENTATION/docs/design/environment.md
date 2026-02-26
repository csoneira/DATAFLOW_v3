## Terminal sessions

The TRASGO team decided to build a `tmux` multiplexer to develop a series of operations in a multiple, permanently open shell. Everything could be done through the main terminal session, but one at a time, and our computer has to be connected. This way we separate visually diffetent executions, as well as we can execute in the background the measuring operations. Any activity on the multiplexer will be shown in real time to every user connected to a `tmux` session. We can access it through the terminal inside the miniTRASGO computer. First we see the `tmux` sessions avaliable:

    tmux ls
It will give as output:
- `0: 7 windows (created ...) (attached)`: utilized for measurement operations.
- `1: 13 windows (created ...) (attached)`: utilized for data treatment.
- `webserver: 1 windows (created ...)`: gives info on the trigger activity; not interactive.

So the line to enter any session is:

    tmux attach -t <session name: 0, 1 or webserver>
  
To leave the `tmux` session just press `CTRL + B, D`.

> To erase a `tmux` window just type `exit`, to create one just press `CTRL+B, C`. You can rename them with `tmux rename-window <new name>`.

## Folder structure

- `/home/rpcuser/bin`: includes software to set on, off and restart (named *powercycle*) the FEE and TRB and also the software to communicate with the HV and the gas flow meters.
- `/home/rpcuser/logs`: all the data stored relative to environment and rates (a special log). It stores that of the day, then gets it into the `done/` directory.
- `/media/externalDisk/gate`: all the tools and data but the logs.
    - `bin`: scripts to copy files and do some stuff (it is used by other bigger routines).
    - `python`: the special `.log`, since it defines a LUT in python that creates a log file communicating with the TRB (the rest of the logs come from the I2C hub).
    - `software`: it has all the relevant information on the first processing of data (all the calculations from Alberto Blanco's Matlab/Octave scripts). It also has scripts to run alarms, initialize the DAQ, the DCS (Data Control System), etc.
    - `system`: it has all the lookUp tables in `lookUpTables` (but the rate one), and also the most important directory: `devices`. We will review its content in the dataflow section.


## Some relevant tools

### crontab

`crontab -e` opens a crontab window where you can schedule certain operations on the linux terminal. To translate to the date and hour format used by crontab just enter [Contrab guru](https://crontab.guru/). Some processes can also be set to execute in the moment of reboot.

### Look Up Tables (LUTs)
Stored in `/media/externalDisk/gate/system/lookUpTables`. Every device involved in the system has a LUT designated in order to explain how to read the files and also some alarm configured. For example the LookUpTableTRB.m has an alarm system which is the following:

    {<0 or 1 if it is on/off>, <minimum value to triger the alarm>, <how many times has to occur to trigger the alarm>}
    {<0 or 1 if it is on/off>, <maximum value to triger the alarm>, <how many times has to occur to trigger the alarm>}
It is a pending objective to write them all in excel format.

### Data Acquisition Backbone Core (DABC)
The DABC is a software framework designed for distributed data acquisition. It serves as a backbone for managing and processing data from various experiments. One of its key features is its plug-in mechanisms, allowing it to be easily extended to support different data formats and experiments. In the context of test set-ups using the trb3 frontend readout, specific plug-ins have been developed to receive and merge HADES trbnet data packets through UDP connections. This enables efficient handling and combination of data from different sources within the HADES experiment. It is adapted for its use in miniTRASGO.
