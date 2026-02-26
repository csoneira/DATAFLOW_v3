# Preparing the minGO

The `crontab` is configured to automatically execute the needed software when rebooting the system. It is possible, though, that some execution needs to be stopped and then relaunched, so it is interesting to know what are the especific required programs to work with minGO.

Currently the miniTRASGO is operating in a room with controlled 21ºC.

## The DCS (Data Control System)
The connection through the I2C protocol from the mingo PC to the hub containing the environment, HV and gas flow sensors is established thanks to the script with path: 

    media/externalDisk/gate/bin/dcs_om.sh

## Starting the DAQ (Data Acquisition)
In the directory `home/rpcuser/userscripts/trb399sc/` there are some essential scripts for the mingo operation. The first step is to start the data acquisition. The program to get the data from the TRB is found in `./startup_TRB399.sh`, which has a shorcut, `./startDAQ`.

This starts the data adquisition system, but does not save any time and width information of the event into files but the rate, which does not need the acquisition of the whole event, only the register of the incoming rates to the TRB (this creates the only `.log` that does not come from the I2C hub). Once this is started we can set the triggers we want in the Central Trigger System (CTS), the subpage of the web-based DAQ control found in

    minitrasgo.fis.ucm.es:1234/cts/cts.htm
There we can select the standard trigger we want to use, that will be saved as Trigger Type (TT) 1. The other triggers, that can be automatically selected from a script, are in the directory `home/rpcuser/trbsoft/userscripts/trb399sc/trigger`: those are called when automatically performing through the `crontab` the self trigger (which is considering a hit in any layer a trigger and it is stored with the category TT 2).

## High Voltage control
To calculate the necessary HV one needs to know the temperature of the room. From it, the density of the gas can be calculated and these RPC detectors must work in a certain regime of the Townsend ratio: E/rho (electric field applied/gas density), in particular around 240 Towsends for miniTRASGO. This means that the HV is chosen accordingly with the density of the gas, and hence with the temperature. We have several options, anyway, we could also (and it is the option we are choosing) set the voltage constant around the plateau and later correcting the rates by the efficiency, which is a better option than changing everytime the potential: an error could be devastating for the detector and also we are changing constantly the regime of the electric field, which needs to stabilize inside the RPC.

At any window, execute `home/rpcuser/bin/HV/hv` with the following arguments:

- `-b <bus>` : Bus number
- `-I <Ilim>` : Current limiter (μA)
- `-V <Vset>`: High voltage value (kV)
- `-on` : Turn HV ON
- `-off` : Turn HV OFF

Examples:

    ./hv -b 0 -I 1 -V 5.5 -on 
<!-- tsk -->
    ./hv -b 0 -off

To see the information on HV and intensity in real time just type:

    watch -n 1 ./hv -b 0

## TRB
To check the window in time of the trigger we can go to the TDC, write c001 in the search and then, once the page is loaded, see the c801 row. It says how wide the window is before and after the trigger: we could even shorten the window before, since we know that usually all the events are in -150 ns (we can see that in the Q1_F, etc files).
