# Operating miniTRASGO
This section describes how to configure, calibrate, measure and monitor the detector.  
Before those steps, establish a connection to the miniTRASGO PC.

## Connecting to miniTRASGO
You can connect via SSH for command-line access or through a remote desktop session.

### SSH
Use SSH to start the high-voltage supply, the DAQ and other command-line tasks. Add the following entry to your `~/.ssh/config` (or supply the same parameters on the command line):

```
Host mingo01
    HostName minitrasgo.fis.ucm.es
    User rpcuser
```

Enter the password when prompted. Exit the session with `exit`.

### Remote desktop
To access the graphical interface, connect with VNC:

```
vncviewer minitrasgo.fis.ucm.es:0
```

The VNC password matches the SSH password.

The subsequent pages cover:

1. Configuration – powering and setting up the hardware.
2. Calibration – tuning timing and charge parameters.
3. Measuring – running data acquisition and retrieving data.
4. Monitoring – verifying proper operation.

