# Troubleshooting

*Last updated: March 2026*

Below are solutions to common problems encountered during station
operation.  If your issue is not listed here, consult the relevant
subsection of the [Operation](../operation/index.md) or file an issue in the
project repository.

## Strips appear wider than expected
When viewing the XY diagram some strips may look wider. This is usually due to incorrect TDC calibration values. Recalibrating the TDC resolves the issue.

## Negative values in correlation plots
A cloud of negative values indicates the TDC is swapping the leading and trailing edges of the signal. Replacing or reconfiguring the TRB/TDC module (as done on 13 July 2023) fixes the problem.

## Values concentrated on the axes
If the correlation plot contains many points on the axes:

- Only one side of a strip might be reading out. Verify the electronics with a multimeter.
- Front and back channels may be swapped. Reconnect the coaxial cables from the strip to the daughterboard in the correct order.

## `startRun.sh` continues running
Terminate remaining `dabc_exe` processes:

```bash
ps -ef | grep dabc
kill $(ps -ef | grep '[d]abc' | awk '{print $2}')
```

## System is not measuring
- **Error on DAQ:** rerun `./startDAQ`.
- **DAQ on but no events:** ensure the PC firewall is disabled.

## TRB not detected
If the TRB was replaced, assign it an IP address from the miniTRASGO PC. This is rarely required and should only be done when necessary.

