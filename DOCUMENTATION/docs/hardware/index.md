# Hardware

This section documents the physical detector system and station infrastructure required to operate DATAFLOW_v3 end-to-end.

## Section contents

- [Detector Stations](detector-stations.md): RPC stack design, coordinate conventions, and station-level deployment context.
- [DAQ and Infrastructure](daq-and-infrastructure.md): electronics chain, software environment, and data acquisition path.
- [Maintenance and Calibration](maintenance.md): routine operations, checks, and calibration procedures.

## Hardware-software coupling

Hardware behavior is tightly coupled to software assumptions in both pipelines:

- Geometry and coordinate conventions feed simulation and reconstruction.
- DAQ data formats feed operational ingestion.
- Sensor and HV behavior influence quality-control and correction logic.

