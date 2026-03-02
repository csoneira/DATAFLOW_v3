# Detector Stations

## Detector concept

miniTRASGO stations are compact RPC-based cosmic-ray telescopes built around a four-plane tracking stack.

### Core elements

- Four parallel RPC planes (about 30 x 30 cm active area)
- Two gas gaps per RPC (R134a operation in avalanche mode)
- Front/back strip readout per channel for timing-difference positioning
- Dedicated high-voltage, slow-control, and DAQ subsystems

## Coordinate and geometry conventions

Project conventions are shared across detector operation, analysis, and simulation:

- `X`, `Y`: detector-plane axes
- `Z`: stack axis (used consistently in software contracts)
- Plane indexing: 1 to 4
- Strip indexing: 1 to 4 per modeled plane segment in digital-twin contracts

These conventions are critical for cross-validating real and simulated outputs.

## Deployment context

Stations are deployed across collaborating institutions to support multi-site monitoring and comparative studies across geomagnetic/environmental conditions.

See [Collaborators](../collaborators/index.md) for the institutional map.

## Reference material

- Legacy hardware narrative: [Detector Hardware Notes](../design/hardware.md)
- Methods and conventions in simulation: <https://github.com/csoneira/DATAFLOW_v3/blob/main/MINGO_DIGITAL_TWIN/DOCS/METHODS_AND_DATA_MODEL.md>

