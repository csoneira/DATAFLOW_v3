# Detector design

The detector design notes collect reference information needed to reproduce a
miniTRASGO station or audit an existing build. Hardware, firmware, and software
components are documented with the conventions used in assembly drawings, data
acquisition code, and analysis notebooks. Adopting a common language across
sites ensures that geometry descriptions, calibration constants, and coordinate
systems can be shared unambiguously within the CASTRO collaboration.

Key topics covered in this section include:

- **Mechanical layout.** Reference frames, module numbering, and detector
  spacing used in Monte Carlo simulations and reconstruction code.
- **Front-end and trigger electronics.** The interface between RPC pickup strips
  and the TRB3sc digitiser, including biasing, threshold settings, and watchdog
  logic.
- **Software environment.** Supported operating systems, container images, and
  dependencies required to run the monitoring and reconstruction pipelines.

For detailed schematics and configuration examples, refer to the dedicated
subpages linked in the navigation sidebar.
