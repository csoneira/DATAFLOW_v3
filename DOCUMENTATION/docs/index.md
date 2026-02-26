# miniTRASGO Documentation

*Last updated: September 2025*

The miniTRASGO (miniature TRASGO) programme delivers a transportable cosmic-ray
tracking station that combines multi-gap Resistive Plate Chambers (RPCs) with a
compact, low-power electronics stack. The telescope has been optimised for
long-term monitoring of the secondary muon flux and its correlation with solar
events and atmospheric dynamics, following the design described in
Soneira-Landín *et al.* (2025a, 2025b) and the broader TRASGO family roadmap
outlined by García-Castro *et al.* (2021). The documentation collected here
supports every stage of the detector lifecycle—from fabrication and assembly to
operation, data quality assurance, and collaborative analysis.

## System overview

- **Detector concept.** Four 30×30 cm² RPC planes arranged in a tracking stack
  provide sub-centimetre spatial resolution and sub-nanosecond timing. The dual
  gas-gap geometry maintains >90% detection efficiency while limiting charge
  build-up, enabling stable operation in a wide range of environments.
- **Electronics.** HADES-derived front-end boards feed TRB3sc digitiser modules
  that stream time-over-threshold observables to the on-board single-board
  computer. The control chain supports remote firmware updates, watchdog
  recovery, and environmental monitoring for temperature, pressure, and gas
  flow.
- **Network deployment.** Stations in Madrid, Warsaw, and Puebla are providing
  continuous coverage, with additional nodes in commissioning for Monterrey and
  other CASTRO collaboration partners. Cross-site analysis leverages common data
  formats and calibration workflows developed within the miniTRASGO analysis
  software suite.

## Project resources

- **Logbook.** Track fabrication, operation, and maintenance tasks in the
  [miniTRASGO logbook](https://docs.google.com/spreadsheets/d/1ato36QkIXCxFkDT_LtAaLjPP7pvLcor-xZAP4fy00l0/edit?gid=816573551#gid=816573551).
- **Notebook.** Consult the detailed calculation record in the
  [miniTRASGO notebook](https://docs.google.com/document/d/e/2PACX-1vQD_Zr4IW3ZZl0sX4jf7jcD_9tdXbD-kIDpydO0k9HWhewk7yDDWJEF4PC4ARUu1nk-mdEexsyPXXY-/pub).
- **Documentation repository.** This site is built from the
  [miniTRASGO documentation repository](https://github.com/cayesoneira/miniTRASGO).
- **Software toolkit.** Detector control, calibration, and analysis tools live in
  the [miniTRASGO-analysis repository](https://github.com/cayesoneira/miniTRASGO-analysis/tree/main).

## Accessing a detector

### Secure shell (SSH)

Use the dedicated RPC user to reach the control computer:

```bash
ssh rpcuser@minitrasgo.fis.ucm.es
```

### Web-based DAQ control

When connected to the local laboratory network, open the data acquisition web
panel in your browser:

```
http://minitrasgo.fis.ucm.es:1234
```

Additional networking, tunnelling, and monitoring workflows are documented in
the [Operation](operation/index.md) section of this site.

## References

- C. Soneira-Landín *et al.*, "miniTRASGO: A compact RPC tracker for cosmic ray
  studies," *Nucl. Instrum. Methods Phys. Res. A* (2025).
- C. Soneira-Landín *et al.*, "miniTRASGO: Design and initial results of a
  compact Resistive Plate Chamber telescope for worldwide cosmic ray
  monitoring," *Adv. Space Res.* (2025).
- D. García-Castro *et al.*, "The TRASGO Project. Present status and results,"
  *Phys. Atom. Nucl.* 84, 1070–1079 (2021).

