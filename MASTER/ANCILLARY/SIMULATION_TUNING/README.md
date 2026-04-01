# Simulation Tuning

This directory groups simulation-tuning studies that compare `MINGO00`
(digital-twin outputs) against selected real stations.

These studies are advisory-only. They read the current simulation config,
produce evidence plots plus recommended values or ranges, and do not modify
the digital-twin physics YAML files automatically.

The shared selection lives in:

- `config_simulation_tuning.yaml`

Current studies:

- `TDIF_ACTIVE_LENGTH/`
  Compare calibrated `T_dif` distributions to tune the effective active strip
  length and the reconstructed X / timing width.
- `INDUCTION_SECTION/`
  Compare the same-plane strip multiplicity vector (`1, 2, 3, 4` active strips)
  directly between simulation and real data, using only adjacent topologies,
  plus an empirical-efficiency matched comparison that gives a first gamma
  scale recommendation.
- `CHARGE_SPECTRUM/`
  Compare calibrated charge spectra plane by plane, matching simulated and real
  files by similar empirical plane efficiency, with nearest-neighbour fallback
  when strict overlap is too sparse, and report both charge-scale and
  charge-shape recommendations.

The selection config is shared so the same real-data epoch and station set can
be used consistently across tuning studies.
