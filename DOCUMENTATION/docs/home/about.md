# About the miniTRASGO

## Mission and context

miniTRASGO extends the TRASGO (TRAcK reconStructing bOx) concept into a compact
instrument that can be deployed quickly across a global network of stations. It
supports the CASTRO collaboration in correlating cosmic-ray arrival directions
with solar energetic particle events, atmospheric tides, and geomagnetic
disturbances. The design philosophy draws on the large-area TRAGALDABAS and
TRISTAN detectors while prioritising transportability, simplified services, and
remote operation.

## Resistive Plate Chambers in miniTRASGO

Resistive Plate Chambers (RPCs) provide the primary sensing technology. Each RPC
consists of two 1 mm gas gaps separated by 2 mm glass electrodes with surface
resistivity tuned to suppress discharges while maintaining charge evacuation.
Ionisation avalanches initiated by traversing charged particles are amplified by
high electric fields, yielding signals with gains up to 10⁶. The double-gap
configuration balances efficiency and timing: thin gaps deliver sub-nanosecond
resolution, while multiple stages recover the >90% detection efficiency typical
of thicker chambers without incurring space-charge accumulation.

miniTRASGO operates in avalanche mode with 1,1,1,2-tetrafluoroethane (R134a) as
the working gas. Although classified as a greenhouse gas, R134a is readily
available and compatible with the environmental safety protocols defined in the
miniTRASGO operations manual. Gas handling emphasises controlled flow, pressure
relief during transport, and leak mitigation to preserve mixture purity over
long unattended runs.

## Detector architecture

Four 30 × 30 cm² RPC planes, instrumented with orthogonal pickup strips on both
faces, form a tracking volume capable of reconstructing multi-particle bundles.
HADES-derived front-end electronics condition the analogue signals and feed TRB3
time-to-digital converter boards that deliver time-over-threshold information to
an embedded single-board computer. Supporting subsystems include:

- **High-voltage distribution.** Independent positive and negative lines, driven
  by a software-controlled supply connected over I²C.
- **Slow controls.** Environmental probes (temperature, pressure, humidity) and
  gas flow meters integrated into the supervisory data stream.
- **Data handling.** Local storage on solid-state drives with mirrored transfer
  to the collaborative analysis pipeline maintained in the
  [miniTRASGO-analysis](https://github.com/cayesoneira/miniTRASGO-analysis/tree/main)
  repository.

## Collaboration landscape

The TRASGO family encompasses installations in Spain, Portugal, Poland, Mexico,
and Colombia. CASTRO coordination workshops harmonise calibration strategies and
data formats, ensuring compatibility between miniTRASGO nodes and larger
instruments such as TRAGALDABAS. Active contributors include:

- Warsaw University of Technology (Poland)
- Benemérita Universidad Autónoma de Puebla (Mexico)
- Universidad Pedagógica y Tecnológica de Colombia (Tunja, Colombia)
- Universidad Complutense de Madrid (Spain)
- IFIC – Instituto de Física Corpuscular (Valencia, Spain)
- Universidade de Santiago de Compostela (Spain)
- LIP – Laboratório de Instrumentação e Física Experimental de Partículas (Coimbra, Portugal)

Collectively, these teams advance detector hardware, environmental modelling,
and physics exploitation, positioning miniTRASGO as the lightweight backbone of
the TRASGO network.
