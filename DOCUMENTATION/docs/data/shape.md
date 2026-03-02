# Data shape and derived variables

*Last updated: March 2026*

This page describes the internal structure of the measurement data produced by
the detector and the common quantities derived from it.  Understanding the
"shape" of the data is useful when writing analysis or monitoring scripts.

## Raw data format

Each event recorded by the DAQ contains the following primitives for each strip
(side front/back):

- **Time** (`T<n>_F`, `T<n>_B`): timestamp of leading edge in nanoseconds.
- **Charge** (`Q<n>_F`, `Q<n>_B`): LVDS pulse width (TOT) in arbitrary units (AU).
- **Event metadata**: run number, trigger type, plane index, etc.

The raw data files (after unpacking) use variable names of the form `T1_F` for
the front channel of strip 1; similar patterns apply for all 32 strips and four
planes.  Times are stored relative to the event builder clock; absolute timing
is recovered via synchronization with the RPC computer clock.

## Derived coordinates

- **X position** – computed from the time difference:

  \[ X = (T_F - T_B) \cdot v_p + X_{offset} \]

  where \(v_p\) is the propagation velocity (calibrated) and the offset comes
  from cable length differences.  This quantity is calibrated with the
time‑to‑position procedure described in [Calibration](../operation/calibration.md).

- **Y position** – discrete strip index.  The wide strip (usually strip 8) has
  lower spatial resolution.

- **Charge** – average of front and back measurements for the strip; the
  algorithm selects the maximum among coincident hits when multiple strips are
  fired simultaneously.

- **Efficiency** – computed by selecting events where three of the four RPC
  planes fire and evaluating the response of the fourth.  Efficiency maps are
  produced by binning the (X,Y) coordinates of these test events.

- **Rate** – total count of accepted triggers per unit time; can be further
  subdivided into coincidence/self‑trigger modes.

## Advanced quantities

- **Track angles** – by chaining hits across the four planes, incident
  angles \(\theta,\phi\) can be reconstructed.
- **Multiplicity** – number of strips triggered within a coincidence window
  (useful for bundle/air‑shower studies).
- **Streamer flag** – events with charge above a configurable threshold
  (typically set to separate the high‑charge tail of the spectrum).

## Data files produced by the analysis scripts

The MATLAB `ana.m` routine (see [Dataflow](dataflow.md)) produces summary
variables which are stored in `Vars/TT1` and `Vars/TT2` directories.  The
most commonly used arrays are documented in the Dataflow page.

Use `python` with `pandas` or `h5py` to inspect these files in the offline
analysis environment; their column names mirror the variables described
above.

## Example inspection

```python
import pandas as pd

# load a small sample of the dictionary CSV to inspect field names
df = pd.read_csv('MINGO_DICTIONARY_CREATION_AND_TEST/STEP_1_2_BUILD_DICTIONARY/OUTPUTS/FILES/dictionary.csv', nrows=1)
print(df.columns.tolist())
```

This example demonstrates the correspondence between simulation variables and
real data field names.

---

_Local preview: run `mkdocs serve` in the documentation root._

