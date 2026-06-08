# Gate Tutorial

This file explains how to define gates for the `STEP_2_DETOUR` test framework.

The gate configuration is read from:

- [gates.yaml](/home/mingo/DATAFLOW_v3/MINGO_ANALYSIS/MINGO_ANALYSIS_SCRIPTS/STAGES/STAGE_1/EVENT_DATA/STEP_2_DETOUR/configs/gates.yaml)

The processing command is:

```bash
python3 -m src.main --selection-config configs/selection.yaml --gate-config configs/gates.yaml
```

## What a gate does

A gate is a boolean rule evaluated on every event row in the input dataframe.

If an event satisfies a gate, the corresponding bit is set in the output `gate_mask` column.

Example:

- `bit: 0` means set bit 0
- `bit: 1` means set bit 1
- if an event satisfies both gates, both bits are set in `gate_mask`

So `gate_mask` is a compact `uint64` bitmask, not a text label.

## YAML structure

Each gate must have:

- a unique gate name
- a unique `bit` between `0` and `63`
- an `expression`
- an optional `description`

Template:

```yaml
version: 1

gates:
  my_gate_name:
    bit: 0
    description: "Short human-readable note"
    expression: "(charge_event > 50)"
```

## Expression rules

Expressions are evaluated with `pandas.DataFrame.eval(...)`.

Use dataframe column names directly inside the expression.

Correct logic operators:

- `&` for AND
- `|` for OR
- `~` for NOT

Correct comparison operators:

- `==`
- `!=`
- `>`
- `>=`
- `<`
- `<=`

Important rule:

- always wrap each comparison in parentheses

Good:

```yaml
expression: "((charge_event > 50) | (corr_tt > 1000)) & (theta_err < 0.1)"
```

Bad:

```yaml
expression: "(charge_event > 50) or (corr_tt > 1000)"
```

Do not use Python scalar operators:

- `and`
- `or`
- `not`

These are wrong for pandas Series logic.

## Current useful STEP_2 columns

Typical columns in the current `STEP_2` files include:

- `Time`
- `raw_tt`
- `clean_tt`
- `cal_tt`
- `list_tt`
- `fit_tt`
- `corr_tt`
- `post_tt`
- `x`, `y`, `theta`, `phi`, `s`
- `x_err`, `y_err`, `theta_err`, `phi_err`, `s_err`
- `charge_event`
- `Q_P1s1` through `Q_P4s4`
- `region`

Use only columns that actually exist in the input files being processed. If an expression references a missing column, processing fails with a validation error.

## Example gate patterns

### 1. Simple threshold

```yaml
large_charge_event:
  bit: 0
  description: "Events with high total charge"
  expression: "(charge_event > 50)"
```

### 2. Range cut

```yaml
central_theta_band:
  bit: 1
  description: "Theta close to zero"
  expression: "(theta > -0.2) & (theta < 0.2)"
```

### 3. Compare two columns

```yaml
corrected_after_fit:
  bit: 2
  description: "Post-processing time greater than fit time"
  expression: "(post_tt > fit_tt)"
```

### 4. Combined logic

```yaml
good_reconstruction_high_charge:
  bit: 3
  description: "High charge with stable reconstructed track"
  expression: "((charge_event > 50) & (theta_err < 0.1)) & (phi_err < 0.1)"
```

### 5. Gate using string values

If the column is text-like:

```yaml
north_region_event:
  bit: 4
  description: "Events in one region label"
  expression: "(region == 'NORTH')"
```

Use double quotes around the full expression and single quotes inside the string comparison.

## Practical advice for setting gates

Start simple.

Good first gates are usually:

- one threshold on one variable
- one geometric cut on `theta`, `phi`, `x`, or `y`
- one quality cut using uncertainty columns such as `theta_err`

Then inspect:

- labelled event files with `gate_mask`
- the per-minute rate table
- the diagnostic plots per gate

If a gate is too broad or too narrow:

- raise or lower thresholds
- tighten or relax ranges
- add one extra condition at a time

## Common mistakes

### Missing parentheses

Wrong:

```yaml
expression: "(theta > 0) & phi > 0"
```

Right:

```yaml
expression: "(theta > 0) & (phi > 0)"
```

### Reusing a bit index

Wrong:

```yaml
gate_a:
  bit: 0
  expression: "(charge_event > 50)"

gate_b:
  bit: 0
  expression: "(theta > 0)"
```

Every gate must have a different bit.

### Using a column that does not exist

Wrong:

```yaml
expression: "(Q > 10)"
```

if the dataframe has no `Q` column.

### Assuming gates are exclusive

Gates are not exclusive unless you make them exclusive by construction.

This is valid:

- one event can satisfy `large_charge_event`
- the same event can also satisfy `positive_theta_event`

That is expected behavior for `gate_mask`.

## Recommended workflow

1. Add one or two simple gates to `gates.yaml`.
2. Run the processing command.
3. Check the output rate table for:
   - `<gate>_count`
   - `<gate>_hz`
   - `<gate>_fraction`
   - `<gate>_percent`
4. Check the diagnostic PNG for each gate.
5. Adjust thresholds and rerun.

## Example full gate file

```yaml
version: 1

gates:
  large_charge_event:
    bit: 0
    description: "Events with charge_event above a simple threshold"
    expression: "(charge_event > 50)"

  positive_theta_event:
    bit: 1
    description: "Events with positive theta"
    expression: "(theta > 0)"

  corrected_after_fit:
    bit: 2
    description: "Events with post-processing time greater than fit time"
    expression: "(post_tt > fit_tt)"

  stable_track_event:
    bit: 3
    description: "Events with small angle uncertainties"
    expression: "(theta_err < 0.1) & (phi_err < 0.1)"
```

## If a gate fails

The code will stop and report errors for cases like:

- duplicated bit indices
- missing `expression`
- bits outside `0..63`
- malformed pandas expressions
- missing dataframe columns in an expression

When that happens, simplify the gate and test again.
