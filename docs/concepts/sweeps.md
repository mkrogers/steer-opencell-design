# Parameter Sweeps

`steer_opencell_design.Utils.Sweep` evaluates a cell at every combination of a
few parameter values (a full-factorial sweep) and records cell-level metrics
for each case, with enough provenance to replay the sweep later and check that
it gives the same numbers.

Every case starts from the same baseline: the cell is serialized once, and each
case deserializes its own, independent copy, changes it through the model's
public setters, and propagates the change to the cell. The cell you pass in is
only ever serialized, never changed.

## Example: cathode mass loading of the LFP flat-wound prismatic cell

```python
import steer_opencell_design as ocd
from steer_opencell_design.Utils.Sweep import (
    BaselineSnapshot, SweepAxis, SweepResult, SweepSpec, run_sweep,
)

# 1. Load the reference cell
cell = ocd.PrismaticCell.from_database(
    "LFP Flat Wound Prismatic Cell", table_name="cell_references"
)
cathode = cell.reference_electrode_assembly.layup.cathode
print(cathode.mass_loading, cathode.mass_loading_range)
# 17.771146850147872 (0.52, 52.0)        mg/cm²

# 2. Configure the sweep
spec = SweepSpec(
    axes=(
        SweepAxis(
            "reference_electrode_assembly.layup.cathode.mass_loading",
            (15.0, 16.0, 17.0, 18.0, 60.0),
        ),
    ),
    metrics=("reversible_capacity", "energy", "mass", "cost",
             "specific_energy", "volumetric_energy", "cost_per_energy"),
)

# 3. Run sequentially
result = run_sweep(cell, spec)

# 4. Inspect the outputs
for case in result.results:
    (path, value), = case.parameters
    if case.ok:
        print(f"{value:5.1f} mg/cm²  {case.metrics['energy']:7.2f} Wh  "
              f"{case.metrics['specific_energy']:6.2f} Wh/kg  "
              f"{case.metrics['cost_per_energy']:5.2f} $/kWh")
    else:
        print(f"{value:5.1f} mg/cm²  {case.error_category}: {case.error_message}")
```

```text
 15.0 mg/cm²   194.99 Wh  114.06 Wh/kg  40.36 $/kWh
 16.0 mg/cm²   209.48 Wh  120.06 Wh/kg  38.70 $/kWh
 17.0 mg/cm²   223.97 Wh  125.82 Wh/kg  37.26 $/kWh
 18.0 mg/cm²   238.47 Wh  131.36 Wh/kg  35.99 $/kWh
 60.0 mg/cm²  constraint_violation: reference_electrode_assembly.layup.cathode.mass_loading = 60.0 is outside its allowed range [0.52, 52.0] mg/cm².
```

The 60 mg/cm² case fails on its own; the cases around it are unaffected.

```python
# 5. Export: the baseline bytes, the full result (manifest + cases), and a table
BaselineSnapshot.from_cell(cell).save("baseline.bin")
result.to_json("sweep.json")
result.to_csv("sweep.csv")

# 6. Replay from the saved files and compare
saved = SweepResult.from_json("sweep.json")
baseline = BaselineSnapshot.load(
    "baseline.bin", expected_sha256=saved.manifest["baseline"]["sha256"]
)
replayed = run_sweep(baseline, SweepSpec.from_dict(saved.manifest["specification"]))
print(replayed.matches(saved))  # [] -> same sweep ID, cases and metrics
```

`matches` compares case IDs, statuses and every metric within
`rtol=1e-9, atol=1e-12`; it ignores timings and the environment block.

## Parameters

A parameter is a dotted path of public properties from the cell, ending in a
settable numeric property — for example
`reference_electrode_assembly.layup.cathode.mass_loading` or
`electrolyte_overfill`. Paths are checked against the baseline before anything
runs; a path is rejected if:

- a segment is not a lowercase public name (no leading `_`, no indexing or calls);
- a segment is not a property of the object it is read from;
- an intermediate value is not a model component (e.g. the `electrode_assemblies` list);
- the last property is read-only or not a number;
- an object on the path is not parented to the one before it, so the change
  would not propagate to the cell.

Values are in the property's public units (see [Units](units.md)). Axes are
applied in the order given; each change is propagated to the cell
(`propagate_changes()`) before the next is applied. Where the owning object
defines `<name>_range`, the value is checked against it, as computed on that
case's cell, before it is set.

## Results

`run_sweep` returns a `SweepResult` with a `manifest` and one `CaseResult` per
case, in case-index order. A case result holds the sweep ID, case ID and index,
the parameter values and units, the values read back after all parameters were
applied, the metrics and their units, the status (`ok` or `failed`), the error
category and message for a failed case, and the elapsed time.

Supported metrics (all from the cell):

| Metric | Unit |
|---|---|
| `reversible_capacity`, `irreversible_capacity` | Ah |
| `energy` | Wh |
| `mass` | g |
| `cost` | $ |
| `volume` | L |
| `specific_energy` | Wh/kg |
| `volumetric_energy` | Wh/L |
| `cost_per_energy` | $/kWh |
| `minimum_operating_voltage`, `maximum_operating_voltage` | V |

A failed case has no metrics; nothing is substituted. Its `error_category` is
one of:

| Category | Meaning |
|---|---|
| `constraint_violation` | The value is outside the property's `<name>_range`. |
| `constraint_error` | The `<name>_range` could not be evaluated. |
| `setter_error` | The property setter rejected the value. |
| `propagation_error` | Propagating the change to the cell raised. |
| `value_not_realized` | A later axis changed this axis's value (e.g. setting `coating_thickness` after `mass_loading` while the calender density is held). |
| `metric_error` / `non_finite_metric` | A metric raised, or was NaN or infinite. |
| `resolution_error` / `reconstruction_error` | The case's cell could not be rebuilt or its path resolved. |

An invalid specification (unknown path or metric, a non-finite or non-numeric
value, a duplicated value or axis, an empty axis, more cases than `max_cases`)
raises `SweepSpecificationError` before any case runs. `max_cases` defaults to
1000; pass a larger value to run a larger sweep.

## Reproducibility

- **Baseline fingerprint.** The baseline is `cell.serialize(compress=False)`.
  It is accepted only if deserializing and re-serializing it gives the same
  bytes; its SHA-256 identifies the baseline.
- **Enumeration.** `itertools.product` over the axes in the order given: the
  first axis varies slowest. Values keep the order given.
- **Canonical values.** Integers stay integers, other reals become `float`,
  `-0.0` becomes `0.0`; booleans, strings, NaN and infinities are rejected.
  Canonical JSON uses sorted keys and compact separators, so floats appear as
  their shortest round-tripping `repr`, identical on every platform.
- **Sweep ID.** SHA-256 of the canonical JSON of the schema version, cell
  class, baseline SHA-256, parameter order, axes (path, unit, values), metrics
  (name, unit) and case count.
- **Case ID.** The first 16 hex characters of the SHA-256 of the schema
  version, cell class, baseline SHA-256 and the case's `[path, value]` pairs in
  axis order. It does not depend on execution order or on the other cases.
- **Manifest.** `manifest["canonical"]` is what the sweep ID hashes.
  `manifest["environment"]` (package and dependency versions, Python, platform,
  git revision, `OPENCELL_ENV`) and the baseline's own metrics are recorded
  beside it but are not hashed, so the same sweep has the same ID on any machine.
- **Floating point.** On one machine and environment, repeated runs give
  bit-identical metrics. The winding kernels are compiled with `fastmath`, so
  other CPUs or library versions can differ in the last digits; compare with
  `matches()` (`rtol=1e-9`, `atol=1e-12`). A value read back can differ from
  the value set in the last digit because of unit conversion
  (`15.0` mg/cm² reads back as `14.999999999999996`); `realized_parameters`
  records what the model holds.

## Limitations

- Sweeps run sequentially in the calling process.
- The container is not resized. A parameter that grows the electrode assemblies
  can leave them larger than the container, and the model reports no error: in
  the example above, mass loadings above about 18.45 mg/cm² make the four flat
  jelly rolls thicker than the can's internal length (58.98 mm). Check the
  assembly dimensions against `encapsulation.internal_*` if that matters.
- A `<name>_range` is enforced only where the owning object defines one.
- Only numeric properties can be swept; components (materials, formulations)
  cannot.
- The baseline file written by `BaselineSnapshot.save` is the package's own
  serialization format; load only files you trust.
