# Parameter Sweeps

`steer_opencell_design.Utils.Sweep` evaluates a cell at every combination of a
few parameter values (a full-factorial sweep) and records cell-level metrics
for each case, with enough provenance to replay the sweep later and check that
it gives the same numbers.

Every case starts from the same baseline: the cell is serialized once, and each
case deserializes its own, independent copy, changes it through the model's
public setters, and propagates the change to the cell. The cell you pass in is
only ever serialized, never changed.

Cases run one after another in the calling process by default
(`workers=1`); `workers=N` runs them in `N` worker processes and returns the
same result. See [Parallel execution](#parallel-execution) for when that is
worth it.

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

`matches` compares the sweep ID, the baseline SHA-256, case IDs and order,
parameters, statuses, error categories, metric units and every metric within
`rtol=1e-9, atol=1e-12`; it ignores timings, the environment block and how the
sweep was executed.

## Parallel execution

```python
# A script must guard its entry point: spawned workers re-import it.
if __name__ == "__main__":
    cell = ocd.PrismaticCell.from_database(
        "LFP Flat Wound Prismatic Cell", table_name="cell_references"
    )
    sequential = run_sweep(cell, spec)             # workers=1, the reference
    parallel = run_sweep(cell, spec, workers=4)    # 4 worker processes

    # Same sweep ID, case IDs, order, statuses and metrics:
    print(parallel.matches(sequential))            # []
    print(parallel.manifest["execution"])
    # {'mode': 'process_pool', 'workers': 4, 'processes': 4, 'start_method': 'spawn'}
```

`workers=N` evaluates the cases in a standard-library
`concurrent.futures.ProcessPoolExecutor`:

- Workers are started with `spawn` on every platform (`start_method="forkserver"`
  is also accepted where available; `fork` is refused), so nothing is inherited
  from the parent process — no open database connection, cache or live cell.
- The baseline is written once to a private temporary file; each worker loads
  it, checks it against the baseline SHA-256 and then rebuilds an independent
  cell for every case, with the same function `workers=1` uses. Only the case
  index, case ID and parameter values (about 100 bytes) go to a worker per case.
- At most `2 * N` cases are queued at once, and results are put back in
  case-index order, so the result never depends on which worker finishes first.
- `workers` must be a positive integer no larger than the number of CPUs the
  process may use; a sweep with fewer cases than workers starts only as many
  workers as there are cases, and a one-case sweep starts none.
- The pool is shut down and its processes joined, and the temporary file
  removed, when the sweep finishes, fails or is interrupted.

`manifest["execution"]` records how a sweep was run. Like the environment
block, it is not part of the sweep ID, so a sequential and a parallel run of
the same sweep have the same ID and case IDs.

On one machine, parallel results are bit-identical to sequential ones: for the
LFP example above, 10, 50 and 100 cases on 2, 4 and 8 workers differ from
`workers=1` by exactly 0 in every metric. Across machines, compare with
`matches()` (see [Reproducibility](#reproducibility)).

### When parallel execution helps

Each parallel sweep starts its own worker processes, and each worker imports
the package and loads the baseline before evaluating anything. Whether that
pays off depends on how long the cases take and on the machine; measure it with
`test/perf/bench_sweep_parallel.py` (see [Performance](#performance)).
Parallelism does not scale linearly, and for small sweeps the sequential
default is faster.

!!! note "The one measured case where parallel was faster"
    The default is `workers=1`. In the benchmark below (LFP flat-wound
    prismatic cell, Apple M1), **100 cases on 4 workers** was the only
    configuration faster than sequential by more than the run-to-run
    variation: 2.85 s against 3.42 s, a 1.20× speedup. Every 10- and 50-case
    sweep was slower in parallel, 100 cases on 2 workers was within the noise,
    and 8 workers never helped. Use `workers > 1` only for sweeps of roughly
    100 or more cases of this size, and measure on your own hardware.

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

## Error handling

A failed case has no metrics; nothing is substituted, and the other cases are
unaffected. Its `error_category` is one of the following, and `error_kind`
groups the categories:

| Category | `error_kind` | Meaning |
|---|---|---|
| `constraint_violation` | `invalid_parameters` | The value is outside the property's `<name>_range`. |
| `value_not_realized` | `invalid_parameters` | A later axis changed this axis's value (e.g. setting `coating_thickness` after `mass_loading` while the calender density is held). |
| `resolution_error` | `invalid_parameters` | The path could not be resolved, or the value read back, on the case's cell. |
| `constraint_error` | `model_evaluation` | The `<name>_range` could not be evaluated. |
| `setter_error` | `model_evaluation` | The property setter rejected the value. |
| `propagation_error` | `model_evaluation` | Propagating the change to the cell raised. |
| `metric_error` / `non_finite_metric` | `model_evaluation` | A metric raised, or was NaN or infinite. |
| `reconstruction_error` | `serialization` | The case's cell could not be deserialized from the baseline. |

```python
for case in result.failed:
    print(case.case_id, case.error_kind, case.error_category, case.error_message)
```

Three kinds of problem are errors rather than failed cases:

- **An invalid specification** (unknown path or metric, a non-finite or
  non-numeric value, a duplicated value or axis, an empty axis, more cases than
  `max_cases`, an invalid `workers` or `start_method`) raises
  `SweepSpecificationError` before any case runs. `max_cases` defaults to 1000;
  pass a larger value to run a larger sweep.
- **A baseline that cannot be snapshotted** reproducibly, or whose metrics
  cannot be read, raises `BaselineSnapshotError` before any case runs.
- **An infrastructure failure** — a worker process that dies, a pool that
  cannot start, an exception that escapes case evaluation — raises
  `SweepExecutionError`. It is never turned into a case result, and the sweep
  does not fall back to sequential execution. The error keeps what is known:

```python
from steer_opencell_design.Utils.Sweep import SweepExecutionError

try:
    result = run_sweep(cell, spec, workers=4)
except SweepExecutionError as error:
    print(error)                      # what happened, and how many cases finished
    print(error.__cause__)            # the underlying exception (worker traceback included)
    finished = error.partial          # a SweepResult with the finished cases, in order
    print(error.in_flight_case_ids)   # running or queued when it failed
    print(error.unfinished_case_ids)  # every case without a result
```

A worker that dies takes the whole pool with it (`BrokenProcessPool`), so the
cases that were queued or running are reported as unfinished; the case that
caused the crash is among `in_flight_case_ids`.

## Exporting results

`to_json` writes the whole result — manifest and every case — and `to_csv` a
flat table with one row per case and the units in the column names
(`param:<path>[unit]`, `realized:<path>[unit]`, `metric:<name>[unit]`):

```python
result.to_json("sweep.json")
result.to_csv("sweep.csv")
rows = SweepResult.read_csv("sweep.csv")   # typed values, same as result.to_rows()
```

The JSON layout is versioned by `manifest["schema_version"]`:

```text
{
  "manifest": {
    "schema_version": 1,
    "sweep_id": "<sha256>",
    "canonical":     {... hashed into sweep_id: cell class, baseline sha256, axes, metrics ...},
    "specification": {... SweepSpec.to_dict() ...},
    "baseline":      {"sha256", "cell_class", "payload_bytes", "metrics"},
    "environment":   {... versions, platform, git ...},   # not hashed
    "execution":     {"mode", "workers", "processes", "start_method"}  # not hashed
  },
  "results": [
    {"case_index", "case_id", "sweep_id", "parameters", "parameter_units",
     "realized_parameters", "metrics", "metric_units", "status",
     "error_category", "error_message", "elapsed_s"},
    ...
  ]
}
```

These files are meant for other tools, such as a future web front end, to
read. No OpenCell application reads them yet; a consumer should check
`schema_version` and treat `canonical` plus the per-case fields other than
`elapsed_s` as the comparable data.

## Performance

`test/perf/bench_sweep_parallel.py` sweeps the cathode mass loading of the LFP
flat-wound prismatic cell (all default metrics; one case in each sweep is out
of range) and times `run_sweep` for 1, 2, 4 and 8 workers. On an Apple M1
(4 performance + 4 efficiency cores, 16 GB, macOS 26.4, Python 3.11.18,
numba 0.68.1), p50 of 5 runs after a warm-up, in seconds:

| Cases | `workers=1` | 2 | 4 | 8 |
|---:|---:|---:|---:|---:|
| 10 | **0.42** | 1.47 | 1.71 | 2.46 |
| 50 | **1.70** | 2.26 | 2.21 | 2.88 |
| 100 | 3.42 | 3.16 | **2.85** (1.20×) | 3.35 |

- A case takes about 35 ms sequentially. Starting a pool until every worker
  has loaded the baseline takes about 0.9 s (2 workers), 1.0 s (4) and 1.5 s
  (8), most of it importing the package in each new interpreter.
- Only 100 cases on 4 workers was faster than sequential by more than the
  run-to-run variation (slowest parallel run 2.92 s, fastest sequential run
  3.31 s). Below roughly 100 cases of this size, the default `workers=1` is
  faster.
- Cases slow down as workers are added (median 35 ms on 1 worker, 48 ms on 4,
  87 ms on 8), because this CPU has only four performance cores; 8 workers
  never helped.
- Each worker held about 0.7 GB of resident memory for this cell, so 8 workers
  need about 6 GB on top of the calling process.
- In a new interpreter, importing the package and loading the cell adds about
  0.75 s before any sweep, the same for either mode.

These numbers are specific to this cell and machine. Cases that take longer
(larger designs, more expensive metrics) amortize the start-up better; run the
benchmark on your own hardware before choosing `workers`.

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
  git revision, `OPENCELL_ENV`), `manifest["execution"]` (sequential or
  process pool, worker count, start method) and the baseline's own metrics are
  recorded beside it but are not hashed, so the same sweep has the same ID on
  any machine and with any number of workers.
- **Floating point.** On one machine and environment, repeated runs give
  bit-identical metrics. The winding kernels are compiled with `fastmath`, so
  other CPUs or library versions can differ in the last digits; compare with
  `matches()` (`rtol=1e-9`, `atol=1e-12`). A value read back can differ from
  the value set in the last digit because of unit conversion
  (`15.0` mg/cm² reads back as `14.999999999999996`); `realized_parameters`
  records what the model holds.

## Limitations

- Every parallel sweep starts new worker processes; there is no persistent
  pool across sweeps. Each worker holds its own copy of the model in memory.
- A finished case's cell is freed by Python's cyclic garbage collector, not
  when the case ends (the model's parent references form cycles), so a
  process's memory grows between collections on long sweeps. On the LFP cell,
  400 sequential cases peaked at about 1.7 GB in one process; a full
  collection after every case would hold it near 0.56 GB but costs about
  34 ms, roughly the time of a case, so the sweep does not force one.
- Parallel execution needs an importable entry point: guard scripts with
  `if __name__ == "__main__":`. Interactive sessions work as long as nothing
  the workers need is defined only in the notebook.
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
