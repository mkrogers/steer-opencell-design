# SPDX-FileCopyrightText: 2024-2026 Stanford University
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Benchmark and parity report for sequential vs. process-parallel sweeps.

Sweeps the cathode mass loading
(``reference_electrode_assembly.layup.cathode.mass_loading``, mg/cm²) of the
LFP flat-wound prismatic reference cell over ``n`` cases (``n - 1`` values
spread over 14-18.4 mg/cm² plus one out-of-range value, which fails as
``constraint_violation``) with every default metric, and reports:

* **Environment**: Python, OpenCell and STEER dependency versions, OS, CPU
  model and logical / physical (performance) cores, git revision.
* **Serialization**: baseline snapshot size and the time to snapshot it, save
  it, and load + verify it (what each worker does at start-up), and the
  pickled size of one case's arguments and result.
* **Process start-up**: time from creating a pool of ``N`` spawned workers
  until every worker has loaded the baseline (``_worker_init``), excluding
  any case.
* **Warm runs**: ``run_sweep`` wall time in this process after a warm-up
  (``--reps`` timed runs per configuration, after one untimed run reported as
  ``first_s``). Every parallel run starts its own pool, so a warm parallel
  run still includes process start-up.
* **Cold runs** (unless ``--no-cold``): a new interpreter per run that
  imports the package, loads the cell and runs the sweep once, timed from
  the parent (``process_s``) and around ``run_sweep`` (``sweep_s``).
* **Parity** (``--parity``): for each size and worker count, the maximum
  absolute and relative difference of every metric from the sequential run,
  and the success / failure counts.

Statistics follow ``bench_spiral_utils``: mean, p50 = ``sorted[n // 2]``,
p95 = ``sorted[round(0.95 * (n - 1))]``, min, max. No sample is discarded.
``speedup`` is the sequential p50 over the configuration's p50.
``beyond_noise`` is true only when the configuration's slowest run is faster
than the sequential fastest run (non-overlapping ranges).

Run with::

    python test/perf/bench_sweep_parallel.py
    python test/perf/bench_sweep_parallel.py --parity
    python test/perf/bench_sweep_parallel.py --workers 1,4 --sizes 100 --no-cold
    python test/perf/bench_sweep_parallel.py --update-baseline

The baseline lives at ``test/perf/BASELINE_SWEEPS.json``; warm-run rows print
their p50 delta against it. Timings are not asserted anywhere: they depend on
the machine. The file is **not** picked up by pytest (it does not start with
``test_``).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing
import os
import pickle
import platform
import resource
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Sequence

os.environ.setdefault("OPENCELL_ENV", "development")

BASELINE_PATH = os.path.join(os.path.dirname(__file__), "BASELINE_SWEEPS.json")
MASS_LOADING = "reference_electrode_assembly.layup.cathode.mass_loading"
CELL_NAME = "LFP Flat Wound Prismatic Cell"
RTOL = 1e-9
ATOL = 1e-12


def _values(count: int) -> List[float]:
    return [14.0 + 4.4 * i / (count - 2) for i in range(count - 1)] + [60.0]


def _load_cell():
    import steer_opencell_design as ocd

    return ocd.PrismaticCell.from_database(CELL_NAME, table_name="cell_references")


def _spec(count: int):
    from steer_opencell_design.Utils.Sweep import SweepAxis, SweepSpec

    return SweepSpec(axes=(SweepAxis(MASS_LOADING, _values(count)),))


def _summary(samples: Sequence[float]) -> Dict[str, float]:
    ordered = sorted(samples)
    return {
        "mean_s": statistics.fmean(ordered),
        "p50_s": ordered[len(ordered) // 2],
        "p95_s": ordered[max(0, int(round(0.95 * (len(ordered) - 1))))],
        "min_s": ordered[0],
        "max_s": ordered[-1],
        "stdev_s": statistics.stdev(ordered) if len(ordered) > 1 else 0.0,
        "samples_s": list(samples),
    }


# ─── Environment ─────────────────────────────────────────────────────────────


def _sysctl(name: str) -> Optional[str]:
    try:
        return subprocess.run(
            ["sysctl", "-n", name], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _cpu() -> Dict[str, Any]:
    info: Dict[str, Any] = {"logical_cores": os.cpu_count()}
    if sys.platform == "darwin":
        info["model"] = _sysctl("machdep.cpu.brand_string")
        info["physical_cores"] = int(_sysctl("hw.physicalcpu") or 0) or None
        performance = _sysctl("hw.perflevel0.physicalcpu")
        efficiency = _sysctl("hw.perflevel1.physicalcpu")
        if performance:
            info["performance_cores"] = int(performance)
        if efficiency:
            info["efficiency_cores"] = int(efficiency)
    elif os.path.exists("/proc/cpuinfo"):
        with open("/proc/cpuinfo") as handle:
            lines = handle.read().splitlines()
        models = [
            l.split(":", 1)[1].strip() for l in lines if l.startswith("model name")
        ]
        cores = {
            l.split(":", 1)[1].strip()
            for l in lines
            if l.startswith(("physical id", "core id"))
        }
        info["model"] = models[0] if models else platform.processor()
        info["physical_cores"] = len(cores) or None
    else:
        info["model"] = platform.processor()
    return info


def _environment() -> Dict[str, Any]:
    from steer_opencell_design.Utils import Sweep

    environment = Sweep._environment()
    environment["os"] = platform.platform()
    environment["cpu"] = _cpu()
    environment["usable_cpus"] = Sweep._usable_cpus()
    environment["start_methods"] = multiprocessing.get_all_start_methods()
    return environment


# ─── Serialization and start-up ──────────────────────────────────────────────


def _serialization(cell, reps: int) -> Dict[str, Any]:
    from steer_opencell_design.Utils.Sweep import (
        BaselineSnapshot,
        _evaluate,
        generate_cases,
    )

    snapshot_s, save_s, load_s = [], [], []
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "baseline.bin")
        for _ in range(reps):
            t0 = time.perf_counter()
            snapshot = BaselineSnapshot.from_cell(cell)
            snapshot_s.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            snapshot.save(path)
            save_s.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            BaselineSnapshot.load(path, expected_sha256=snapshot.sha256)
            load_s.append(time.perf_counter() - t0)
    spec = _spec(10)
    case = generate_cases(spec, snapshot)[0]
    result = _evaluate(snapshot, spec, case, "x" * 64)
    return {
        "payload_bytes": len(snapshot.payload),
        "snapshot": _summary(snapshot_s),
        "save": _summary(save_s),
        "load_and_verify": _summary(load_s),
        "case_arguments_pickled_bytes": len(
            pickle.dumps((case.index, case.case_id, case.parameters))
        ),
        "case_result_pickled_bytes": len(pickle.dumps(result)),
    }


def _timed_init(path, sha256, spec, sweep_id):
    from steer_opencell_design.Utils import Sweep

    Sweep._worker_init(path, sha256, spec, sweep_id)
    global _READY_AT
    _READY_AT = time.time()


def _ready_at(_: int):
    time.sleep(0.05)  # keep one worker from taking every probe
    return os.getpid(), _READY_AT


def _startup(cell, workers: int, start_method: str, reps: int) -> Dict[str, Any]:
    """Seconds from pool creation until every worker has loaded the baseline."""
    from steer_opencell_design.Utils.Sweep import BaselineSnapshot

    snapshot = BaselineSnapshot.from_cell(cell)
    spec = _spec(10)
    samples = []
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "baseline.bin")
        snapshot.save(path)
        for _ in range(reps):
            t0 = time.time()
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=workers,
                mp_context=multiprocessing.get_context(start_method),
                initializer=_timed_init,
                initargs=(path, snapshot.sha256, spec, "x" * 64),
            ) as pool:
                ready = {}
                while len(ready) < workers:
                    for pid, at in pool.map(_ready_at, range(4 * workers)):
                        ready[pid] = at
            samples.append(max(ready.values()) - t0)
    return _summary(samples)


# ─── Sweeps ──────────────────────────────────────────────────────────────────


def _warm(cell, workers: int, size: int, start_method: str, reps: int):
    from steer_opencell_design.Utils.Sweep import run_sweep

    spec = _spec(size)
    t0 = time.perf_counter()
    first = run_sweep(cell, spec, workers=workers, start_method=start_method)
    first_s = time.perf_counter() - t0
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        run_sweep(cell, spec, workers=workers, start_method=start_method)
        samples.append(time.perf_counter() - t0)
    case_s = sorted(r.elapsed_s for r in first.results)
    summary = _summary(samples)
    summary.update(
        first_s=first_s,
        cases_per_s=size / summary["p50_s"],
        median_case_s=case_s[len(case_s) // 2],
    )
    return summary


_COLD = """
import json, os, sys, time
os.environ.setdefault("OPENCELL_ENV", "development")
t0 = time.perf_counter()
sys.path.insert(0, {here!r})
from bench_sweep_parallel import _load_cell, _spec
from steer_opencell_design.Utils.Sweep import run_sweep
if __name__ == "__main__":
    cell = _load_cell()
    t1 = time.perf_counter()
    run_sweep(cell, _spec({size}), workers={workers}, start_method={method!r})
    t2 = time.perf_counter()
    print(json.dumps({{"setup_s": t1 - t0, "sweep_s": t2 - t1}}))
"""


def _cold(workers: int, size: int, start_method: str, reps: int):
    here = os.path.dirname(os.path.abspath(__file__))
    with tempfile.TemporaryDirectory() as folder:
        script = os.path.join(folder, "cold_run.py")
        with open(script, "w") as handle:
            handle.write(
                _COLD.format(here=here, size=size, workers=workers, method=start_method)
            )
        process_s, sweep_s, setup_s = [], [], []
        for _ in range(reps):
            t0 = time.perf_counter()
            out = subprocess.run(
                [sys.executable, script], capture_output=True, text=True, check=True
            ).stdout
            process_s.append(time.perf_counter() - t0)
            timing = json.loads(out.strip().splitlines()[-1])
            sweep_s.append(timing["sweep_s"])
            setup_s.append(timing["setup_s"])
    return {
        "process": _summary(process_s),
        "sweep": _summary(sweep_s),
        "import_and_load": _summary(setup_s),
    }


def _parity(cell, workers: Sequence[int], sizes: Sequence[int], start_method: str):
    from steer_opencell_design.Utils.Sweep import run_sweep

    rows = []
    for size in sizes:
        spec = _spec(size)
        reference = run_sweep(cell, spec)
        for count in workers:
            if count == 1:
                continue
            result = run_sweep(cell, spec, workers=count, start_method=start_method)
            row: Dict[str, Any] = {
                "cases": size,
                "workers": count,
                "same_sweep_id": result.sweep_id == reference.sweep_id,
                "same_case_ids_and_order": [r.case_id for r in result.results]
                == [r.case_id for r in reference.results],
                "same_statuses": [r.status for r in result.results]
                == [r.status for r in reference.results],
                "same_metric_units": all(
                    a.metric_units == b.metric_units
                    for a, b in zip(result.results, reference.results)
                ),
                "succeeded": (len(reference.succeeded), len(result.succeeded)),
                "failed": (len(reference.failed), len(result.failed)),
                "matches": result.matches(reference, rtol=RTOL, atol=ATOL) == [],
                "metrics": {},
            }
            for name in spec.metrics:
                abs_err = rel_err = 0.0
                for a, b in zip(result.results, reference.results):
                    if name in b.metrics:
                        diff = abs(a.metrics[name] - b.metrics[name])
                        abs_err = max(abs_err, diff)
                        if b.metrics[name] != 0:
                            rel_err = max(rel_err, diff / abs(b.metrics[name]))
                row["metrics"][name] = {"max_abs": abs_err, "max_rel": rel_err}
            rows.append(row)
    return rows


# ─── Main ────────────────────────────────────────────────────────────────────


def _ints(text: str) -> List[int]:
    return [int(part) for part in text.split(",") if part]


def _max_rss_mb(who: int) -> float:
    rss = resource.getrusage(who).ru_maxrss
    return rss / (1024 * 1024) if sys.platform == "darwin" else rss / 1024


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--workers", type=_ints, default=[1, 2, 4, 8])
    parser.add_argument("--sizes", type=_ints, default=[10, 50, 100])
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument(
        "--start-method", default="spawn", choices=("spawn", "forkserver")
    )
    parser.add_argument("--no-cold", action="store_true", help="skip cold runs")
    parser.add_argument(
        "--parity", action="store_true", help="print the parity report only"
    )
    parser.add_argument("--json", help="also write all measurements to this file")
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help=f"overwrite {os.path.basename(BASELINE_PATH)} with this run",
    )
    args = parser.parse_args(argv)
    if 1 not in args.workers and not args.parity:
        parser.error("--workers must include 1 (the sequential reference)")
    if args.reps < 1:
        parser.error("--reps must be at least 1")

    from steer_opencell_design.Utils import Sweep

    usable = Sweep._usable_cpus()
    skipped = [w for w in args.workers if w > usable]
    workers = sorted({w for w in args.workers if w <= usable})
    if skipped:
        print(f"# Skipping workers={skipped}: only {usable} CPUs are usable.")

    # Warm-up: import, load the cell, and evaluate one sequential sweep so the
    # numba kernels are loaded from their on-disk cache before timing.
    print("# Warming up...", flush=True)
    cell = _load_cell()
    Sweep.run_sweep(cell, _spec(3))
    environment = _environment()
    print(json.dumps({"environment": environment}, indent=2, sort_keys=True))

    if args.parity:
        rows = _parity(cell, workers, args.sizes, args.start_method)
        print("\ncases,workers,metric,max_abs_error,max_rel_error")
        for row in rows:
            for name, err in row["metrics"].items():
                print(
                    f"{row['cases']},{row['workers']},{name},"
                    f"{err['max_abs']:.3e},{err['max_rel']:.3e}"
                )
        print(
            "\ncases,workers,same_sweep_id,same_case_ids_and_order,same_statuses,"
            "same_metric_units,succeeded(seq/par),failed(seq/par),matches"
        )
        for row in rows:
            print(
                f"{row['cases']},{row['workers']},{row['same_sweep_id']},"
                f"{row['same_case_ids_and_order']},{row['same_statuses']},"
                f"{row['same_metric_units']},"
                f"{row['succeeded'][0]}/{row['succeeded'][1]},"
                f"{row['failed'][0]}/{row['failed'][1]},{row['matches']}"
            )
        if args.json:
            with open(args.json, "w") as handle:
                json.dump(
                    {"environment": environment, "parity": rows},
                    handle,
                    indent=2,
                    sort_keys=True,
                )
        ok = all(r["matches"] and r["same_metric_units"] for r in rows)
        return 0 if ok else 1

    report: Dict[str, Any] = {
        "environment": environment,
        "method": {
            "warmup": "import, load the cell, run one 3-case sequential sweep; "
            "then one untimed run per configuration (reported as first_s)",
            "repetitions": args.reps,
            "clock": "time.perf_counter around run_sweep (cold: around the "
            "subprocess); no sample discarded",
            "start_method": args.start_method,
            "workload": f"{MASS_LOADING} over n values, all default metrics",
        },
    }

    print("# Serialization...", flush=True)
    report["serialization"] = _serialization(cell, args.reps)
    s = report["serialization"]
    print(
        f"payload_bytes={s['payload_bytes']} "
        f"snapshot_p50_s={s['snapshot']['p50_s']:.4f} "
        f"save_p50_s={s['save']['p50_s']:.4f} "
        f"load_and_verify_p50_s={s['load_and_verify']['p50_s']:.4f} "
        f"case_args_bytes={s['case_arguments_pickled_bytes']} "
        f"case_result_bytes={s['case_result_pickled_bytes']}"
    )

    print("\n# Process start-up (pool creation until every worker is ready)")
    print("workers,mean_s,p50_s,p95_s,min_s,max_s")
    report["startup"] = {}
    for count in workers:
        if count == 1:
            continue
        summary = _startup(cell, count, args.start_method, args.reps)
        report["startup"][str(count)] = summary
        print(
            f"{count},{summary['mean_s']:.3f},{summary['p50_s']:.3f},"
            f"{summary['p95_s']:.3f},{summary['min_s']:.3f},{summary['max_s']:.3f}"
        )

    baseline = {}
    if os.path.exists(BASELINE_PATH):
        with open(BASELINE_PATH) as handle:
            baseline = json.load(handle).get("warm", {})

    print("\n# Warm runs")
    print(
        "cases,workers,first_s,mean_s,p50_s,p95_s,min_s,max_s,stdev_s,"
        "cases_per_s,median_case_s,speedup,beyond_noise,delta_vs_baseline_pct"
    )
    report["warm"] = {}
    for size in args.sizes:
        rows = report["warm"].setdefault(str(size), {})
        for count in workers:
            summary = _warm(cell, count, size, args.start_method, args.reps)
            rows[str(count)] = summary
            sequential = rows["1"]
            summary["speedup"] = sequential["p50_s"] / summary["p50_s"]
            summary["beyond_noise"] = (
                count != 1 and summary["max_s"] < sequential["min_s"]
            )
            previous = baseline.get(str(size), {}).get(str(count))
            delta = (
                f"{100 * (summary['p50_s'] / previous['p50_s'] - 1):+.1f}"
                if previous
                else "n/a"
            )
            print(
                f"{size},{count},{summary['first_s']:.3f},{summary['mean_s']:.3f},"
                f"{summary['p50_s']:.3f},{summary['p95_s']:.3f},"
                f"{summary['min_s']:.3f},{summary['max_s']:.3f},"
                f"{summary['stdev_s']:.3f},{summary['cases_per_s']:.1f},"
                f"{summary['median_case_s']:.4f},{summary['speedup']:.2f},"
                f"{summary['beyond_noise']},{delta}",
                flush=True,
            )

    if not args.no_cold:
        print("\n# Cold runs (new interpreter per run)")
        print(
            "cases,workers,process_p50_s,import_and_load_p50_s,sweep_p50_s,"
            "sweep_min_s,sweep_max_s"
        )
        report["cold"] = {}
        for size in args.sizes:
            for count in workers:
                summary = _cold(count, size, args.start_method, args.reps)
                report["cold"].setdefault(str(size), {})[str(count)] = summary
                print(
                    f"{size},{count},{summary['process']['p50_s']:.3f},"
                    f"{summary['import_and_load']['p50_s']:.3f},"
                    f"{summary['sweep']['p50_s']:.3f},"
                    f"{summary['sweep']['min_s']:.3f},"
                    f"{summary['sweep']['max_s']:.3f}",
                    flush=True,
                )

    # Peak resident memory of this process over the whole run, and the largest
    # peak of any finished child: pool workers and, unless --no-cold, the cold
    # runs' interpreters, which run whole sweeps themselves.
    report["resources"] = {
        "parent_peak_rss_mb": _max_rss_mb(resource.RUSAGE_SELF),
        "largest_child_peak_rss_mb_workers_or_cold_runs": _max_rss_mb(
            resource.RUSAGE_CHILDREN
        ),
    }
    print(f"\n# Resources: {json.dumps(report['resources'])}")

    if args.json:
        with open(args.json, "w") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
    if args.update_baseline:
        with open(BASELINE_PATH, "w") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
        print(f"\n# Wrote baseline to {BASELINE_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
