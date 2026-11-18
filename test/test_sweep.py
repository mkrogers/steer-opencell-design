# SPDX-FileCopyrightText: 2024-2026 Stanford University
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for steer_opencell_design.Utils.Sweep.

Numerical equivalence to a hand-evaluated case uses rtol=1e-9, atol=1e-12;
repeated runs on one machine are required to be bit-identical.
"""

import concurrent.futures
import hashlib
import json
import math
import multiprocessing
import os
import tempfile
import time
import unittest
from unittest import mock

import numpy as np

import steer_opencell_design as ocd
from steer_opencell_design.Utils import Sweep
from steer_opencell_design.Utils.Sweep import SweepAxis, SweepSpec, run_sweep

RTOL = 1e-9
ATOL = 1e-12

CATHODE = "reference_electrode_assembly.layup.cathode"
MASS_LOADING = f"{CATHODE}.mass_loading"
CALENDER_DENSITY = f"{CATHODE}.calender_density"
COATING_THICKNESS = f"{CATHODE}.coating_thickness"


def _load_cell():
    return ocd.PrismaticCell.from_database(
        "LFP Flat Wound Prismatic Cell", table_name="cell_references"
    )


def _parity_values(count):
    """``count`` cathode mass loadings: ``count - 1`` spread over 14-18.4 mg/cm²
    (where the jelly rolls still fit the can), then one out of range."""
    return [14.0 + 4.4 * i / (count - 2) for i in range(count - 1)] + [60.0]


# Replacements for ``Sweep._worker_run`` / ``Sweep._worker_init`` that inject
# faults into real worker processes. They are top-level, so a spawned worker
# imports them from this module by name.
def _crash_on_case_3(index, case_id, parameters):
    if index == 3:
        os._exit(13)
    return Sweep._worker_run(index, case_id, parameters)


def _raise_on_case_2(index, case_id, parameters):
    if index == 2:
        raise ValueError("injected worker fault")
    return Sweep._worker_run(index, case_id, parameters)


def _finish_in_reverse(index, case_id, parameters):
    time.sleep(0.05 * (8 - index))
    return Sweep._worker_run(index, case_id, parameters)


def _answer_for_another_case(index, case_id, parameters):
    result = Sweep._worker_run(index, case_id, parameters)
    return Sweep.CaseResult.from_dict(dict(result.to_dict(), case_index=index + 100))


def _failing_init(path, sha256, spec, sweep_id):
    raise RuntimeError("injected initializer fault")


def _evaluate_by_hand(cell, parameters, metrics):
    """Apply a case the documented way: set, then propagate, on a fresh copy."""
    fresh = type(cell).deserialize(cell.serialize())
    for path, value in parameters:
        *parents, name = path.split(".")
        owner = fresh
        for segment in parents:
            owner = getattr(owner, segment)
        setattr(owner, name, value)
        if owner is not fresh:
            owner.propagate_changes()
    return {name: float(getattr(fresh, name)) for name in metrics}


class TestSweepAxis(unittest.TestCase):
    def test_values_are_canonicalized_in_order(self):
        axis = SweepAxis(MASS_LOADING, [np.float64(17.5), np.int64(16), -0.0, 18])
        self.assertEqual(axis.values, (17.5, 16, 0.0, 18))
        self.assertEqual([type(v) for v in axis.values], [float, int, float, int])
        self.assertEqual(math.copysign(1.0, axis.values[2]), 1.0)

    def test_default_and_explicit_units(self):
        self.assertEqual(SweepAxis(MASS_LOADING, (1.0,)).unit, "mg/cm²")
        self.assertEqual(SweepAxis(CALENDER_DENSITY, (1.0,)).unit, "g/cm³")
        self.assertIsNone(SweepAxis("n_electrode_assembly", (1,)).unit)
        self.assertEqual(SweepAxis("n_electrode_assembly", (1,), unit="-").unit, "-")
        with self.assertRaises(Sweep.SweepSpecificationError):
            SweepAxis(MASS_LOADING, (1.0,), unit=3)

    def test_invalid_values_are_rejected(self):
        for values in (
            (),
            (True,),
            ("17",),
            (None,),
            (float("nan"),),
            (float("inf"),),
            (1 + 2j,),
            (17.0, 17),
            "17",
            17.0,
        ):
            with (
                self.subTest(values=values),
                self.assertRaises(Sweep.SweepSpecificationError),
            ):
                SweepAxis(MASS_LOADING, values)

    def test_invalid_paths_are_rejected(self):
        for path in (
            "",
            None,
            3,
            "_mass_loading",
            f"{CATHODE}._mass_loading",
            "__class__",
            "a..b",
            "Cathode",
            "electrode_assemblies[0]",
            "a.b()",
            "a;b",
            " mass_loading",
        ):
            with (
                self.subTest(path=path),
                self.assertRaises(Sweep.SweepSpecificationError),
            ):
                SweepAxis(path, (1.0,))

    def test_round_trip(self):
        axis = SweepAxis(MASS_LOADING, (16.0, 17.0))
        self.assertEqual(SweepAxis.from_dict(axis.to_dict()), axis)
        with self.assertRaises(Sweep.SweepSpecificationError):
            SweepAxis.from_dict({"path": MASS_LOADING, "values": [1.0], "extra": 1})


class TestSweepSpec(unittest.TestCase):
    def test_counts_and_order(self):
        spec = SweepSpec(
            axes=(
                SweepAxis(MASS_LOADING, (1.0, 2.0, 3.0)),
                SweepAxis(CALENDER_DENSITY, (4.0, 5.0)),
            )
        )
        self.assertEqual(spec.case_count, 6)
        self.assertEqual(spec.parameter_order, (MASS_LOADING, CALENDER_DENSITY))
        self.assertEqual(spec.metrics, Sweep.DEFAULT_METRICS)
        self.assertEqual(spec.max_cases, Sweep.DEFAULT_MAX_CASES)

    def test_invalid_specifications_are_rejected(self):
        axis = SweepAxis(MASS_LOADING, (1.0,))
        cases = {
            "no axes": dict(axes=()),
            "axes not a sequence": dict(axes="abc"),
            "not an axis": dict(axes=({"path": MASS_LOADING, "values": [1.0]},)),
            "duplicate axis": dict(axes=(axis, SweepAxis(MASS_LOADING, (2.0,)))),
            "unknown metric": dict(axes=(axis,), metrics=("energy", "colour")),
            "unhashable metric": dict(axes=(axis,), metrics=(["energy"],)),
            "metric string": dict(axes=(axis,), metrics="energy"),
            "no metrics": dict(axes=(axis,), metrics=()),
            "duplicate metric": dict(axes=(axis,), metrics=("energy", "energy")),
            "zero max": dict(axes=(axis,), max_cases=0),
            "bool max": dict(axes=(axis,), max_cases=True),
            "float max": dict(axes=(axis,), max_cases=10.0),
            "schema": dict(axes=(axis,), schema_version=2),
            "bool schema": dict(axes=(axis,), schema_version=True),
        }
        for label, kwargs in cases.items():
            with self.subTest(label), self.assertRaises(Sweep.SweepSpecificationError):
                SweepSpec(**kwargs)

    def test_case_count_is_bounded_unless_raised_explicitly(self):
        axes = (
            SweepAxis(MASS_LOADING, tuple(range(1, 41))),
            SweepAxis(CALENDER_DENSITY, tuple(range(1, 31))),
        )
        with self.assertRaisesRegex(Sweep.SweepSpecificationError, "1200 cases"):
            SweepSpec(axes=axes)
        self.assertEqual(SweepSpec(axes=axes, max_cases=1200).case_count, 1200)

    def test_round_trip(self):
        spec = SweepSpec(
            axes=(SweepAxis(MASS_LOADING, (16.0, 17.0)),),
            metrics=("energy", "mass"),
            max_cases=5,
        )
        self.assertEqual(
            SweepSpec.from_dict(json.loads(json.dumps(spec.to_dict()))), spec
        )
        for bad in ({}, {"axes": "x"}, {"axes": [], "other": 1}, []):
            with (
                self.subTest(bad=bad),
                self.assertRaises(Sweep.SweepSpecificationError),
            ):
                SweepSpec.from_dict(bad)


class TestLFPFlatWoundSweep(unittest.TestCase):
    """A realistic sweep of the LFP flat-wound prismatic reference cell."""

    @classmethod
    def setUpClass(cls):
        cls.cell = _load_cell()
        cls.baseline_bytes = cls.cell.serialize(compress=False)
        cls.baseline_mass_loading = (
            cls.cell.reference_electrode_assembly.layup.cathode.mass_loading
        )
        cls.spec = SweepSpec(
            axes=(
                SweepAxis(MASS_LOADING, (16.0, 17.0, 19.5)),
                SweepAxis(CALENDER_DENSITY, (2.5, 2.6)),
            )
        )
        cls.result = run_sweep(cls.cell, cls.spec)

    def test_baseline_is_unchanged(self):
        self.assertEqual(self.cell.serialize(compress=False), self.baseline_bytes)
        self.assertEqual(
            self.result.manifest["baseline"]["sha256"],
            hashlib.sha256(self.baseline_bytes).hexdigest(),
        )

    def test_full_factorial_enumeration(self):
        self.assertEqual(len(self.result.results), 6)
        self.assertEqual([r.case_index for r in self.result.results], list(range(6)))
        expected = [
            ((MASS_LOADING, m), (CALENDER_DENSITY, d))
            for m in (16.0, 17.0, 19.5)
            for d in (2.5, 2.6)
        ]
        self.assertEqual([r.parameters for r in self.result.results], expected)
        self.assertEqual(len({r.case_id for r in self.result.results}), 6)
        self.assertTrue(all(r.ok for r in self.result.results))

    def test_identical_specifications_give_identical_ids(self):
        snapshot = Sweep.BaselineSnapshot.from_cell(_load_cell())
        twin = SweepSpec.from_dict(self.spec.to_dict())
        self.assertEqual(
            [c.case_id for c in Sweep.generate_cases(self.spec, snapshot)],
            [c.case_id for c in Sweep.generate_cases(twin, snapshot)],
        )
        self.assertEqual(
            [c.case_id for c in Sweep.generate_cases(twin, snapshot)],
            [r.case_id for r in self.result.results],
        )
        self.assertEqual(run_sweep(snapshot, twin).sweep_id, self.result.sweep_id)

    def test_case_ids_do_not_depend_on_the_rest_of_the_sweep(self):
        single = run_sweep(
            self.cell,
            SweepSpec(
                axes=(
                    SweepAxis(MASS_LOADING, (19.5,)),
                    SweepAxis(CALENDER_DENSITY, (2.6,)),
                )
            ),
        )
        self.assertEqual(single.results[0].case_id, self.result.results[5].case_id)
        self.assertNotEqual(single.sweep_id, self.result.sweep_id)
        self.assertEqual(single.results[0].metrics, self.result.results[5].metrics)

    def test_sweep_id_is_the_hash_of_the_canonical_identity(self):
        canonical = self.result.manifest["canonical"]
        text = json.dumps(
            canonical, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        self.assertEqual(
            self.result.sweep_id, hashlib.sha256(text.encode()).hexdigest()
        )
        self.assertNotIn("environment", canonical)
        self.assertEqual(canonical["parameter_order"], [MASS_LOADING, CALENDER_DENSITY])
        self.assertEqual(canonical["case_count"], 6)
        fewer_metrics = SweepSpec(axes=self.spec.axes, metrics=("energy",))
        self.assertNotEqual(
            run_sweep(self.cell, fewer_metrics).sweep_id, self.result.sweep_id
        )

    def test_manifest_records_provenance(self):
        manifest = self.result.manifest
        self.assertEqual(manifest["schema_version"], Sweep.SWEEP_SCHEMA_VERSION)
        self.assertEqual(
            manifest["canonical"]["cell_class"],
            "steer_opencell_design.Constructions.Cells.PrismaticCell.PrismaticCell",
        )
        self.assertEqual(manifest["specification"], self.spec.to_dict())
        self.assertEqual(
            manifest["canonical"]["metrics"],
            [{"name": n, "unit": Sweep.METRIC_UNITS[n]} for n in Sweep.DEFAULT_METRICS],
        )
        environment = manifest["environment"]
        self.assertEqual(environment["package_version"], ocd.__version__)
        for name in ("steer-core", "steer-materials", "numpy", "numba"):
            self.assertIsInstance(environment["dependencies"][name], str)
        self.assertIn("python", environment)
        self.assertIn("git", environment)

    def test_repeated_runs_are_identical(self):
        again = run_sweep(self.cell, self.spec)
        self.assertEqual(again.matches(self.result), [])
        for first, second in zip(self.result.results, again.results):
            self.assertEqual(first.metrics, second.metrics)  # bit-identical
            self.assertEqual(first.realized_parameters, second.realized_parameters)

    def test_matches_each_case_evaluated_by_hand(self):
        for result in self.result.results:
            expected = _evaluate_by_hand(
                self.cell, result.parameters, Sweep.DEFAULT_METRICS
            )
            for name, value in expected.items():
                with self.subTest(case=result.case_index, metric=name):
                    self.assertTrue(
                        math.isclose(
                            result.metrics[name], value, rel_tol=RTOL, abs_tol=ATOL
                        ),
                        f"{result.metrics[name]!r} != {value!r}",
                    )

    def test_changes_reach_the_cell_metrics(self):
        baseline = self.result.manifest["baseline"]["metrics"]
        low, high = self.result.results[0], self.result.results[5]
        self.assertLess(low.metrics["energy"], baseline["energy"])
        self.assertGreater(high.metrics["energy"], baseline["energy"])
        self.assertGreater(high.metrics["mass"], low.metrics["mass"])
        self.assertGreater(
            high.metrics["reversible_capacity"], low.metrics["reversible_capacity"]
        )
        # read back through mg/cm² -> kg/m² -> mg/cm²: equal within the tolerance
        realized = high.realized_parameters[MASS_LOADING]
        self.assertTrue(
            math.isclose(realized, 19.5, rel_tol=RTOL, abs_tol=ATOL), realized
        )
        self.assertEqual(high.parameter_units[MASS_LOADING], "mg/cm²")
        self.assertEqual(high.metric_units["energy"], "Wh")

    def test_every_assembly_copy_receives_the_change(self):
        snapshot = Sweep.BaselineSnapshot.from_cell(self.cell)
        cell = snapshot.new_cell()
        Sweep._apply(cell, self.spec, ((MASS_LOADING, 19.5),))
        for assembly in cell.electrode_assemblies:
            self.assertAlmostEqual(assembly.layup.cathode.mass_loading, 19.5, places=12)

    def test_cells_are_independent(self):
        snapshot = Sweep.BaselineSnapshot.from_cell(self.cell)
        first, second = snapshot.new_cell(), snapshot.new_cell()
        for path in (
            (),
            ("reference_electrode_assembly",),
            ("reference_electrode_assembly", "layup"),
            ("reference_electrode_assembly", "layup", "cathode"),
            ("reference_electrode_assembly", "layup", "cathode", "formulation"),
            ("reference_electrode_assembly", "layup", "cathode", "current_collector"),
            ("encapsulation",),
            ("electrolyte",),
        ):
            a, b, original = first, second, self.cell
            for name in path:
                a, b, original = (
                    getattr(a, name),
                    getattr(b, name),
                    getattr(original, name),
                )
            with self.subTest(path=path):
                self.assertIsNot(a, b)
                self.assertIsNot(a, original)
        Sweep._apply(first, self.spec, ((MASS_LOADING, 19.5),))
        self.assertEqual(second.serialize(compress=False), snapshot.payload)

    def test_cases_cannot_affect_one_another(self):
        reversed_spec = SweepSpec(
            axes=(
                SweepAxis(MASS_LOADING, (19.5, 17.0, 16.0)),
                SweepAxis(CALENDER_DENSITY, (2.6, 2.5)),
            )
        )
        reversed_result = run_sweep(self.cell, reversed_spec)
        by_id = {r.case_id: r for r in reversed_result.results}
        for result in self.result.results:
            self.assertEqual(by_id[result.case_id].metrics, result.metrics)

    def test_a_cell_level_parameter(self):
        result = run_sweep(
            self.cell,
            SweepSpec(
                axes=(SweepAxis("electrolyte_overfill", (10.0, 30.0)),),
                metrics=("mass", "energy"),
            ),
        )
        low, high = result.results
        self.assertTrue(low.ok and high.ok)
        self.assertEqual(low.parameter_units["electrolyte_overfill"], "%")
        self.assertGreater(high.metrics["mass"], low.metrics["mass"])
        self.assertEqual(high.metrics["energy"], low.metrics["energy"])

    def test_paths_that_do_not_apply_are_rejected_before_running(self):
        for path in (
            f"{CATHODE}.no_such_property",
            "energy",  # read-only
            "name",  # not a number
            f"{CATHODE}.serialize",  # a method
            "electrode_assemblies.layup",  # a list, not a component
            "operating_voltage_window.minimum",  # a tuple
            f"{CATHODE}.mass_loading.real",  # past a number
        ):
            with self.subTest(path=path):
                spec = SweepSpec(axes=(SweepAxis(path, (1.0,)),))
                with mock.patch.object(Sweep, "_evaluate") as evaluate:
                    with self.assertRaises(Sweep.SweepSpecificationError):
                        run_sweep(self.cell, spec)
                evaluate.assert_not_called()
        with self.assertRaises(Sweep.SweepSpecificationError):
            run_sweep(self.cell, {"axes": []})

    def test_missing_parent_references_are_detected(self):
        cell = Sweep.BaselineSnapshot.from_cell(self.cell).new_cell()
        cell.reference_electrode_assembly.layup.cathode._set_parent(None)
        with self.assertRaisesRegex(Sweep.SweepSpecificationError, "not parented"):
            Sweep._resolve(cell, MASS_LOADING)

    def test_failing_cases_do_not_halt_later_ones(self):
        spec = SweepSpec(
            axes=(SweepAxis(MASS_LOADING, (60.0, 19.5)),), metrics=("energy", "mass")
        )
        failed, passed = run_sweep(self.cell, spec).results
        self.assertEqual(failed.status, "failed")
        self.assertEqual(failed.error_category, "constraint_violation")
        self.assertIn("outside its allowed range", failed.error_message)
        self.assertEqual(failed.metrics, {})
        self.assertEqual(failed.realized_parameters, {})
        self.assertTrue(passed.ok)
        expected = _evaluate_by_hand(
            self.cell, ((MASS_LOADING, 19.5),), ("energy", "mass")
        )
        self.assertEqual(passed.metrics, expected)

    def test_a_setter_error_is_reported_per_case(self):
        wide = property(lambda self: (-1e9, 1e9))
        spec = SweepSpec(
            axes=(SweepAxis(MASS_LOADING, (-1.0, 19.5)),), metrics=("energy",)
        )
        with mock.patch.object(ocd.Cathode, "mass_loading_range", wide):
            failed, passed = run_sweep(self.cell, spec).results
        self.assertEqual(failed.error_category, "setter_error")
        self.assertIn("positive", failed.error_message)
        self.assertTrue(passed.ok)

    def test_a_propagation_error_is_reported_per_case(self):
        spec = SweepSpec(axes=(SweepAxis(MASS_LOADING, (19.5,)),), metrics=("energy",))
        with mock.patch.object(
            ocd.Cathode, "propagate_changes", side_effect=RuntimeError("boom")
        ):
            (result,) = run_sweep(self.cell, spec).results
        self.assertEqual(result.error_category, "propagation_error")
        self.assertIn("boom", result.error_message)

    def test_a_value_changed_by_a_later_axis_fails_its_case(self):
        spec = SweepSpec(
            axes=(
                SweepAxis(MASS_LOADING, (19.5,)),
                SweepAxis(COATING_THICKNESS, (60.0,)),
            ),
            metrics=("energy",),
        )
        (result,) = run_sweep(self.cell, spec).results
        self.assertEqual(result.error_category, "value_not_realized")
        self.assertIn(MASS_LOADING, result.error_message)

    def test_metric_errors_are_reported_and_never_substituted(self):
        cls = type(self.cell)
        threshold = self.baseline_mass_loading + 1.0

        def energy(cell):
            if cell.reference_electrode_assembly.layup.cathode.mass_loading > threshold:
                return float("nan")
            return cell._energy

        def broken(cell):
            if cell.reference_electrode_assembly.layup.cathode.mass_loading > threshold:
                raise RuntimeError("unreadable")
            return cell._mass

        spec = SweepSpec(
            axes=(SweepAxis(MASS_LOADING, (19.5, 16.0)),), metrics=("energy", "mass")
        )
        with mock.patch.object(cls, "energy", property(energy)):
            nan_case, fine = run_sweep(self.cell, spec).results
        self.assertEqual(nan_case.error_category, "non_finite_metric")
        self.assertEqual(nan_case.metrics, {})
        self.assertTrue(fine.ok)
        with mock.patch.object(cls, "mass", property(broken)):
            raised, fine = run_sweep(self.cell, spec).results
        self.assertEqual(raised.error_category, "metric_error")
        self.assertIn("unreadable", raised.error_message)
        self.assertTrue(fine.ok)

    def test_an_unreadable_baseline_metric_stops_the_sweep(self):
        spec = SweepSpec(axes=(SweepAxis(MASS_LOADING, (19.5,)),), metrics=("energy",))
        with mock.patch.object(
            type(self.cell), "energy", property(lambda cell: float("nan"))
        ):
            with self.assertRaises(Sweep.BaselineSnapshotError):
                run_sweep(self.cell, spec)

    def test_a_reconstruction_error_is_reported_per_case(self):
        cls = type(self.cell)
        real = cls.deserialize
        calls = []

        def flaky(payload):
            calls.append(1)
            if len(calls) == 3:  # snapshot check, probe, then the first case
                raise RuntimeError("corrupt")
            return real(payload)

        spec = SweepSpec(
            axes=(SweepAxis(MASS_LOADING, (16.0, 19.5)),), metrics=("energy",)
        )
        with mock.patch.object(cls, "deserialize", side_effect=flaky):
            first, second = run_sweep(self.cell, spec).results
        self.assertEqual(first.error_category, "reconstruction_error")
        self.assertTrue(second.ok)

    def test_unreadable_ranges_and_paths_are_reported_per_case(self):
        spec = SweepSpec(axes=(SweepAxis(MASS_LOADING, (19.5,)),), metrics=("energy",))

        def unreadable(cathode):
            raise RuntimeError("no range")

        with mock.patch.object(ocd.Cathode, "mass_loading_range", property(unreadable)):
            (result,) = run_sweep(self.cell, spec).results
        self.assertEqual(result.error_category, "constraint_error")
        self.assertIn("no range", result.error_message)

        real = Sweep._resolve
        calls = []

        def resolve_once(cell, path):
            calls.append(path)
            if len(calls) > 1:  # the probe resolves; the case cell does not
                raise Sweep.SweepSpecificationError("gone")
            return real(cell, path)

        with mock.patch.object(Sweep, "_resolve", side_effect=resolve_once):
            (result,) = run_sweep(self.cell, spec).results
        self.assertEqual(result.error_category, "resolution_error")

    def test_a_metric_the_cell_lacks_is_rejected_before_running(self):
        with mock.patch.dict(Sweep.METRIC_UNITS, {"bogus": "x"}):
            spec = SweepSpec(
                axes=(SweepAxis(MASS_LOADING, (19.5,)),), metrics=("bogus",)
            )
            with self.assertRaisesRegex(Sweep.SweepSpecificationError, "no metric"):
                run_sweep(self.cell, spec)

    def test_json_round_trip(self):
        text = self.result.to_json()
        self.assertEqual(Sweep.SweepResult.from_dict(json.loads(text)), self.result)
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "sweep.json")
            self.result.to_json(path)
            self.assertEqual(Sweep.SweepResult.from_json(path), self.result)
        self.assertEqual(
            Sweep.CaseResult.from_dict(self.result.results[0].to_dict()),
            self.result.results[0],
        )

    def test_csv_round_trip(self):
        spec = SweepSpec(
            axes=(SweepAxis(MASS_LOADING, (60.0, 19.5)),), metrics=("energy", "cost")
        )
        result = run_sweep(self.cell, spec)
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "sweep.csv")
            text = result.to_csv(path)
            rows = Sweep.SweepResult.read_csv(path)
        self.assertEqual(rows, result.to_rows())
        header = text.splitlines()[0]
        self.assertIn(f"param:{MASS_LOADING}[mg/cm²]", header)
        self.assertIn("metric:energy[Wh]", header)
        self.assertIn("metric:cost[$]", header)
        self.assertIsNone(rows[0]["metric:energy[Wh]"])
        self.assertEqual(
            rows[1]["metric:energy[Wh]"], result.results[1].metrics["energy"]
        )
        self.assertEqual(
            Sweep.SweepResult(manifest=result.manifest, results=()).to_csv(), "\n"
        )

    def test_replay_from_saved_files(self):
        snapshot = Sweep.BaselineSnapshot.from_cell(self.cell)
        with tempfile.TemporaryDirectory() as folder:
            baseline_path = os.path.join(folder, "baseline.bin")
            result_path = os.path.join(folder, "sweep.json")
            snapshot.save(baseline_path)
            self.result.to_json(result_path)

            saved = Sweep.SweepResult.from_json(result_path)
            loaded = Sweep.BaselineSnapshot.load(
                baseline_path, expected_sha256=saved.manifest["baseline"]["sha256"]
            )
            replayed = run_sweep(
                loaded, SweepSpec.from_dict(saved.manifest["specification"])
            )
            self.assertEqual(replayed.matches(saved), [])
            with self.assertRaisesRegex(Sweep.BaselineSnapshotError, "expected"):
                Sweep.BaselineSnapshot.load(baseline_path, expected_sha256="0" * 64)

    def test_matches_reports_differences(self):
        other = Sweep.SweepResult.from_dict(json.loads(self.result.to_json()))
        tweaked = other.results[0].to_dict()
        tweaked["metrics"]["energy"] *= 1 + 1e-6
        tweaked["status"] = "failed"
        tweaked["realized_parameters"] = {}
        other = Sweep.SweepResult(
            manifest=dict(other.manifest, sweep_id="x"),
            results=(Sweep.CaseResult.from_dict(tweaked),),
        )
        problems = self.result.matches(other)
        self.assertTrue(any("sweep_id" in p for p in problems))
        self.assertTrue(any("cases" in p for p in problems))
        self.assertTrue(any("energy" in p for p in problems))
        self.assertTrue(any("status" in p for p in problems))
        self.assertTrue(any("keys differ" in p for p in problems))

    def test_matches_checks_baseline_and_units(self):
        other = Sweep.SweepResult.from_dict(json.loads(self.result.to_json()))
        renamed = other.results[0].to_dict()
        renamed["metric_units"] = dict(renamed["metric_units"], energy="kWh")
        baseline = dict(other.manifest["baseline"], sha256="0" * 64)
        other = Sweep.SweepResult(
            manifest=dict(other.manifest, baseline=baseline),
            results=(Sweep.CaseResult.from_dict(renamed),) + other.results[1:],
        )
        problems = self.result.matches(other)
        self.assertTrue(any("baseline sha256" in p for p in problems))
        self.assertTrue(any("metric_units" in p for p in problems))
        self.assertEqual(len(problems), 2)

    def test_succeeded_and_failed(self):
        self.assertEqual(len(self.result.succeeded), 6)
        self.assertEqual(self.result.failed, ())


class TestBaselineSnapshot(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cell = _load_cell()

    def test_snapshot_is_canonical_and_fingerprinted(self):
        first = Sweep.BaselineSnapshot.from_cell(self.cell)
        second = Sweep.BaselineSnapshot.from_cell(self.cell)
        self.assertEqual(first.payload, second.payload)
        self.assertEqual(first.sha256, hashlib.sha256(first.payload).hexdigest())
        self.assertEqual(
            Sweep.BaselineSnapshot.from_bytes(first.payload, first.sha256), first
        )
        self.assertIs(type(first.new_cell()), ocd.PrismaticCell)

    def test_rejects_what_is_not_a_reproducible_cell(self):
        with self.assertRaises(Sweep.BaselineSnapshotError):
            Sweep.BaselineSnapshot.from_cell(self.cell.reference_electrode_assembly)
        with self.assertRaises(Sweep.BaselineSnapshotError):
            Sweep.BaselineSnapshot.from_bytes("not bytes")
        cathode = self.cell.reference_electrode_assembly.layup.cathode
        with self.assertRaises(Sweep.BaselineSnapshotError):
            Sweep.BaselineSnapshot.from_bytes(cathode.serialize(compress=False))
        # compressed bytes do not re-serialize to themselves
        with self.assertRaisesRegex(Sweep.BaselineSnapshotError, "reproducibly"):
            Sweep.BaselineSnapshot.from_bytes(self.cell.serialize(compress=True))

    def test_rejects_a_payload_of_another_class(self):
        snapshot = Sweep.BaselineSnapshot.from_cell(self.cell)
        with mock.patch.object(ocd.PrismaticCell, "deserialize", return_value=object()):
            with self.assertRaises(Sweep.BaselineSnapshotError):
                snapshot.new_cell()
            with self.assertRaises(Sweep.BaselineSnapshotError):
                Sweep.BaselineSnapshot.from_cell(self.cell)


class TestEnvironment(unittest.TestCase):
    def test_git_state_is_optional(self):
        with mock.patch.object(Sweep.subprocess, "run", side_effect=OSError("no git")):
            self.assertIsNone(Sweep._git_state())
        with mock.patch.object(Sweep.Path, "exists", return_value=False):
            self.assertIsNone(Sweep._git_state())

    def test_missing_dependencies_are_recorded_as_none(self):
        with mock.patch.object(
            Sweep.metadata, "version", side_effect=Sweep.metadata.PackageNotFoundError
        ):
            versions = Sweep._environment()["dependencies"]
        self.assertTrue(all(value is None for value in versions.values()))

    def test_open_ended_ranges(self):
        class Owner:
            value = property(lambda self: 1.0, lambda self, v: None)
            value_range = property(lambda self: (None, 2.0))

        self.assertEqual(Sweep._allowed_range(Owner(), "value"), (-math.inf, 2.0))
        self.assertIsNone(Sweep._allowed_range(Owner(), "other"))


class TestExecutionOptions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cell = _load_cell()
        cls.spec = SweepSpec(
            axes=(SweepAxis(MASS_LOADING, (16.0, 17.0)),), metrics=("energy",)
        )

    def test_invalid_worker_counts_are_rejected(self):
        too_many = Sweep._usable_cpus() + 1
        for workers in (0, -1, True, 2.5, "2", None, too_many):
            with self.subTest(workers=workers):
                with self.assertRaisesRegex(Sweep.SweepSpecificationError, "workers"):
                    run_sweep(self.cell, self.spec, workers=workers)

    def test_invalid_start_methods_are_rejected(self):
        for method in ("fork", "thread", 1):
            with self.subTest(method=method):
                with self.assertRaisesRegex(
                    Sweep.SweepSpecificationError, "start_method"
                ):
                    run_sweep(self.cell, self.spec, workers=2, start_method=method)
        with mock.patch.object(
            Sweep.multiprocessing, "get_all_start_methods", return_value=["spawn"]
        ):
            with self.assertRaisesRegex(Sweep.SweepSpecificationError, "platform"):
                run_sweep(self.cell, self.spec, workers=2, start_method="forkserver")

    def test_usable_cpus(self):
        with mock.patch.object(
            Sweep.os, "sched_getaffinity", create=True, return_value={0, 1, 2}
        ):
            self.assertEqual(Sweep._usable_cpus(), 3)
        with (
            mock.patch.object(
                Sweep.os, "sched_getaffinity", create=True, side_effect=AttributeError
            ),
            mock.patch.object(Sweep.os, "cpu_count", return_value=None),
        ):
            self.assertEqual(Sweep._usable_cpus(), 1)

    def test_one_worker_starts_no_process(self):
        with mock.patch.object(
            concurrent.futures, "ProcessPoolExecutor", side_effect=AssertionError
        ):
            result = run_sweep(self.cell, self.spec)
            single = run_sweep(
                self.cell,
                SweepSpec(axes=(SweepAxis(MASS_LOADING, (16.0,)),)),
                workers=2,
            )
        self.assertEqual(
            result.manifest["execution"],
            {"mode": "sequential", "workers": 1, "processes": 0, "start_method": None},
        )
        self.assertEqual(single.manifest["execution"]["mode"], "sequential")
        self.assertEqual(single.manifest["execution"]["workers"], 2)


class TestParallelSweep(unittest.TestCase):
    """Process-pool execution of the LFP flat-wound prismatic reference cell."""

    @classmethod
    def setUpClass(cls):
        cls.cell = _load_cell()
        cls.baseline_bytes = cls.cell.serialize(compress=False)
        cls.small = SweepSpec(
            axes=(SweepAxis(MASS_LOADING, _parity_values(8)),),
            metrics=("energy", "mass", "specific_energy"),
        )
        cls.small_reference = run_sweep(cls.cell, cls.small)

    def tearDown(self):
        self.assertEqual(multiprocessing.active_children(), [])

    def _assert_parity(self, reference, result):
        self.assertEqual(result.sweep_id, reference.sweep_id)
        self.assertEqual(
            result.manifest["baseline"]["sha256"],
            reference.manifest["baseline"]["sha256"],
        )
        self.assertEqual(
            [(r.case_index, r.case_id) for r in result.results],
            [(r.case_index, r.case_id) for r in reference.results],
        )
        for mine, theirs in zip(result.results, reference.results):
            self.assertEqual(mine.parameters, theirs.parameters)
            self.assertEqual(mine.status, theirs.status)
            self.assertEqual(mine.error_category, theirs.error_category)
            self.assertEqual(mine.error_message, theirs.error_message)
            self.assertEqual(mine.metric_units, theirs.metric_units)
            self.assertEqual(set(mine.metrics), set(theirs.metrics))
            for name, value in theirs.metrics.items():
                self.assertTrue(
                    math.isclose(mine.metrics[name], value, rel_tol=RTOL, abs_tol=ATOL),
                    f"case {mine.case_index} {name}: {mine.metrics[name]!r} != {value!r}",
                )
        self.assertEqual(result.matches(reference), [])

    def test_parity_with_sequential_execution(self):
        worker_counts = [n for n in (2, 4, 8) if n <= Sweep._usable_cpus()]
        for count in (10, 50, 100):
            spec = SweepSpec(
                axes=(SweepAxis(MASS_LOADING, _parity_values(count)),),
                metrics=Sweep.DEFAULT_METRICS,
            )
            reference = run_sweep(self.cell, spec)
            self.assertEqual(len(reference.results), count)
            self.assertEqual(len(reference.succeeded), count - 1)
            self.assertEqual(
                [r.error_category for r in reference.failed], ["constraint_violation"]
            )
            for workers in worker_counts:
                with self.subTest(cases=count, workers=workers):
                    result = run_sweep(self.cell, spec, workers=workers)
                    self.assertEqual(
                        result.manifest["execution"],
                        {
                            "mode": "process_pool",
                            "workers": workers,
                            "processes": workers,
                            "start_method": "spawn",
                        },
                    )
                    self._assert_parity(reference, result)
        self.assertEqual(self.cell.serialize(compress=False), self.baseline_bytes)

    def test_model_failures_are_isolated_in_workers(self):
        spec = SweepSpec(
            axes=(
                SweepAxis(MASS_LOADING, (16.0, 19.5)),
                SweepAxis(COATING_THICKNESS, (60.0, 0.0)),
            ),
            metrics=("energy",),
        )
        reference = run_sweep(self.cell, spec)
        self.assertEqual(len(reference.failed), 4)
        result = run_sweep(self.cell, spec, workers=2)
        self._assert_parity(reference, result)

    def test_forkserver_start_method(self):
        if "forkserver" not in multiprocessing.get_all_start_methods():
            self.skipTest("forkserver is not available on this platform")
        result = run_sweep(self.cell, self.small, workers=2, start_method="forkserver")
        self.assertEqual(result.manifest["execution"]["start_method"], "forkserver")
        self._assert_parity(self.small_reference, result)

    def test_completion_order_does_not_change_the_result(self):
        with mock.patch.object(Sweep, "_worker_run", _finish_in_reverse):
            result = run_sweep(self.cell, self.small, workers=4)
        self._assert_parity(self.small_reference, result)

    def test_replay_from_a_stored_manifest(self):
        reference = run_sweep(self.cell, self.small, workers=2)
        with tempfile.TemporaryDirectory() as folder:
            baseline_path = os.path.join(folder, "baseline.bin")
            result_path = os.path.join(folder, "sweep.json")
            Sweep.BaselineSnapshot.from_cell(self.cell).save(baseline_path)
            reference.to_json(result_path)

            saved = Sweep.SweepResult.from_json(result_path)
            baseline = Sweep.BaselineSnapshot.load(
                baseline_path, expected_sha256=saved.manifest["baseline"]["sha256"]
            )
            spec = SweepSpec.from_dict(saved.manifest["specification"])
            for workers in (1, 2):
                with self.subTest(workers=workers):
                    replayed = run_sweep(baseline, spec, workers=workers)
                    self.assertEqual(
                        replayed.manifest["canonical"], saved.manifest["canonical"]
                    )
                    self._assert_parity(saved, replayed)

    def test_a_worker_that_dies_is_an_execution_error(self):
        with mock.patch.object(Sweep, "_worker_run", _crash_on_case_3):
            with self.assertRaisesRegex(
                Sweep.SweepExecutionError, "pool broke"
            ) as caught:
                run_sweep(self.cell, self.small, workers=2)
        error = caught.exception
        self.assertIsInstance(
            error.__cause__, concurrent.futures.process.BrokenProcessPool
        )
        crashed = self.small_reference.results[3].case_id
        self.assertIn(crashed, error.unfinished_case_ids)
        self.assertIn(crashed, error.in_flight_case_ids)
        finished = [r.case_id for r in error.partial.results]
        self.assertNotIn(crashed, finished)
        self.assertEqual(
            set(error.unfinished_case_ids) | set(finished),
            {r.case_id for r in self.small_reference.results},
        )
        by_id = {r.case_id: r for r in self.small_reference.results}
        for result in error.partial.results:
            self.assertEqual(result.metrics, by_id[result.case_id].metrics)
        indices = [r.case_index for r in error.partial.results]
        self.assertEqual(indices, sorted(indices))

    def test_an_exception_escaping_a_worker_is_an_execution_error(self):
        with mock.patch.object(Sweep, "_worker_run", _raise_on_case_2):
            with self.assertRaisesRegex(
                Sweep.SweepExecutionError, r"Case 2 .* raised in its worker"
            ) as caught:
                run_sweep(self.cell, self.small, workers=2)
        error = caught.exception
        self.assertIsInstance(error.__cause__, ValueError)
        self.assertIn("injected worker fault", str(error))
        case_2 = self.small_reference.results[2].case_id
        self.assertIn(case_2, error.in_flight_case_ids)
        self.assertIn(case_2, error.unfinished_case_ids)
        self.assertNotIn(case_2, [r.case_id for r in error.partial.results])

    def test_a_worker_answering_for_another_case_is_an_execution_error(self):
        with mock.patch.object(Sweep, "_worker_run", _answer_for_another_case):
            with self.assertRaisesRegex(Sweep.SweepExecutionError, "returned case"):
                run_sweep(self.cell, self.small, workers=2)

    def test_a_pool_that_cannot_start_is_an_execution_error(self):
        with mock.patch.object(Sweep, "_worker_init", _failing_init):
            with self.assertRaisesRegex(
                Sweep.SweepExecutionError, "pool broke"
            ) as caught:
                run_sweep(self.cell, self.small, workers=2)
        error = caught.exception
        self.assertEqual(error.partial.results, ())
        self.assertEqual(
            error.unfinished_case_ids,
            tuple(r.case_id for r in self.small_reference.results),
        )

    def test_a_pool_that_rejects_cases_is_an_execution_error(self):
        pool = mock.MagicMock()
        pool.submit.side_effect = RuntimeError("cannot schedule new futures")
        with mock.patch.object(
            concurrent.futures, "ProcessPoolExecutor", return_value=pool
        ):
            with self.assertRaisesRegex(
                Sweep.SweepExecutionError, "cannot accept cases"
            ):
                run_sweep(self.cell, self.small, workers=2)
        pool.shutdown.assert_called_once_with(wait=True, cancel_futures=True)

    def test_workers_and_baseline_file_are_cleaned_up(self):
        folders = []
        real = tempfile.TemporaryDirectory

        def spy(*args, **kwargs):
            folder = real(*args, **kwargs)
            folders.append(folder.name)
            return folder

        with mock.patch.object(Sweep.tempfile, "TemporaryDirectory", spy):
            run_sweep(self.cell, self.small, workers=2)
            with mock.patch.object(
                concurrent.futures, "wait", side_effect=KeyboardInterrupt
            ):
                with self.assertRaises(KeyboardInterrupt):
                    run_sweep(self.cell, self.small, workers=2)
        self.assertEqual(len(folders), 2)
        for folder in folders:
            self.assertFalse(os.path.exists(folder))

    def test_baseline_is_unchanged(self):
        run_sweep(self.cell, self.small, workers=2)
        self.assertEqual(self.cell.serialize(compress=False), self.baseline_bytes)


class TestWorkerFunctions(unittest.TestCase):
    """The worker entry points, run in this process."""

    @classmethod
    def setUpClass(cls):
        cls.cell = _load_cell()
        cls.snapshot = Sweep.BaselineSnapshot.from_cell(cls.cell)
        cls.spec = SweepSpec(
            axes=(SweepAxis(MASS_LOADING, (16.0,)),), metrics=("mass",)
        )

    def tearDown(self):
        Sweep._WORKER_STATE = None

    def test_worker_evaluates_like_the_sequential_runner(self):
        (case,) = Sweep.generate_cases(self.spec, self.snapshot)
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "baseline.bin")
            self.snapshot.save(path)
            with self.assertRaisesRegex(Sweep.BaselineSnapshotError, "expected"):
                Sweep._worker_init(path, "0" * 64, self.spec, "sweep")
            Sweep._worker_init(path, self.snapshot.sha256, self.spec, "sweep")
        result = Sweep._worker_run(case.index, case.case_id, case.parameters)
        expected = Sweep._evaluate(self.snapshot, self.spec, case, "sweep")
        self.assertEqual(result.metrics, expected.metrics)
        self.assertEqual(result.case_id, case.case_id)
        with self.assertRaisesRegex(RuntimeError, "different case ID"):
            Sweep._worker_run(case.index, "0" * 16, case.parameters)

    def test_an_uninitialized_worker_refuses_cases(self):
        Sweep._WORKER_STATE = None
        with self.assertRaisesRegex(RuntimeError, "not initialized"):
            Sweep._worker_run(0, "0" * 16, ((MASS_LOADING, 16.0),))


class TestSequentialExecutionErrors(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cell = _load_cell()
        cls.spec = SweepSpec(
            axes=(SweepAxis(MASS_LOADING, (16.0, 17.0, 18.0)),), metrics=("energy",)
        )

    def test_an_exception_escaping_a_case_is_an_execution_error(self):
        real = Sweep._evaluate
        calls = []

        def flaky(*args):
            calls.append(args)
            if len(calls) == 2:
                raise MemoryError("injected")
            return real(*args)

        with mock.patch.object(Sweep, "_evaluate", flaky):
            with self.assertRaisesRegex(
                Sweep.SweepExecutionError, r"Case 1 .* outside case evaluation"
            ) as caught:
                run_sweep(self.cell, self.spec)
        error = caught.exception
        self.assertIsInstance(error.__cause__, MemoryError)
        self.assertEqual([r.case_index for r in error.partial.results], [0])
        self.assertEqual(len(error.unfinished_case_ids), 2)
        self.assertEqual(error.in_flight_case_ids, error.unfinished_case_ids[:1])

    def test_a_parameter_that_cannot_be_read_back_fails_its_case(self):
        real = Sweep._resolve
        calls = []

        def resolve(cell, path):
            calls.append(path)
            if len(calls) == 3:  # the probe, the set, then the read-back
                raise RuntimeError("detached")
            return real(cell, path)

        spec = SweepSpec(axes=(SweepAxis(MASS_LOADING, (16.0,)),), metrics=("energy",))
        with mock.patch.object(Sweep, "_resolve", resolve):
            (result,) = run_sweep(self.cell, spec).results
        self.assertEqual(result.error_category, "resolution_error")
        self.assertIn("cannot be read back", result.error_message)
        self.assertEqual(result.metrics, {})


class TestErrorKinds(unittest.TestCase):
    def test_every_category_has_a_kind(self):
        categories = {
            value
            for name, value in vars(Sweep).items()
            if name.isupper()
            and name.endswith(("_ERROR", "_VIOLATION", "_METRIC", "_REALIZED"))
        }
        self.assertEqual(categories, set(Sweep.ERROR_KINDS))
        self.assertEqual(
            set(Sweep.ERROR_KINDS.values()),
            {"invalid_parameters", "model_evaluation", "serialization"},
        )

    def test_error_kind_of_a_result(self):
        fields = dict(
            sweep_id="s",
            case_id="c",
            case_index=0,
            parameters=(),
            parameter_units={},
            realized_parameters={},
            metrics={},
            metric_units={},
            error_message=None,
            elapsed_s=0.0,
        )
        kinds = {
            "setter_error": "model_evaluation",
            "constraint_violation": "invalid_parameters",
            "reconstruction_error": "serialization",
        }
        for category, kind in kinds.items():
            result = Sweep.CaseResult(
                status="failed", error_category=category, **fields
            )
            self.assertEqual(result.error_kind, kind)
        ok = Sweep.CaseResult(status="ok", error_category=None, **fields)
        self.assertIsNone(ok.error_kind)


if __name__ == "__main__":
    unittest.main()
