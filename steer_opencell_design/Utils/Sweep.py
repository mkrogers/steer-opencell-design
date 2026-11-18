# SPDX-FileCopyrightText: 2024-2026 Stanford University
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Reproducible, isolated parameter sweeps over a cell design.

A sweep evaluates a cell at every combination (full factorial) of a few
parameter values and reads a fixed set of cell-level metrics from each::

    from steer_opencell_design.Utils.Sweep import SweepAxis, SweepSpec, run_sweep

    spec = SweepSpec(
        axes=(
            SweepAxis(
                "reference_electrode_assembly.layup.cathode.mass_loading",
                (16.0, 17.0, 18.0),
            ),
        ),
        metrics=("energy", "mass", "specific_energy"),
    )
    result = run_sweep(cell, spec)

How a case is evaluated
-----------------------
The baseline cell is serialized once with the package's own
``serialize(compress=False)``; those bytes are the *baseline snapshot*. The
snapshot is accepted only if deserializing and re-serializing it reproduces the
same bytes, and its SHA-256 identifies the baseline. Every case deserializes a
new cell from the snapshot with the concrete cell class, so no case shares an
object with the baseline or with another case, and the supplied cell is only
ever serialized, never changed.

Each parameter is applied through the public property setter of the object
that owns it, in the order the axes were declared, and then propagated to the
cell with that object's ``propagate_changes()`` (a cell-level property
recalculates the cell in its own setter). Before a value is set it is checked
against the owner's ``<name>_range`` when the owner defines one, using the
range of the case's cell at that moment. Propagation re-assigns each object to
its parent through the parent's setter, so the resolver checks that every
object on the path is parented to the one before it under that path segment;
a path whose parent references are missing is rejected rather than evaluated
with stale cell metrics.

Parameter paths
---------------
A path is a dotted chain of public property names starting at the cell, such
as ``reference_electrode_assembly.layup.cathode.mass_loading``. Each segment
must be a lowercase identifier that does not start with an underscore, every
intermediate segment must be a property holding a model component, and the
last must be a property with a setter whose current value is a real number.
Paths are resolved with ``inspect.getattr_static`` and ``getattr`` only:
nothing is evaluated, no list is indexed, and no private attribute is reached.

Canonicalization and floating point
-----------------------------------
Values are kept in the order given. Integers (including NumPy integers) stay
integers; other reals become Python ``float``; ``-0.0`` becomes ``0.0``;
booleans, strings, NaN and infinities are rejected, as are two values of one
axis that compare equal. Canonical JSON is ``json.dumps`` with sorted keys,
compact separators and ``allow_nan=False``, so a float is written as its
shortest round-tripping ``repr``, which is the same on every platform.

- ``sweep_id`` is the SHA-256 of the canonical JSON of the sweep's identity:
  schema version, cell class, baseline SHA-256, the axes in declared order
  (path, unit, values) and the metrics (name, unit).
- A ``case_id`` is the first 16 hex characters of the SHA-256 of the canonical
  JSON of the schema version, cell class, baseline SHA-256 and the case's
  ``[path, value]`` pairs in declared order. It does not depend on the
  execution order or on the other cases, so the same case has the same ID in
  any sweep over the same baseline.
- Cases are enumerated with ``itertools.product`` over the axes in declared
  order: the first axis varies slowest and the last fastest.

Machine-specific metadata (versions, platform, git revision, timings) is kept
out of both fingerprints. Metrics are compared across runs with a relative
tolerance of ``1e-9`` and an absolute tolerance of ``1e-12``; on one machine
and environment repeated runs are bit-identical.

Limitations
-----------
- Sweeps run sequentially in the calling process.
- The container is never resized: a parameter that grows an electrode assembly
  can leave it larger than the container, and the model reports no error.
- A ``<name>_range`` is checked only where the owner defines one; a later axis
  that changes an earlier axis's value fails that case (``value_not_realized``).
"""

from __future__ import annotations

import csv
import hashlib
import inspect
import io
import itertools
import json
import math
import numbers
import os
import platform
import re
import subprocess
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from steer_core.Mixins.Propagation import PropagationMixin

__all__ = [
    "BaselineSnapshot",
    "BaselineSnapshotError",
    "CaseResult",
    "DEFAULT_MAX_CASES",
    "DEFAULT_METRICS",
    "METRIC_UNITS",
    "PARAMETER_UNITS",
    "SWEEP_SCHEMA_VERSION",
    "SweepAxis",
    "SweepCase",
    "SweepResult",
    "SweepSpec",
    "SweepSpecificationError",
    "generate_cases",
    "run_sweep",
]

Number = Union[int, float]

SWEEP_SCHEMA_VERSION = 1
"""Version of the sweep specification, manifest and result layout."""

DEFAULT_MAX_CASES = 1000
"""Largest case count a specification accepts unless ``max_cases`` is raised."""

CASE_ID_LENGTH = 16
RTOL = 1e-9
ATOL = 1e-12

STATUS_OK = "ok"
STATUS_FAILED = "failed"

# Error categories of a failed case.
RECONSTRUCTION_ERROR = "reconstruction_error"
RESOLUTION_ERROR = "resolution_error"
CONSTRAINT_VIOLATION = "constraint_violation"
CONSTRAINT_ERROR = "constraint_error"
SETTER_ERROR = "setter_error"
PROPAGATION_ERROR = "propagation_error"
VALUE_NOT_REALIZED = "value_not_realized"
METRIC_ERROR = "metric_error"
NON_FINITE_METRIC = "non_finite_metric"

METRIC_UNITS: Dict[str, str] = {
    "reversible_capacity": "Ah",
    "irreversible_capacity": "Ah",
    "energy": "Wh",
    "mass": "g",
    "cost": "$",
    "volume": "L",
    "specific_energy": "Wh/kg",
    "volumetric_energy": "Wh/L",
    "cost_per_energy": "$/kWh",
    "minimum_operating_voltage": "V",
    "maximum_operating_voltage": "V",
}
"""Cell properties a sweep may read, with the units they return."""

DEFAULT_METRICS: Tuple[str, ...] = tuple(METRIC_UNITS)

PARAMETER_UNITS: Dict[str, str] = {
    "mass_loading": "mg/cm²",
    "calender_density": "g/cm³",
    "coating_thickness": "µm",
    "porosity": "%",
    "electrolyte_overfill": "%",
}
"""Documented units of common parameters, keyed by the last path segment.

An axis whose last segment is not listed has no unit unless one is given.
"""

_SEGMENT = re.compile(r"[a-z][a-z0-9_]*\Z")
_INTEGER_TEXT = re.compile(r"-?\d+\Z")

_DEPENDENCIES = (
    "steer-opencell-design",
    "steer-core",
    "steer-materials",
    "steer-opencell-data",
    "numpy",
    "numba",
    "scipy",
    "pandas",
    "msgpack",
    "msgpack-numpy",
    "lz4",
)


class SweepSpecificationError(ValueError):
    """A sweep specification is invalid, or does not apply to the baseline cell."""


class BaselineSnapshotError(ValueError):
    """A baseline cell cannot be snapshotted reproducibly."""


# ---------------------------------------------------------------------------
# Canonical forms
# ---------------------------------------------------------------------------


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical_number(value: Any, where: str) -> Number:
    """Return ``value`` as a finite ``int`` or ``float``, or raise."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise SweepSpecificationError(
            f"{where}: expected a finite real number, got {type(value).__name__} {value!r}."
        )
    if isinstance(value, numbers.Integral):
        return int(value)
    number = float(value)
    if not math.isfinite(number):
        raise SweepSpecificationError(
            f"{where}: expected a finite number, got {number!r}."
        )
    return number + 0.0  # -0.0 -> 0.0


def _is_real(value: Any) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def _parse_path(path: Any) -> Tuple[str, ...]:
    if not isinstance(path, str) or not path:
        raise SweepSpecificationError(
            f"A parameter path must be a non-empty string, got {path!r}."
        )
    segments = tuple(path.split("."))
    for segment in segments:
        if not _SEGMENT.match(segment):
            raise SweepSpecificationError(
                f"Parameter path {path!r}: segment {segment!r} is not a public property "
                "name (lowercase letters, digits and underscores, not starting with '_')."
            )
    return segments


# ---------------------------------------------------------------------------
# Specification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SweepAxis:
    """One swept parameter: a property path and the values it takes, in order.

    Parameters
    ----------
    path : str
        Dotted public property path from the cell, e.g.
        ``"reference_electrode_assembly.layup.cathode.mass_loading"``.
    values : sequence of real numbers
        The values, in the order they are enumerated; finite, distinct and at
        least one. They are in the property's public units.
    unit : str, optional
        Unit of the values, recorded with the results. Defaults to the entry of
        ``PARAMETER_UNITS`` for the path's last segment, if any.
    """

    path: str
    values: Tuple[Number, ...]
    unit: Optional[str] = None

    def __post_init__(self) -> None:
        segments = _parse_path(self.path)
        if isinstance(self.values, (str, bytes)) or not isinstance(
            self.values, Iterable
        ):
            raise SweepSpecificationError(
                f"Axis {self.path!r}: values must be a sequence of numbers, got {self.values!r}."
            )
        values = tuple(
            _canonical_number(value, f"Axis {self.path!r} value {index}")
            for index, value in enumerate(self.values)
        )
        if not values:
            raise SweepSpecificationError(f"Axis {self.path!r} has no values.")
        seen: Dict[float, Number] = {}
        for value in values:
            if float(value) in seen:
                raise SweepSpecificationError(
                    f"Axis {self.path!r} lists the value {value!r} more than once."
                )
            seen[float(value)] = value
        unit = self.unit if self.unit is not None else PARAMETER_UNITS.get(segments[-1])
        if unit is not None and not isinstance(unit, str):
            raise SweepSpecificationError(
                f"Axis {self.path!r}: unit must be a string or None."
            )
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "unit", unit)

    @property
    def segments(self) -> Tuple[str, ...]:
        """The path split into property names."""
        return tuple(self.path.split("."))

    def to_dict(self) -> Dict[str, Any]:
        """JSON-compatible form, read back by :meth:`from_dict`."""
        return {"path": self.path, "values": list(self.values), "unit": self.unit}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SweepAxis":
        """Build an axis from :meth:`to_dict` output."""
        if not isinstance(data, Mapping) or set(data) - {"path", "values", "unit"}:
            raise SweepSpecificationError(f"Not a sweep axis: {data!r}.")
        return cls(
            path=data.get("path"), values=data.get("values"), unit=data.get("unit")
        )


@dataclass(frozen=True)
class SweepSpec:
    """What a sweep varies and what it measures.

    Parameters
    ----------
    axes : sequence of SweepAxis
        The swept parameters, applied to each case in this order. Paths must be
        distinct.
    metrics : sequence of str
        Cell metrics to read, from ``METRIC_UNITS``. Defaults to all of them.
    max_cases : int
        Upper bound on the number of cases (the product of the axis lengths).
        Raise it explicitly to run a larger sweep.
    schema_version : int
        Sweep schema version; only ``SWEEP_SCHEMA_VERSION`` is accepted.
    """

    axes: Tuple[SweepAxis, ...]
    metrics: Tuple[str, ...] = DEFAULT_METRICS
    max_cases: int = DEFAULT_MAX_CASES
    schema_version: int = SWEEP_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SWEEP_SCHEMA_VERSION or isinstance(
            self.schema_version, bool
        ):
            raise SweepSpecificationError(
                f"Unsupported sweep schema version {self.schema_version!r}; "
                f"this version reads {SWEEP_SCHEMA_VERSION}."
            )
        if isinstance(self.axes, (str, bytes)) or not isinstance(self.axes, Iterable):
            raise SweepSpecificationError("axes must be a sequence of SweepAxis.")
        axes = tuple(self.axes)
        if not axes:
            raise SweepSpecificationError("A sweep needs at least one axis.")
        paths = set()
        for axis in axes:
            if not isinstance(axis, SweepAxis):
                raise SweepSpecificationError(f"Not a SweepAxis: {axis!r}.")
            if axis.path in paths:
                raise SweepSpecificationError(
                    f"The parameter {axis.path!r} is swept twice."
                )
            paths.add(axis.path)

        if isinstance(self.metrics, (str, bytes)) or not isinstance(
            self.metrics, Iterable
        ):
            raise SweepSpecificationError("metrics must be a sequence of metric names.")
        metrics = tuple(self.metrics)
        if not metrics:
            raise SweepSpecificationError("A sweep needs at least one metric.")
        for name in metrics:
            if not isinstance(name, str) or name not in METRIC_UNITS:
                raise SweepSpecificationError(
                    f"Unknown metric {name!r}; supported metrics are {sorted(METRIC_UNITS)}."
                )
        if len(set(metrics)) != len(metrics):
            raise SweepSpecificationError("A metric is requested more than once.")

        if (
            isinstance(self.max_cases, bool)
            or not isinstance(self.max_cases, int)
            or self.max_cases < 1
        ):
            raise SweepSpecificationError(
                f"max_cases must be a positive integer, got {self.max_cases!r}."
            )
        count = math.prod(len(axis.values) for axis in axes)
        if count > self.max_cases:
            raise SweepSpecificationError(
                f"The sweep has {count} cases, more than max_cases={self.max_cases}. "
                "Pass a larger max_cases to run it."
            )
        object.__setattr__(self, "axes", axes)
        object.__setattr__(self, "metrics", metrics)

    @property
    def case_count(self) -> int:
        """Number of cases: the product of the axis lengths."""
        return math.prod(len(axis.values) for axis in self.axes)

    @property
    def parameter_order(self) -> Tuple[str, ...]:
        """Axis paths in the order they are enumerated and applied."""
        return tuple(axis.path for axis in self.axes)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-compatible form, read back by :meth:`from_dict`."""
        return {
            "schema_version": self.schema_version,
            "axes": [axis.to_dict() for axis in self.axes],
            "metrics": list(self.metrics),
            "max_cases": self.max_cases,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SweepSpec":
        """Build a specification from :meth:`to_dict` output."""
        allowed = {"schema_version", "axes", "metrics", "max_cases"}
        if not isinstance(data, Mapping) or set(data) - allowed or "axes" not in data:
            raise SweepSpecificationError(f"Not a sweep specification: {data!r}.")
        axes = data["axes"]
        if not isinstance(axes, list):
            raise SweepSpecificationError("axes must be a list.")
        return cls(
            axes=tuple(SweepAxis.from_dict(axis) for axis in axes),
            metrics=tuple(data.get("metrics", DEFAULT_METRICS)),
            max_cases=data.get("max_cases", DEFAULT_MAX_CASES),
            schema_version=data.get("schema_version", SWEEP_SCHEMA_VERSION),
        )


# ---------------------------------------------------------------------------
# Baseline snapshot
# ---------------------------------------------------------------------------


def _cell_base() -> type:
    from steer_opencell_design.Constructions.Cells.Base import _Cell

    return _Cell


def _class_path(cls: type) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


@dataclass(frozen=True)
class BaselineSnapshot:
    """The serialized baseline cell every case is rebuilt from.

    Build one with :meth:`from_cell` (or :meth:`load` / :meth:`from_bytes`);
    construction checks that the bytes are canonical, i.e. that deserializing
    and re-serializing them reproduces them exactly.
    """

    payload: bytes = field(repr=False)
    sha256: str
    cell_class: str
    _cls: type = field(repr=False, compare=False)

    @classmethod
    def from_cell(cls, cell: Any) -> "BaselineSnapshot":
        """Snapshot ``cell`` without modifying it (it is only serialized)."""
        if not isinstance(cell, _cell_base()):
            raise BaselineSnapshotError(
                f"A sweep baseline must be a cell, got {type(cell).__name__}."
            )
        return cls._verified(cell.serialize(compress=False), type(cell))

    @classmethod
    def from_bytes(
        cls, payload: bytes, expected_sha256: Optional[str] = None
    ) -> "BaselineSnapshot":
        """Snapshot from bytes written by :meth:`save`, optionally checking their SHA-256."""
        if not isinstance(payload, (bytes, bytearray)):
            raise BaselineSnapshotError("A baseline snapshot payload must be bytes.")
        payload = bytes(payload)
        digest = hashlib.sha256(payload).hexdigest()
        if expected_sha256 is not None and digest != expected_sha256:
            raise BaselineSnapshotError(
                f"Baseline SHA-256 is {digest}, expected {expected_sha256}."
            )
        from steer_core.Mixins.Serializer import SerializerMixin

        cell = SerializerMixin.deserialize(payload)
        if not isinstance(cell, _cell_base()):
            raise BaselineSnapshotError(
                f"The payload holds a {type(cell).__name__}, not a cell."
            )
        return cls._verified(payload, type(cell))

    @classmethod
    def load(
        cls, path: Union[str, os.PathLike], expected_sha256: Optional[str] = None
    ) -> "BaselineSnapshot":
        """Read a snapshot written by :meth:`save`."""
        return cls.from_bytes(Path(path).read_bytes(), expected_sha256=expected_sha256)

    def save(self, path: Union[str, os.PathLike]) -> None:
        """Write the snapshot bytes to ``path``."""
        Path(path).write_bytes(self.payload)

    @classmethod
    def _verified(cls, payload: bytes, cell_cls: type) -> "BaselineSnapshot":
        rebuilt = cell_cls.deserialize(payload)
        if type(rebuilt) is not cell_cls:
            raise BaselineSnapshotError(
                f"Deserializing the baseline gave a {type(rebuilt).__name__}, "
                f"not a {cell_cls.__name__}."
            )
        if rebuilt.serialize(compress=False) != payload:
            raise BaselineSnapshotError(
                "The baseline does not serialize reproducibly: deserializing and "
                "re-serializing it changes its bytes, so its cases would not start "
                "from the baseline."
            )
        return cls(
            payload=payload,
            sha256=hashlib.sha256(payload).hexdigest(),
            cell_class=_class_path(cell_cls),
            _cls=cell_cls,
        )

    def new_cell(self) -> Any:
        """A new, independent cell deserialized from the snapshot."""
        cell = self._cls.deserialize(self.payload)
        if type(cell) is not self._cls:
            raise BaselineSnapshotError(
                f"Deserializing the baseline gave a {type(cell).__name__}, "
                f"not a {self._cls.__name__}."
            )
        return cell


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SweepCase:
    """One combination of parameter values."""

    index: int
    case_id: str
    parameters: Tuple[Tuple[str, Number], ...]


def _case_id(
    snapshot: BaselineSnapshot, parameters: Sequence[Tuple[str, Number]]
) -> str:
    identity = {
        "schema_version": SWEEP_SCHEMA_VERSION,
        "cell_class": snapshot.cell_class,
        "baseline_sha256": snapshot.sha256,
        "parameters": [[path, value] for path, value in parameters],
    }
    return _sha256(_canonical_json(identity))[:CASE_ID_LENGTH]


def generate_cases(spec: SweepSpec, snapshot: BaselineSnapshot) -> List[SweepCase]:
    """Enumerate the full-factorial cases of ``spec`` (first axis slowest)."""
    paths = spec.parameter_order
    cases = []
    for index, values in enumerate(
        itertools.product(*(axis.values for axis in spec.axes))
    ):
        parameters = tuple(zip(paths, values))
        cases.append(
            SweepCase(
                index=index,
                case_id=_case_id(snapshot, parameters),
                parameters=parameters,
            )
        )
    return cases


# ---------------------------------------------------------------------------
# Applying parameters
# ---------------------------------------------------------------------------


def _resolve(cell: Any, path: str) -> Tuple[Any, str]:
    """Return ``(owner, name)`` for a settable numeric property path on ``cell``.

    Raises SweepSpecificationError when the path does not name one, or when an
    object on it is not parented to the one before it (so that a change would
    not propagate to the cell).
    """
    segments = _parse_path(path)
    owner = cell
    for depth, name in enumerate(segments):
        attribute = inspect.getattr_static(type(owner), name, None)
        if not isinstance(attribute, property):
            raise SweepSpecificationError(
                f"Parameter path {path!r}: {type(owner).__name__} has no public property {name!r}."
            )
        if depth == len(segments) - 1:
            if attribute.fset is None:
                raise SweepSpecificationError(
                    f"Parameter path {path!r}: {type(owner).__name__}.{name} is read-only."
                )
            current = getattr(owner, name)
            if not _is_real(current):
                raise SweepSpecificationError(
                    f"Parameter path {path!r}: {type(owner).__name__}.{name} is "
                    f"{type(current).__name__}, not a number."
                )
            return owner, name
        child = getattr(owner, name)
        if not isinstance(child, PropagationMixin):
            raise SweepSpecificationError(
                f"Parameter path {path!r}: {type(owner).__name__}.{name} is "
                f"{type(child).__name__}, not a model component."
            )
        if (
            child._get_parent() is not owner
            or getattr(child, "_parent_attr_name", None) != name
        ):
            raise SweepSpecificationError(
                f"Parameter path {path!r}: {type(child).__name__} is not parented to "
                f"{type(owner).__name__} as {name!r}, so a change would not reach the cell."
            )
        owner = child
    raise AssertionError("unreachable")  # pragma: no cover


def _allowed_range(owner: Any, name: str) -> Optional[Tuple[float, float]]:
    attribute = inspect.getattr_static(type(owner), f"{name}_range", None)
    if not isinstance(attribute, property):
        return None
    bounds = getattr(owner, f"{name}_range")
    low, high = bounds
    return (
        float(low) if low is not None else -math.inf,
        float(high) if high is not None else math.inf,
    )


class _CaseFailure(Exception):
    def __init__(self, category: str, message: str) -> None:
        super().__init__(message)
        self.category = category
        self.message = message


def _describe(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _read_metrics(cell: Any, metrics: Sequence[str]) -> Dict[str, float]:
    values = {}
    for name in metrics:
        try:
            value = float(getattr(cell, name))
        except Exception as exc:
            raise _CaseFailure(METRIC_ERROR, f"{name}: {_describe(exc)}") from exc
        if not math.isfinite(value):
            raise _CaseFailure(NON_FINITE_METRIC, f"{name} is {value!r}.")
        values[name] = value
    return values


def _apply(
    cell: Any, spec: SweepSpec, parameters: Sequence[Tuple[str, Number]]
) -> Dict[str, float]:
    units = {axis.path: axis.unit for axis in spec.axes}
    for path, value in parameters:
        try:
            owner, name = _resolve(cell, path)
        except SweepSpecificationError as exc:
            raise _CaseFailure(RESOLUTION_ERROR, str(exc)) from exc
        try:
            bounds = _allowed_range(owner, name)
        except Exception as exc:
            raise _CaseFailure(
                CONSTRAINT_ERROR, f"{path}_range: {_describe(exc)}"
            ) from exc
        if bounds is not None and not bounds[0] <= value <= bounds[1]:
            unit = f" {units[path]}" if units[path] else ""
            raise _CaseFailure(
                CONSTRAINT_VIOLATION,
                f"{path} = {value!r} is outside its allowed range "
                f"[{bounds[0]!r}, {bounds[1]!r}]{unit}.",
            )
        try:
            setattr(owner, name, value)
        except Exception as exc:
            raise _CaseFailure(
                SETTER_ERROR, f"{path} = {value!r}: {_describe(exc)}"
            ) from exc
        if owner is not cell:
            try:
                owner.propagate_changes()
            except Exception as exc:
                raise _CaseFailure(
                    PROPAGATION_ERROR, f"{path}: {_describe(exc)}"
                ) from exc

    realized = {}
    for path, value in parameters:
        owner, name = _resolve(cell, path)
        actual = float(getattr(owner, name))
        if not math.isclose(actual, value, rel_tol=RTOL, abs_tol=ATOL):
            raise _CaseFailure(
                VALUE_NOT_REALIZED,
                f"{path} was set to {value!r} but is {actual!r} after all parameters were applied.",
            )
        realized[path] = actual
    return realized


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CaseResult:
    """The outcome of one case.

    ``metrics`` and ``realized_parameters`` are empty when ``status`` is
    ``"failed"``; ``error_category`` and ``error_message`` then say why.
    ``elapsed_s`` is wall time and is not part of any comparison.
    """

    sweep_id: str
    case_id: str
    case_index: int
    parameters: Tuple[Tuple[str, Number], ...]
    parameter_units: Dict[str, Optional[str]]
    realized_parameters: Dict[str, float]
    metrics: Dict[str, float]
    metric_units: Dict[str, str]
    status: str
    error_category: Optional[str]
    error_message: Optional[str]
    elapsed_s: float

    @property
    def ok(self) -> bool:
        """Whether the case was evaluated."""
        return self.status == STATUS_OK

    def to_dict(self) -> Dict[str, Any]:
        """JSON-compatible form, read back by :meth:`from_dict`."""
        return {
            "sweep_id": self.sweep_id,
            "case_id": self.case_id,
            "case_index": self.case_index,
            "parameters": [[path, value] for path, value in self.parameters],
            "parameter_units": dict(self.parameter_units),
            "realized_parameters": dict(self.realized_parameters),
            "metrics": dict(self.metrics),
            "metric_units": dict(self.metric_units),
            "status": self.status,
            "error_category": self.error_category,
            "error_message": self.error_message,
            "elapsed_s": self.elapsed_s,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CaseResult":
        """Build a result from :meth:`to_dict` output."""
        values = dict(data)
        values["parameters"] = tuple(
            (path, value) for path, value in values["parameters"]
        )
        return cls(**values)


def _environment() -> Dict[str, Any]:
    import steer_opencell_design

    versions = {}
    for name in _DEPENDENCIES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "package_version": steer_opencell_design.__version__,
        "dependencies": versions,
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "git": _git_state(),
        "opencell_env": os.environ.get("OPENCELL_ENV", "development"),
    }


def _git_state() -> Optional[Dict[str, Any]]:
    """The revision of the source checkout this package runs from, if any."""
    root = Path(__file__).resolve().parents[2]
    if not (root / ".git").exists():
        return None
    try:
        revision = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return {"revision": revision, "dirty": bool(status.strip())}


def _build_manifest(
    spec: SweepSpec, snapshot: BaselineSnapshot, baseline_metrics: Dict[str, float]
) -> Dict[str, Any]:
    canonical = {
        "schema_version": spec.schema_version,
        "cell_class": snapshot.cell_class,
        "baseline_sha256": snapshot.sha256,
        "parameter_order": list(spec.parameter_order),
        "parameters": [axis.to_dict() for axis in spec.axes],
        "metrics": [
            {"name": name, "unit": METRIC_UNITS[name]} for name in spec.metrics
        ],
        "case_count": spec.case_count,
        "enumeration": "full-factorial, first axis slowest",
    }
    return {
        "schema_version": spec.schema_version,
        "sweep_id": _sha256(_canonical_json(canonical)),
        "canonical": canonical,
        "specification": spec.to_dict(),
        "baseline": {
            "sha256": snapshot.sha256,
            "cell_class": snapshot.cell_class,
            "payload_bytes": len(snapshot.payload),
            "metrics": baseline_metrics,
        },
        "environment": _environment(),
    }


def _csv_header(name: str, unit: Optional[str]) -> str:
    return f"{name}[{unit}]" if unit else name


def _csv_value(text: str) -> Any:
    if text == "":
        return None
    if _INTEGER_TEXT.match(text):
        return int(text)
    return float(text)


@dataclass(frozen=True)
class SweepResult:
    """A sweep's manifest and its case results, in case-index order."""

    manifest: Dict[str, Any]
    results: Tuple[CaseResult, ...]

    @property
    def sweep_id(self) -> str:
        """SHA-256 of the sweep's canonical identity (see the module docs)."""
        return self.manifest["sweep_id"]

    @property
    def succeeded(self) -> Tuple[CaseResult, ...]:
        """Results whose case was evaluated."""
        return tuple(result for result in self.results if result.ok)

    @property
    def failed(self) -> Tuple[CaseResult, ...]:
        """Results whose case failed."""
        return tuple(result for result in self.results if not result.ok)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-compatible form, read back by :meth:`from_dict`."""
        return {
            "manifest": self.manifest,
            "results": [result.to_dict() for result in self.results],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SweepResult":
        """Build a result from :meth:`to_dict` output."""
        return cls(
            manifest=dict(data["manifest"]),
            results=tuple(CaseResult.from_dict(result) for result in data["results"]),
        )

    def to_json(self, path: Union[str, os.PathLike, None] = None) -> str:
        """Return the result as JSON, and write it to ``path`` if given."""
        text = json.dumps(self.to_dict(), sort_keys=True, indent=2, allow_nan=False)
        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
        return text

    @classmethod
    def from_json(cls, path: Union[str, os.PathLike]) -> "SweepResult":
        """Read a result from a file written by :meth:`to_json`."""
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def to_rows(self) -> List[Dict[str, Any]]:
        """One flat dict per case, keyed by the CSV column names."""
        canonical = self.manifest["canonical"]
        rows = []
        for result in self.results:
            row: Dict[str, Any] = {
                "sweep_id": result.sweep_id,
                "case_index": result.case_index,
                "case_id": result.case_id,
                "status": result.status,
                "error_category": result.error_category,
                "error_message": result.error_message,
                "elapsed_s": result.elapsed_s,
            }
            parameters = dict(result.parameters)
            for axis in canonical["parameters"]:
                path, unit = axis["path"], axis["unit"]
                row[_csv_header(f"param:{path}", unit)] = parameters[path]
                row[_csv_header(f"realized:{path}", unit)] = (
                    result.realized_parameters.get(path)
                )
            for metric in canonical["metrics"]:
                row[_csv_header(f"metric:{metric['name']}", metric["unit"])] = (
                    result.metrics.get(metric["name"])
                )
            rows.append(row)
        return rows

    def to_csv(self, path: Union[str, os.PathLike, None] = None) -> str:
        """Return the results as CSV, and write them to ``path`` if given.

        Floats are written as their ``repr``, so :meth:`read_csv` reads back
        exactly the values of :meth:`to_rows`. Units are in the column names.
        """
        rows = self.to_rows()
        buffer = io.StringIO()
        writer = csv.DictWriter(
            buffer, fieldnames=list(rows[0]) if rows else [], lineterminator="\n"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        ""
                        if value is None
                        else repr(value) if isinstance(value, float) else value
                    )
                    for key, value in row.items()
                }
            )
        text = buffer.getvalue()
        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
        return text

    @staticmethod
    def read_csv(path: Union[str, os.PathLike]) -> List[Dict[str, Any]]:
        """Read rows written by :meth:`to_csv`, typed as in :meth:`to_rows`."""
        text_columns = {
            "sweep_id",
            "case_id",
            "status",
            "error_category",
            "error_message",
        }
        with open(path, newline="", encoding="utf-8") as handle:
            rows = []
            for raw in csv.DictReader(handle):
                rows.append(
                    {
                        key: (
                            (value or None)
                            if key in text_columns
                            else _csv_value(value)
                        )
                        for key, value in raw.items()
                    }
                )
        return rows

    def matches(
        self, other: "SweepResult", rtol: float = RTOL, atol: float = ATOL
    ) -> List[str]:
        """Differences from ``other``; empty when the two sweeps agree.

        Compares the sweep ID, the cases in order (ID, status, error category)
        and every metric and realized parameter within ``rtol`` / ``atol``.
        Timings and environment metadata are ignored.
        """
        problems = []
        if self.sweep_id != other.sweep_id:
            problems.append(f"sweep_id {self.sweep_id} != {other.sweep_id}")
        if len(self.results) != len(other.results):
            problems.append(f"{len(self.results)} cases != {len(other.results)} cases")
        for mine, theirs in zip(self.results, other.results):
            label = f"case {mine.case_index}"
            for name in (
                "case_id",
                "case_index",
                "parameters",
                "status",
                "error_category",
            ):
                if getattr(mine, name) != getattr(theirs, name):
                    problems.append(
                        f"{label}: {name} {getattr(mine, name)!r} != {getattr(theirs, name)!r}"
                    )
            for kind in ("metrics", "realized_parameters"):
                a, b = getattr(mine, kind), getattr(theirs, kind)
                if set(a) != set(b):
                    problems.append(f"{label}: {kind} keys differ")
                    continue
                for key in a:
                    if not math.isclose(a[key], b[key], rel_tol=rtol, abs_tol=atol):
                        problems.append(f"{label}: {key} {a[key]!r} != {b[key]!r}")
        return problems


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


def _evaluate(
    snapshot: BaselineSnapshot, spec: SweepSpec, case: SweepCase, sweep_id: str
) -> CaseResult:
    started = time.perf_counter()
    realized: Dict[str, float] = {}
    metrics: Dict[str, float] = {}
    category = message = None
    try:
        try:
            cell = snapshot.new_cell()
        except Exception as exc:
            raise _CaseFailure(RECONSTRUCTION_ERROR, _describe(exc)) from exc
        realized = _apply(cell, spec, case.parameters)
        metrics = _read_metrics(cell, spec.metrics)
    except _CaseFailure as failure:
        realized, metrics = {}, {}
        category, message = failure.category, failure.message
    return CaseResult(
        sweep_id=sweep_id,
        case_id=case.case_id,
        case_index=case.index,
        parameters=case.parameters,
        parameter_units={axis.path: axis.unit for axis in spec.axes},
        realized_parameters=realized,
        metrics=metrics,
        metric_units={name: METRIC_UNITS[name] for name in spec.metrics},
        status=STATUS_OK if category is None else STATUS_FAILED,
        error_category=category,
        error_message=message,
        elapsed_s=time.perf_counter() - started,
    )


def run_sweep(baseline: Any, spec: SweepSpec) -> SweepResult:
    """Evaluate every case of ``spec`` on ``baseline``, sequentially.

    Parameters
    ----------
    baseline : cell or BaselineSnapshot
        The cell to sweep (it is serialized once and never modified), or a
        snapshot of one, e.g. loaded with :meth:`BaselineSnapshot.load` to
        replay a sweep.
    spec : SweepSpec
        What to vary and what to measure.

    Returns
    -------
    SweepResult
        The manifest and one result per case, in case-index order. A case that
        fails is reported in its result and does not stop the others.

    Raises
    ------
    SweepSpecificationError
        If a path or metric does not apply to the baseline cell; nothing is
        evaluated then.
    BaselineSnapshotError
        If the baseline cannot be snapshotted reproducibly.
    """
    if not isinstance(spec, SweepSpec):
        raise SweepSpecificationError(
            f"spec must be a SweepSpec, got {type(spec).__name__}."
        )
    snapshot = (
        baseline
        if isinstance(baseline, BaselineSnapshot)
        else BaselineSnapshot.from_cell(baseline)
    )

    probe = snapshot.new_cell()
    for axis in spec.axes:
        _resolve(probe, axis.path)
    for name in spec.metrics:
        if not isinstance(inspect.getattr_static(type(probe), name, None), property):
            raise SweepSpecificationError(
                f"{type(probe).__name__} has no metric {name!r}."
            )
    try:
        baseline_metrics = _read_metrics(probe, spec.metrics)
    except _CaseFailure as failure:
        raise BaselineSnapshotError(
            f"The baseline's metrics cannot be read: {failure.message}"
        ) from failure

    manifest = _build_manifest(spec, snapshot, baseline_metrics)
    sweep_id = manifest["sweep_id"]
    results = tuple(
        _evaluate(snapshot, spec, case, sweep_id)
        for case in generate_cases(spec, snapshot)
    )
    return SweepResult(manifest=manifest, results=results)
