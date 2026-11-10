"""Data-fitted correction: a body-frame force and torque that the physics model misses.

The correction maps features of the vehicle's state and rotor thrusts to the
part of the measured force and torque that the rest of the plant model does
not explain (fit it with ``wheeled_uav.sysid.fit_correction`` on flight logs,
after the physical parameters are identified). Two kinds:

- ``linear``: ``y = W x_n + b`` on standardized features ``x_n``;
- ``knn``: the mean of the outputs of the ``k`` recorded samples nearest in
  standardized feature space (optionally weighted by inverse distance).

Outputs are clipped to the range seen in the data, so the model
never extrapolates to forces it has not seen.

File format (JSON), version ``wheeled_uav.correction/1``::

    {
      "format": "wheeled_uav.correction/1",
      "kind": "linear" | "knn",
      "features": ["velocity_body_x", ...],   # a subset of FEATURE_NAMES
      "outputs": ["force_body_x", ...],       # a subset of OUTPUT_NAMES
      "feature_mean": [...], "feature_scale": [...],
      "output_min": [...], "output_max": [...],
      "linear": {"weights": [[...], ...], "bias": [...]},             # kind linear
      "knn": {"k": 8, "weighting": "uniform" | "distance",
              "points": [[...], ...], "values": [[...], ...]},      # kind knn, raw units
      "provenance": {...}                                            # free-form
    }
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "CORRECTION_FORMAT",
    "FEATURE_NAMES",
    "OUTPUT_NAMES",
    "CorrectionModel",
    "fit_knn_correction",
    "fit_linear_correction",
]

CORRECTION_FORMAT = "wheeled_uav.correction/1"

# Canonical features, all computable from a flight log as well as in the simulator.
FEATURE_NAMES = (
    "velocity_body_x",  # airspeed in the body frame [m/s] (velocity minus wind)
    "velocity_body_y",
    "velocity_body_z",
    "angular_velocity_body_x",  # [rad/s]
    "angular_velocity_body_y",
    "angular_velocity_body_z",
    "rotor_thrust_1",  # delivered thrust of each rotor [N], in actuation.rotors order
    "rotor_thrust_2",
    "rotor_thrust_3",
    "rotor_thrust_4",
    "collective_thrust",  # sum of the rotor thrusts [N]
    "airspeed",  # |airspeed| [m/s]
    "height",  # body origin above the ground [m]
)

# Body-frame force [N] at the centre of mass and torque [N m] about it.
OUTPUT_NAMES = ("force_body_x", "force_body_y", "force_body_z", "torque_body_x", "torque_body_y", "torque_body_z")


def _as_matrix(values: Any, rows: int | None, columns: int, what: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[1] != columns or (rows is not None and array.shape[0] != rows):
        raise ValueError(f"correction: {what} has shape {array.shape}, expected ({rows if rows is not None else 'N'}, {columns})")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"correction: {what} contains non-finite values")
    return array


def _as_vector(values: Any, length: int, what: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.shape != (length,) or not np.all(np.isfinite(array)):
        raise ValueError(f"correction: {what} must be {length} finite numbers")
    return array


@dataclass
class CorrectionModel:
    kind: str
    features: tuple[str, ...]
    outputs: tuple[str, ...]
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    output_min: np.ndarray
    output_max: np.ndarray
    weights: np.ndarray | None = None
    bias: np.ndarray | None = None
    k: int = 8
    weighting: str = "uniform"
    points: np.ndarray | None = None  # standardized
    values: np.ndarray | None = None
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        unknown = [name for name in self.features if name not in FEATURE_NAMES]
        if unknown or not self.features:
            raise ValueError(f"correction: unknown or no features {unknown}; allowed: {', '.join(FEATURE_NAMES)}")
        unknown = [name for name in self.outputs if name not in OUTPUT_NAMES]
        if unknown or not self.outputs:
            raise ValueError(f"correction: unknown or no outputs {unknown}; allowed: {', '.join(OUTPUT_NAMES)}")
        if np.any(self.feature_scale <= 0.0):
            raise ValueError("correction: feature_scale must be positive")
        self._output_rows = np.array([OUTPUT_NAMES.index(name) for name in self.outputs])
        self._feature_columns = np.array([FEATURE_NAMES.index(name) for name in self.features])

    # -- evaluation -----------------------------------------------------
    def evaluate(self, feature_vector: np.ndarray) -> np.ndarray:
        """Body-frame [force; torque] (6 values) for a full FEATURE_NAMES vector."""
        x = (np.asarray(feature_vector, dtype=float)[self._feature_columns] - self.feature_mean) / self.feature_scale
        if self.kind == "linear":
            y = self.weights @ x + self.bias
        else:
            distances = np.linalg.norm(self.points - x, axis=1)
            k = min(self.k, distances.size)
            nearest = np.argpartition(distances, k - 1)[:k]
            if self.weighting == "distance":
                weights = 1.0 / np.maximum(distances[nearest], 1.0e-9)
                y = weights @ self.values[nearest] / np.sum(weights)
            else:
                y = np.mean(self.values[nearest], axis=0)
        y = np.clip(y, self.output_min, self.output_max)
        result = np.zeros(len(OUTPUT_NAMES))
        result[self._output_rows] = y
        return result

    # -- files ------------------------------------------------------------
    @classmethod
    def from_dict(cls, document: dict[str, Any]) -> CorrectionModel:
        if not isinstance(document, dict):
            raise ValueError("correction: the file must hold a JSON object")
        if document.get("format") != CORRECTION_FORMAT:
            raise ValueError(f'correction: "format" must be "{CORRECTION_FORMAT}", got {document.get("format")!r}')
        kind = document.get("kind")
        if kind not in ("linear", "knn"):
            raise ValueError('correction: "kind" must be "linear" or "knn"')
        for name in ("features", "outputs"):
            names = document.get(name, [])
            if not isinstance(names, list) or not all(isinstance(entry, str) for entry in names):
                raise ValueError(f'correction: "{name}" must be a list of names')
        for name in (kind, "provenance"):
            if not isinstance(document.get(name, {}), dict):
                raise ValueError(f'correction: "{name}" must be an object')
        features = tuple(document.get("features", ()))
        outputs = tuple(document.get("outputs", ()))
        n_features, n_outputs = len(features), len(outputs)
        common = dict(
            kind=kind,
            features=features,
            outputs=outputs,
            feature_mean=_as_vector(document.get("feature_mean"), n_features, "feature_mean"),
            feature_scale=_as_vector(document.get("feature_scale"), n_features, "feature_scale"),
            output_min=_as_vector(document.get("output_min"), n_outputs, "output_min"),
            output_max=_as_vector(document.get("output_max"), n_outputs, "output_max"),
            provenance=dict(document.get("provenance", {})),
        )
        if kind == "linear":
            linear = document.get("linear", {})
            return cls(
                weights=_as_matrix(linear.get("weights"), n_outputs, n_features, "linear.weights"),
                bias=_as_vector(linear.get("bias"), n_outputs, "linear.bias"),
                **common,
            )
        knn = document.get("knn", {})
        points = _as_matrix(knn.get("points"), None, n_features, "knn.points")
        values = _as_matrix(knn.get("values"), points.shape[0], n_outputs, "knn.values")
        k = knn.get("k", 8)
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise ValueError("correction: knn.k must be a positive integer")
        weighting = knn.get("weighting", "uniform")
        if weighting not in ("uniform", "distance"):
            raise ValueError('correction: knn.weighting must be "uniform" or "distance"')
        model = cls(k=k, weighting=weighting, points=points, values=values, **common)
        model.points = (points - model.feature_mean) / model.feature_scale
        return model

    @classmethod
    def from_file(cls, path: str | Path) -> CorrectionModel:
        with Path(path).open(encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    def to_dict(self) -> dict[str, Any]:
        document: dict[str, Any] = {
            "format": CORRECTION_FORMAT,
            "kind": self.kind,
            "features": list(self.features),
            "outputs": list(self.outputs),
            "feature_mean": self.feature_mean.tolist(),
            "feature_scale": self.feature_scale.tolist(),
            "output_min": self.output_min.tolist(),
            "output_max": self.output_max.tolist(),
        }
        if self.kind == "linear":
            document["linear"] = {"weights": self.weights.tolist(), "bias": self.bias.tolist()}
        else:
            raw_points = self.points * self.feature_scale + self.feature_mean
            document["knn"] = {"k": self.k, "weighting": self.weighting, "points": raw_points.tolist(), "values": self.values.tolist()}
        document["provenance"] = self.provenance
        return document

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=1) + "\n", encoding="utf-8")


def _standardization(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(features, axis=0)
    scale = np.std(features, axis=0)
    return mean, np.where(scale > 1.0e-12, scale, 1.0)


def _checked(features: np.ndarray, targets: np.ndarray, feature_names, output_names) -> tuple[np.ndarray, np.ndarray]:
    features = np.asarray(features, dtype=float)
    targets = np.asarray(targets, dtype=float)
    if features.ndim != 2 or features.shape[1] != len(feature_names):
        raise ValueError("features must be an (N, len(feature_names)) array")
    if targets.ndim != 2 or targets.shape != (features.shape[0], len(output_names)):
        raise ValueError("targets must be an (N, len(output_names)) array")
    finite = np.all(np.isfinite(features), axis=1) & np.all(np.isfinite(targets), axis=1)
    if np.count_nonzero(finite) < 2:
        raise ValueError("need at least two finite samples to fit a correction")
    return features[finite], targets[finite]


def fit_linear_correction(
    features: np.ndarray,
    targets: np.ndarray,
    feature_names: tuple[str, ...] | list[str],
    output_names: tuple[str, ...] | list[str],
    *,
    ridge: float = 1.0e-3,
    provenance: dict[str, Any] | None = None,
) -> CorrectionModel:
    """Ridge regression of ``targets`` on standardized ``features``."""
    features, targets = _checked(features, targets, feature_names, output_names)
    mean, scale = _standardization(features)
    x = (features - mean) / scale
    design = np.hstack([x, np.ones((x.shape[0], 1))])
    regularizer = ridge * x.shape[0] * np.eye(design.shape[1])
    regularizer[-1, -1] = 0.0  # no penalty on the bias
    solution = np.linalg.solve(design.T @ design + regularizer, design.T @ targets)
    return CorrectionModel(
        kind="linear",
        features=tuple(feature_names),
        outputs=tuple(output_names),
        feature_mean=mean,
        feature_scale=scale,
        output_min=np.min(targets, axis=0),
        output_max=np.max(targets, axis=0),
        weights=solution[:-1].T.copy(),
        bias=solution[-1].copy(),
        provenance=dict(provenance or {}),
    )


def fit_knn_correction(
    features: np.ndarray,
    targets: np.ndarray,
    feature_names: tuple[str, ...] | list[str],
    output_names: tuple[str, ...] | list[str],
    *,
    k: int = 8,
    weighting: str = "uniform",
    max_points: int = 4000,
    seed: int = 0,
    provenance: dict[str, Any] | None = None,
) -> CorrectionModel:
    """Store (a random subset of at most ``max_points``) samples for k-nearest-neighbour lookup."""
    features, targets = _checked(features, targets, feature_names, output_names)
    if features.shape[0] > max_points:
        keep = np.sort(np.random.default_rng(seed).choice(features.shape[0], size=max_points, replace=False))
        features, targets = features[keep], targets[keep]
    mean, scale = _standardization(features)
    model = CorrectionModel(
        kind="knn",
        features=tuple(feature_names),
        outputs=tuple(output_names),
        feature_mean=mean,
        feature_scale=scale,
        output_min=np.min(targets, axis=0),
        output_max=np.max(targets, axis=0),
        k=int(k),
        weighting=weighting,
        points=(features - mean) / scale,
        values=targets.copy(),
        provenance=dict(provenance or {}),
    )
    return model
