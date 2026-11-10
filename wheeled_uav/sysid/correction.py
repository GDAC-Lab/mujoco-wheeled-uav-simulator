"""Fit the data-fitted correction (wheeled_uav.plant.correction) to the force the physical model leaves unexplained.

The target is the body-frame force ``m f_b - F_thrust - F_drag`` (measured
minus modeled, with the identified actuation and drag), the features are
those of wheeled_uav.plant.correction.FEATURE_NAMES computed from the log.
Fit it after the physical parameters, on flights other than the ones used to
judge it (check the k-step prediction error on validation flights).
"""

from __future__ import annotations

import numpy as np

from ..flightlog import FlightLog
from ..plant.correction import FEATURE_NAMES, CorrectionModel, fit_knn_correction, fit_linear_correction
from .actuation import ActuationFit, simulate_rotor_thrusts
from .signals import body_velocity, specific_force_body

__all__ = ["DEFAULT_CORRECTION_FEATURES", "correction_dataset", "fit_correction", "stencil_average"]

DEFAULT_CORRECTION_FEATURES = (
    "velocity_body_x",
    "velocity_body_y",
    "velocity_body_z",
    "angular_velocity_body_x",
    "angular_velocity_body_y",
    "angular_velocity_body_z",
    "collective_thrust",
)
FORCE_OUTPUTS = ("force_body_x", "force_body_y", "force_body_z")


def stencil_average(values: np.ndarray) -> np.ndarray:
    """(x[k-1] + 2 x[k] + x[k+1]) / 4: the mean over +-1 sample that a central-difference derivative sees."""
    values = np.asarray(values, dtype=float)
    result = values.copy()
    result[1:-1] = 0.25 * values[:-2] + 0.5 * values[1:-1] + 0.25 * values[2:]
    return result


def _drag_force(airspeed: np.ndarray, aerodynamics: dict) -> np.ndarray:
    linear = np.asarray(aerodynamics.get("rotor_drag_coefficients_n_s_per_m", (0.0, 0.0, 0.0)), dtype=float)
    area = np.asarray(aerodynamics.get("body_drag_area_m2", (0.0, 0.0, 0.0)), dtype=float)
    density = float(aerodynamics.get("air_density_kg_per_m3", 1.204))
    return -(linear * airspeed + 0.5 * density * area * np.abs(airspeed) * airspeed)


def correction_dataset(
    log: FlightLog,
    *,
    mass: float,
    gravity: float,
    actuation: ActuationFit,
    aerodynamics: dict | None = None,
    rotor_axes: np.ndarray | None = None,
    ground_height_m: float = 0.0,
    smooth_s: float = 0.0,
    edge_s: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Features (N, len(FEATURE_NAMES)) and unexplained body forces (N, 3) at the usable samples."""
    aerodynamics = aerodynamics or {}
    wind = aerodynamics.get("wind_m_per_s")
    thrusts = simulate_rotor_thrusts(log, actuation)
    axes = np.tile([0.0, 0.0, 1.0], (thrusts.shape[1], 1)) if rotor_axes is None else np.asarray(rotor_axes, dtype=float)
    airspeed = body_velocity(log, wind)
    modeled = thrusts @ axes + _drag_force(airspeed, aerodynamics)
    if not log.has("specific_force_body"):
        modeled = stencil_average(modeled)  # the derivative of the velocity averages the force over +-1 sample
    unexplained = mass * specific_force_body(log, gravity, smooth_s) - modeled
    rates = log["angular_velocity_body"] if log.has("angular_velocity_body") else np.zeros((len(log), 3))
    height = log["position"][:, 2] - ground_height_m
    features = np.column_stack(
        [airspeed, rates, thrusts, np.sum(thrusts, axis=1), np.linalg.norm(airspeed, axis=1), height]
    )
    used = log.valid_mask() & (log.time >= log.time[0] + edge_s) & (log.time <= log.time[-1] - edge_s)
    used &= np.all(np.isfinite(features), axis=1) & np.all(np.isfinite(unexplained), axis=1)
    return features[used], unexplained[used]


def fit_correction(
    log: FlightLog,
    *,
    mass: float,
    gravity: float,
    actuation: ActuationFit,
    aerodynamics: dict | None = None,
    rotor_axes: np.ndarray | None = None,
    kind: str = "knn",
    features=DEFAULT_CORRECTION_FEATURES,
    k: int = 8,
    ridge: float = 1.0e-3,
    max_points: int = 4000,
    provenance: dict | None = None,
) -> CorrectionModel:
    """A correction of the body forces (``kind`` "knn" or "linear") over the chosen features."""
    all_features, targets = correction_dataset(
        log, mass=mass, gravity=gravity, actuation=actuation, aerodynamics=aerodynamics, rotor_axes=rotor_axes
    )
    columns = [FEATURE_NAMES.index(name) for name in features]
    chosen = all_features[:, columns]
    details = {"samples": int(targets.shape[0]), "unexplained_rms_n": np.sqrt(np.mean(targets**2, axis=0)).tolist(), **(provenance or {})}
    if kind == "linear":
        return fit_linear_correction(chosen, targets, tuple(features), FORCE_OUTPUTS, ridge=ridge, provenance=details)
    if kind == "knn":
        return fit_knn_correction(chosen, targets, tuple(features), FORCE_OUTPUTS, k=k, max_points=max_points, provenance=details)
    raise ValueError('kind must be "knn" or "linear"')
