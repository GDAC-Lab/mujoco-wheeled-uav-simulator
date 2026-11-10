"""Identify the drag in the body x and y axes from a flight log.

``m f_b = F_thrust,b - D v_b - q |v_b| v_b + offset`` per body axis, with
``f_b`` the specific force (accelerometer or derivative of the velocity),
``v_b`` the body-frame airspeed and ``F_thrust,b`` the thrust's component
along the axis (zero for rotors whose thrust is along body z). Linear least
squares per axis, coefficients constrained to be non-negative.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from ..flightlog import FlightLog
from .signals import body_velocity, derivative_order, specific_force_body, usable_mask

__all__ = ["DragFit", "fit_drag"]


@dataclass(frozen=True)
class DragFit:
    rotor_drag_coefficients_n_s_per_m: tuple[float, float]  # body x, y
    body_drag_area_m2: tuple[float, float] | None  # body x, y (quadratic term), when fitted
    offset_n: tuple[float, float]
    rms_n: tuple[float, float]
    samples: int

    def as_dict(self) -> dict:
        return asdict(self)


def _nonnegative_least_squares(design: np.ndarray, target: np.ndarray, constrained: list[int]) -> np.ndarray:
    """Least squares with the listed coefficients kept >= 0 (drop the negative ones and refit)."""
    active = list(range(design.shape[1]))
    while True:
        solution = np.zeros(design.shape[1])
        solution[active], *_ = np.linalg.lstsq(design[:, active], target, rcond=None)
        negative = [index for index in constrained if index in active and solution[index] < 0.0]
        if not negative:
            return solution
        active.remove(min(negative, key=lambda index: solution[index]))


def fit_drag(
    log: FlightLog,
    *,
    mass: float,
    gravity: float = 9.81,
    quadratic: bool = False,
    air_density_kg_per_m3: float = 1.204,
    wind_m_per_s=None,
    thrust_force_body: np.ndarray | None = None,
    smooth_s: float = 0.0,
    edge_s: float = 0.1,
) -> DragFit:
    """Fit the linear (and optionally quadratic) drag of the body x and y axes."""
    force = mass * specific_force_body(log, gravity, smooth_s)
    if thrust_force_body is not None:
        force = force - np.asarray(thrust_force_body, dtype=float)
    airspeed = body_velocity(log, wind_m_per_s)
    # Each central difference averages the force over +-1 sample: average the airspeed the same way.
    for _ in range(derivative_order(log)):
        airspeed = 0.5 * airspeed + 0.25 * (np.roll(airspeed, 1, axis=0) + np.roll(airspeed, -1, axis=0))
    used = usable_mask(log, smooth_s) & (log.time >= log.time[0] + edge_s) & (log.time <= log.time[-1] - edge_s)
    used &= np.all(np.isfinite(force), axis=1)
    if np.count_nonzero(used) < 20:
        raise ValueError("fit_drag: fewer than 20 usable samples")
    linear, area, offsets, rms = [], [], [], []
    for axis in (0, 1):
        speed = airspeed[used, axis]
        columns = [-speed]
        if quadratic:
            columns.append(-0.5 * air_density_kg_per_m3 * np.abs(speed) * speed)
        columns.append(np.ones_like(speed))
        design = np.column_stack(columns)
        solution = _nonnegative_least_squares(design, force[used, axis], constrained=list(range(len(columns) - 1)))
        residual = force[used, axis] - design @ solution
        linear.append(float(solution[0]))
        if quadratic:
            area.append(float(solution[1]))
        offsets.append(float(solution[-1]))
        rms.append(float(np.sqrt(np.mean(residual**2))))
    return DragFit(
        rotor_drag_coefficients_n_s_per_m=(linear[0], linear[1]),
        body_drag_area_m2=(area[0], area[1]) if quadratic else None,
        offset_n=(offsets[0], offsets[1]),
        rms_n=(rms[0], rms[1]),
        samples=int(np.count_nonzero(used)),
    )
