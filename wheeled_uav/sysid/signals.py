"""Signal helpers shared by the estimators: smoothing, derivatives, delays, derived channels."""

from __future__ import annotations

import numpy as np

from ..flightlog import FlightLog

__all__ = [
    "body_velocity",
    "estimate_delay",
    "smooth",
    "specific_force_body",
    "world_acceleration",
]


def smooth(values: np.ndarray, window: int) -> np.ndarray:
    """Centred moving average over ``window`` samples (zero phase); the window shrinks at the ends."""
    values = np.asarray(values, dtype=float)
    if window <= 1:
        return values.copy()
    flat = values.reshape(values.shape[0], -1)
    half = window // 2
    cumulative = np.vstack([np.zeros((1, flat.shape[1])), np.cumsum(flat, axis=0)])
    index = np.arange(flat.shape[0])
    low = np.clip(index - half, 0, flat.shape[0])
    high = np.clip(index + half + 1, 0, flat.shape[0])
    result = (cumulative[high] - cumulative[low]) / (high - low)[:, None]
    return result.reshape(values.shape)


def world_acceleration(log: FlightLog, smooth_s: float = 0.0) -> np.ndarray:
    """World acceleration from the velocity channel (or the position), by central differences."""
    if log.has("velocity"):
        velocity = log["velocity"]
    else:
        velocity = np.gradient(log["position"], log.time, axis=0)
    window = int(round(smooth_s / log.sample_period)) if smooth_s > 0.0 else 0
    return np.gradient(smooth(velocity, window), log.time, axis=0)


def specific_force_body(log: FlightLog, gravity: float, smooth_s: float = 0.0) -> np.ndarray:
    """What an accelerometer reads, in the body frame: the channel if logged, else from the velocity.

    ``gravity`` is the magnitude (9.81); the world z axis points up.
    """
    if log.has("specific_force_body"):
        force = log["specific_force_body"]
        window = int(round(smooth_s / log.sample_period)) if smooth_s > 0.0 else 0
        return smooth(force, window)
    acceleration = world_acceleration(log, smooth_s)
    acceleration[:, 2] += gravity
    return np.einsum("nji,nj->ni", log.rotation, acceleration)


def body_velocity(log: FlightLog, wind: np.ndarray | None = None) -> np.ndarray:
    """Airspeed in the body frame (world velocity minus the wind, rotated into the body)."""
    velocity = log["velocity"] if log.has("velocity") else np.gradient(log["position"], log.time, axis=0)
    if wind is not None:
        velocity = velocity - np.asarray(wind, dtype=float)
    return np.einsum("nji,nj->ni", log.rotation, velocity)


def estimate_delay(time: np.ndarray, cause: np.ndarray, effect: np.ndarray, *, max_delay_s: float, rate_hz: float = 1000.0) -> float:
    """Delay of ``effect`` behind ``cause`` [s] by cross-correlation of their derivatives.

    Both are resampled to ``rate_hz``; the peak is refined by a parabola
    through its neighbours. A first-order lag adds its time constant to what
    this finds (the lag's group delay), so use it for pure delays (or as the
    start of a joint fit).
    """
    grid = np.arange(time[0], time[-1], 1.0 / rate_hz)
    cause_rate = np.gradient(np.interp(grid, time, cause), grid)
    effect_rate = np.gradient(np.interp(grid, time, effect), grid)
    cause_rate -= cause_rate.mean()
    effect_rate -= effect_rate.mean()
    max_lag = int(round(max_delay_s * rate_hz))
    n = grid.size
    correlation = np.array([np.dot(cause_rate[: n - lag], effect_rate[lag:]) / (n - lag) for lag in range(max_lag + 1)])
    best = int(np.argmax(correlation))
    offset = 0.0
    if 0 < best < max_lag:
        left, middle, right = correlation[best - 1 : best + 2]
        denominator = left - 2.0 * middle + right
        if denominator < 0.0:
            offset = 0.5 * (left - right) / denominator
    return (best + offset) / rate_hz
