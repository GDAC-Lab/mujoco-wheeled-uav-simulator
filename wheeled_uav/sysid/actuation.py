"""Identify the rotor actuation from a flight log: delay, time constants (up/down), thrust scale.

Model (the plant model's, wheeled_uav.plant.actuation): each rotor's
commanded thrust is delayed by ``d``; ``s = sqrt(T)`` (proportional to the
rotor speed) follows ``sqrt(command)`` as a first-order lag with ``tau_up``
or ``tau_down``; the thrust along the body z axis is
``scale * sum(T_i) - d_z v_bz + offset``. The measurement is the mass times
the specific force along body z (accelerometer, or the derivative of the
velocity). Scaling commutes with the lag in ``sqrt(T)``, so for each delay
and pair of time constants the scale, vertical drag and offset follow by
linear least squares; the delay and time constants are searched on a grid:
first a symmetric time constant (an LTI filter, computed by FFT for all
delays at once), then up and down separately around the best.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from ..flightlog import FlightLog
from .signals import body_velocity, smooth, specific_force_body

__all__ = ["ActuationFit", "fit_actuation", "simulate_rotor_thrusts"]


@dataclass(frozen=True)
class ActuationFit:
    delay_s: float
    motor_time_constant_up_s: float
    motor_time_constant_down_s: float
    thrust_scale: float
    vertical_drag_n_s_per_m: float
    thrust_offset_n: float
    rms_n: float
    r_squared: float
    samples: int

    def plant_actuation(self) -> dict[str, float]:
        """The plant.actuation keys this fit determines."""
        return {
            "delay_s": round(self.delay_s, 6),
            "motor_time_constant_up_s": round(self.motor_time_constant_up_s, 6),
            "motor_time_constant_down_s": round(self.motor_time_constant_down_s, 6),
            "thrust_scale": round(self.thrust_scale, 6),
        }

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _lag_symmetric(target: np.ndarray, alpha: float) -> np.ndarray:
    """First-order lag y[k] = y[k-1] + alpha (x[k] - y[k-1]), y[-1] = x[0], along axis 0, via FFT convolution."""
    n = target.shape[0]
    if alpha >= 1.0:
        return target.copy()
    decay = 1.0 - alpha
    length = min(n, int(math.ceil(math.log(1.0e-12) / math.log(decay))) + 1)
    kernel = alpha * decay ** np.arange(length)
    size = 1 << int(math.ceil(math.log2(n + length)))
    spectrum = np.fft.rfft(kernel, size)
    start = target[0]
    centred = target - start
    result = np.fft.irfft(np.fft.rfft(centred, size, axis=0) * spectrum[:, None], size, axis=0)[:n]
    return result + start


def _lag_asymmetric(target: np.ndarray, alpha_up: np.ndarray, alpha_down: np.ndarray) -> np.ndarray:
    """The switching lag for K (alpha_up, alpha_down) pairs at once: returns (n, K) collective thrust."""
    state = np.repeat(target[:1], alpha_up.size, axis=0)  # (K, rotors)
    up = alpha_up[:, None]
    down = alpha_down[:, None]
    collective = np.empty((target.shape[0], alpha_up.size))
    for index in range(target.shape[0]):
        goal = target[index][None, :]
        state = state + np.where(goal > state, up, down) * (goal - state)
        collective[index] = np.sum(state * state, axis=1)
    return collective


def _shift(values: np.ndarray, steps: int) -> np.ndarray:
    """Delay by ``steps`` samples along axis 0, holding the first value."""
    if steps <= 0:
        return values
    return np.concatenate([np.repeat(values[:1], steps, axis=0), values[:-steps]], axis=0)


def _alpha(time_constant: float, dt: float) -> float:
    return 1.0 if time_constant <= 0.0 else 1.0 - math.exp(-dt / time_constant)


def _box_half_open(values: np.ndarray, half: int) -> np.ndarray:
    """Mean over grid indices [i - half, i + half): the average force between two velocity samples 2 * half apart."""
    if half <= 0:
        return values
    n = values.shape[0]
    cumulative = np.concatenate([np.zeros((1,) + values.shape[1:]), np.cumsum(values, axis=0)])
    index = np.arange(n)
    low = np.clip(index - half, 0, n)
    high = np.clip(index + half, 1, n)
    low = np.minimum(low, high - 1)
    return (cumulative[high] - cumulative[low]) / (high - low).reshape((-1,) + (1,) * (values.ndim - 1))


@dataclass
class _Problem:
    grid: np.ndarray
    dt: float
    sqrt_command: np.ndarray  # (n_grid, rotors)
    sample_index: np.ndarray  # grid index of each used log sample
    measured: np.ndarray  # mass * specific force along body z at the used samples
    vertical_speed: np.ndarray
    # How the measurement averages the force: a central difference over +-half grid steps, then the smoothing.
    stencil_half: int
    smoothing: int

    def measure_like(self, collective_grid: np.ndarray) -> np.ndarray:
        """The model's collective force put through the measurement's averaging, at the used samples."""
        averaged = _box_half_open(collective_grid, self.stencil_half)
        return smooth(averaged, self.smoothing)[self.sample_index]

    def regress(self, collective_grid: np.ndarray, fit_vertical_drag: bool) -> tuple[np.ndarray, np.ndarray]:
        """Least squares for each column of collective_grid (n_grid, K): SSE (K,) and [scale, speed, offset] (3, K)."""
        filtered = self.measure_like(collective_grid)
        sse = np.empty(filtered.shape[1])
        coefficients = np.zeros((3, filtered.shape[1]))
        for k in range(filtered.shape[1]):
            design = [filtered[:, k], np.ones_like(self.measured)]
            if fit_vertical_drag:
                design.insert(1, self.vertical_speed)
            design = np.column_stack(design)
            solution, *_ = np.linalg.lstsq(design, self.measured, rcond=None)
            residual = self.measured - design @ solution
            sse[k] = float(residual @ residual)
            coefficients[:, k] = solution if fit_vertical_drag else [solution[0], 0.0, solution[1]]
        return sse, coefficients


def _problem(log: FlightLog, *, mass: float, gravity: float, rate_hz: float, smooth_s: float, min_collective_fraction: float, edge_s: float) -> _Problem:
    if not log.has("rotor_thrust_command"):
        raise ValueError("fit_actuation needs the rotor_thrust_command channel")
    dt = 1.0 / rate_hz
    grid = np.arange(log.time[0], log.time[-1], dt)
    command = np.maximum(log.sample("rotor_thrust_command", grid), 0.0)
    force = mass * specific_force_body(log, gravity, smooth_s)[:, 2]
    vertical_speed = body_velocity(log)[:, 2]
    collective = np.sum(log["rotor_thrust_command"], axis=1)
    used = log.valid_mask()
    used &= (log.time >= log.time[0] + edge_s) & (log.time <= log.time[-1] - edge_s)
    used &= collective >= min_collective_fraction * mass * gravity
    used &= np.isfinite(force)
    if np.count_nonzero(used) < 20:
        raise ValueError("fit_actuation: fewer than 20 usable samples (in flight, away from the ends, valid)")
    sample_index = np.clip(np.round((log.time[used] - grid[0]) / dt).astype(int), 0, grid.size - 1)
    period = log.sample_period
    smoothing_samples = int(round(smooth_s / period)) if smooth_s > 0.0 else 0
    return _Problem(
        grid=grid,
        dt=dt,
        sqrt_command=np.sqrt(command),
        sample_index=sample_index,
        measured=force[used],
        vertical_speed=vertical_speed[used],
        stencil_half=0 if log.has("specific_force_body") else int(round(period / dt)),
        smoothing=max(1, int(round(smoothing_samples * period / dt))) if smoothing_samples > 1 else 1,
    )


def fit_actuation(
    log: FlightLog,
    *,
    mass: float,
    gravity: float = 9.81,
    max_delay_s: float = 0.04,
    rate_hz: float = 1000.0,
    smooth_s: float = 0.0,
    time_constants_s: np.ndarray | None = None,
    asymmetric: bool = True,
    fit_vertical_drag: bool = True,
    min_collective_fraction: float = 0.3,
    edge_s: float = 0.1,
) -> ActuationFit:
    """Fit delay, time constants and thrust scale (see the module docstring)."""
    problem = _problem(
        log, mass=mass, gravity=gravity, rate_hz=rate_hz, smooth_s=smooth_s, min_collective_fraction=min_collective_fraction, edge_s=edge_s
    )
    dt = problem.dt
    taus = np.geomspace(0.004, 0.2, 40) if time_constants_s is None else np.asarray(time_constants_s, dtype=float)
    delays = np.arange(0, int(round(max_delay_s / dt)) + 1)
    # Stage 1: symmetric time constant, every delay (the lag is LTI, so the delay is a shift of its output).
    best = (math.inf, 0, float(taus[0]))
    for tau in taus:
        lagged = _lag_symmetric(problem.sqrt_command, _alpha(float(tau), dt))
        collective = np.sum(lagged * lagged, axis=1)
        shifted = np.column_stack([_shift(collective, int(steps)) for steps in delays])
        sse, _ = problem.regress(shifted, fit_vertical_drag)
        k = int(np.argmin(sse))
        if sse[k] < best[0]:
            best = (float(sse[k]), int(delays[k]), float(tau))
    _, delay_steps, tau = best
    tau_up = tau_down = tau
    # Stage 2: up and down separately, on grids that shrink around the best pair (about 1 % at the end),
    # at the delays next to the best one.
    if asymmetric:
        spread, reach = 2.0, 2
        for _ in range(5):
            factors = np.geomspace(1.0 / spread, spread, 9)
            pairs = np.array([(tau_up * up, tau_down * down) for up in factors for down in factors])
            alpha_up = np.array([_alpha(value, dt) for value in pairs[:, 0]])
            alpha_down = np.array([_alpha(value, dt) for value in pairs[:, 1]])
            best_sse, chosen = math.inf, (delay_steps, tau_up, tau_down)
            for steps in range(max(0, delay_steps - reach), delay_steps + reach + 1):
                collective = _lag_asymmetric(_shift(problem.sqrt_command, steps), alpha_up, alpha_down)
                sse, _ = problem.regress(collective, fit_vertical_drag)
                k = int(np.argmin(sse))
                if sse[k] < best_sse:
                    best_sse, chosen = float(sse[k]), (steps, float(pairs[k, 0]), float(pairs[k, 1]))
            delay_steps, tau_up, tau_down = chosen
            spread, reach = math.sqrt(spread), 1
    collective = _lag_asymmetric(
        _shift(problem.sqrt_command, delay_steps), np.array([_alpha(tau_up, dt)]), np.array([_alpha(tau_down, dt)])
    )
    sse, coefficients = problem.regress(collective, fit_vertical_drag)
    scale, speed_coefficient, offset = coefficients[:, 0]
    residual_rms = math.sqrt(sse[0] / problem.measured.size)
    variance = float(np.var(problem.measured))
    return ActuationFit(
        delay_s=delay_steps * dt,
        motor_time_constant_up_s=float(tau_up),
        motor_time_constant_down_s=float(tau_down),
        thrust_scale=float(scale),
        vertical_drag_n_s_per_m=float(-speed_coefficient),
        thrust_offset_n=float(offset),
        rms_n=residual_rms,
        r_squared=1.0 - residual_rms**2 / variance if variance > 0.0 else float("nan"),
        samples=int(problem.measured.size),
    )


def simulate_rotor_thrusts(log: FlightLog, fit: ActuationFit, *, rate_hz: float = 1000.0) -> np.ndarray:
    """Each rotor's delivered thrust at the log's samples under the fitted actuation (N, rotors)."""
    dt = 1.0 / rate_hz
    grid = np.arange(log.time[0], log.time[-1] + dt, dt)
    target = _shift(np.sqrt(np.maximum(log.sample("rotor_thrust_command", grid), 0.0)), int(round(fit.delay_s / dt)))
    alpha_up, alpha_down = _alpha(fit.motor_time_constant_up_s, dt), _alpha(fit.motor_time_constant_down_s, dt)
    state = target[0].copy()
    thrusts = np.empty_like(target)
    for index in range(grid.size):
        goal = target[index]
        state = state + np.where(goal > state, alpha_up, alpha_down) * (goal - state)
        thrusts[index] = state * state
    sample_index = np.clip(np.round((log.time - grid[0]) / dt).astype(int), 0, grid.size - 1)
    return fit.thrust_scale * thrusts[sample_index]
