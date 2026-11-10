"""Excitation flights in the simulator: closed-loop flights that excite the dynamics, logged as FlightLogs.

They serve two purposes: checking that the estimators recover known plant
parameters (the tests do this), and showing what a real excitation flight
should contain (moving set-points for drag, a pseudo-random collective
perturbation for the actuation's delay and time constants).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..controllers.hover import build_hover_controller_config, compute_hover_control
from ..flightlog import FlightLog, rotation_to_quaternion
from ..sim import Simulator

__all__ = ["excitation_flight", "record_flight"]


def _perturbation_schedule(duration_s: float, rng: np.random.Generator, *, min_hold_s: float, max_hold_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Switching times and signs of a random binary signal (holds between min_hold_s and max_hold_s)."""
    times, signs = [0.0], [1.0]
    while times[-1] < duration_s:
        times.append(times[-1] + rng.uniform(min_hold_s, max_hold_s))
        signs.append(-signs[-1])
    return np.asarray(times), np.asarray(signs)


def record_flight(sim: Simulator, steps: int, command, *, measured: bool = False) -> FlightLog:
    """Fly ``steps`` control periods with ``command(state) -> rotor thrusts`` and log them.

    ``measured``: log what the controller receives (the plant's sensor
    model), else the true state. Battery and rotor-speed channels are logged
    when the plant reports them.
    """
    rows: dict[str, list] = {name: [] for name in ("position", "velocity", "quaternion", "angular_velocity_body", "rotor_thrust_command")}
    extra: dict[str, list] = {"battery_voltage": [], "battery_current": [], "rotor_speed": []}
    times = []
    state = sim.state
    for _ in range(steps):
        thrusts = np.asarray(command(state), dtype=float)
        if measured:
            rotation = np.asarray(state["rotation_matrix"], dtype=float).reshape(3, 3)
            rows["position"].append(state["position"])
            rows["velocity"].append(state["velocity"])
            rows["quaternion"].append(rotation_to_quaternion(rotation)[0])
            rows["angular_velocity_body"].append(state["angular_velocity_body"])
        else:
            truth = sim.truth()
            rows["position"].append(truth["position"])
            rows["velocity"].append(truth["velocity"])
            rows["quaternion"].append(truth["quaternion"])
            rows["angular_velocity_body"].append(truth["angular_velocity_body"])
        plant = state.get("plant", {})
        extra["battery_voltage"].append(plant.get("battery_voltage_v"))
        extra["battery_current"].append(plant.get("battery_current_a"))
        extra["rotor_speed"].append(plant.get("rotor_speeds_rad_s"))
        rows["rotor_thrust_command"].append(thrusts)
        times.append(sim.time)
        state = sim.step(thrusts)
    channels = {name: np.asarray(values, dtype=float) for name, values in rows.items()}
    for name, values in extra.items():
        if all(value is not None for value in values):
            channels[name] = np.asarray(values, dtype=float)
    metadata = {
        "source": "wheeled_uav.sysid.record_flight",
        "measured": measured,
        "seed": sim.scene.request.seed,
        "plant": sim.scene.params.get("plant"),
        "randomization_draws": list(sim.scene.randomization_draws),
    }
    return FlightLog(np.asarray(times), channels, metadata)


def excitation_flight(
    sim: Simulator,
    *,
    duration_s: float = 20.0,
    settle_s: float = 2.0,
    seed: int = 0,
    height_m: float = 1.0,
    amplitude_m: float = 0.6,
    collective_perturbation_n: float = 0.8,
    differential_perturbation_n: float = 0.05,
    measured: bool = False,
) -> FlightLog:
    """Take off, then follow a Lissajous set-point with a random collective perturbation; return the log after ``settle_s``.

    The set-point moves at up to about 1 m/s (drag); the collective
    perturbation switches sign every 40-160 ms (delay and rotor time
    constants); a small differential perturbation excites the attitude.
    """
    state = sim.reset()
    params: dict[str, Any] = sim.scene.params
    config = build_hover_controller_config(params)
    rng = np.random.default_rng(seed)
    total_s = settle_s + duration_s
    switch_times, switch_signs = _perturbation_schedule(total_s, rng, min_hold_s=0.04, max_hold_s=0.16)
    differential = rng.choice([-1.0, 1.0], size=(len(switch_times), 4))
    start = np.asarray(state["position"], dtype=float)

    def setpoint(time: float) -> np.ndarray:
        if time < settle_s:
            return np.array([start[0], start[1], height_m])
        moving = time - settle_s
        ramp = min(1.0, moving / 2.0)
        return np.array(
            [
                start[0] + ramp * amplitude_m * math.sin(2.0 * math.pi * 0.23 * moving),
                start[1] + ramp * amplitude_m * math.sin(2.0 * math.pi * 0.31 * moving + 0.5),
                height_m + ramp * 0.3 * math.sin(2.0 * math.pi * 0.17 * moving),
            ]
        )

    def command(current: dict[str, Any]) -> np.ndarray:
        time = float(current["time"])
        thrusts = compute_hover_control(current, setpoint(time), config)
        if time >= settle_s:
            index = int(np.searchsorted(switch_times, time, side="right") - 1)
            thrusts = thrusts + switch_signs[index] * collective_perturbation_n / 4.0 + differential_perturbation_n * differential[index]
        return np.clip(thrusts, 0.0, config.max_rotor_thrust)

    steps = int(round(total_s / sim.control_period))
    log = record_flight(sim, steps, command, measured=measured)
    log.metadata.update({"excitation": {"duration_s": duration_s, "settle_s": settle_s, "seed": seed}})
    return log.window(start=settle_s)
