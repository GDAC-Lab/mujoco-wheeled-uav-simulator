"""k-step prediction error: how far the simulator drifts from a log when started from it.

From many starting samples of the log, the simulator is put in the logged
state, fed the logged rotor commands, and compared with the log after each
horizon. This is the standard check of a simulator against real flights
(it isolates the plant from the controller: the commands are replayed, not
recomputed) and the number to compare between plant models (with and
without a correction, for example).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..flightlog import FlightLog, quaternion_to_rotation
from ..sim import Simulator

__all__ = ["prediction_error"]


def _statistics(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "max": float(np.max(array)),
    }


def prediction_error(
    log: FlightLog,
    sim: Simulator,
    *,
    horizons_s=(0.05, 0.1, 0.2, 0.5),
    stride_s: float = 0.25,
    warm_up_s: float = 0.3,
    uav_index: int = 0,
) -> dict[str, Any]:
    """Errors at each horizon over starts every ``stride_s`` (position [m], velocity [m/s], attitude [deg]).

    Before each start the plant's rotor model runs through the logged
    commands of the previous ``warm_up_s`` (its delay line and rotor speeds
    are then where the flight had them); the vehicle is put in the logged
    state, with the wheels at rest relative to the body (logs do not
    record wheel speeds). Single-UAV scenes.
    """
    if not log.has("rotor_thrust_command"):
        raise ValueError("prediction_error needs the rotor_thrust_command channel")
    horizons = sorted(float(h) for h in horizons_s)
    sim.reset()
    period = sim.control_period
    steps_for = {h: max(1, int(round(h / period))) for h in horizons}
    last = steps_for[horizons[-1]]
    valid = log.valid_mask()
    velocity = log["velocity"] if log.has("velocity") else np.gradient(log["position"], log.time, axis=0)
    errors: dict[float, dict[str, list[float]]] = {h: {"position_m": [], "velocity_m_per_s": [], "attitude_deg": []} for h in horizons}
    starts = 0
    # The rotor model's warm-up needs the commands before the start.
    start_time = log.time[0] + warm_up_s
    while start_time + last * period <= log.time[-1]:
        index = int(np.searchsorted(log.time, start_time))
        end_index = int(np.searchsorted(log.time, start_time + last * period, side="right"))
        if not np.all(valid[index:end_index]):
            start_time += stride_s
            continue
        t0 = log.time[index]
        sim.set_vehicle_state(
            log["position"][index],
            log["quaternion"][index],
            velocity[index],
            log["angular_velocity_body"][index] if log.has("angular_velocity_body") else (0.0, 0.0, 0.0),
            rotor_thrusts=log.sample("rotor_thrust_command", np.array([t0]))[0],
            settle=False,
            uav_index=uav_index,
        )
        physics_step = float(sim.scene.model.opt.timestep)
        warm_up_steps = int(round(warm_up_s / physics_step))
        sim.warm_start_rotors(log.sample("rotor_thrust_command", t0 + physics_step * np.arange(-warm_up_steps, 0)))
        sim_start = sim.time
        for step in range(1, last + 1):
            command = log.sample("rotor_thrust_command", np.array([t0 + (step - 1) * period]))[0]
            sim.step(command)
            for horizon in horizons:
                if steps_for[horizon] != step:
                    continue
                at = t0 + (sim.time - sim_start)
                truth = sim.truth(uav_index)
                logged_position = log.sample("position", np.array([at]))[0]
                logged_velocity = np.array([np.interp(at, log.time, velocity[:, axis]) for axis in range(3)])
                logged_rotation = quaternion_to_rotation(log.sample("quaternion", np.array([at])))[0]
                relative = logged_rotation.T @ truth["rotation_matrix"]
                angle = math.degrees(math.acos(max(-1.0, min(1.0, 0.5 * (np.trace(relative) - 1.0)))))
                errors[horizon]["position_m"].append(float(np.linalg.norm(truth["position"] - logged_position)))
                errors[horizon]["velocity_m_per_s"].append(float(np.linalg.norm(truth["velocity"] - logged_velocity)))
                errors[horizon]["attitude_deg"].append(angle)
        starts += 1
        start_time += stride_s
    if starts == 0:
        raise ValueError("prediction_error: the log is shorter than the longest horizon (or has no valid stretch)")
    return {
        "starts": starts,
        "horizons_s": {f"{h:g}": {name: _statistics(values) for name, values in errors[h].items()} for h in horizons},
    }
