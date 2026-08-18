"""HIL plant-fidelity effects: sensor noise and actuator dynamics.

Both effects are no-ops in ``baseline`` mode; in ``hil`` mode they are driven
by the ``sensor_fidelity`` and ``actuator_dynamics`` sections of
``vehicle_params.json``.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from ..types import FidelityConfig

__all__ = [
    "ActuatorModel",
    "ActuatorSnapshot",
    "apply_actuator_dynamics_step",
    "apply_sensor_fidelity",
]


@dataclass(frozen=True)
class ActuatorSnapshot:
    requested_ctrl: np.ndarray
    applied_ctrl: np.ndarray


def _rotation_matrix_from_rotvec(rotvec: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(rotvec))
    if angle < 1.0e-12:
        return np.eye(3, dtype=float)
    axis = rotvec / angle
    x_axis, y_axis, z_axis = axis
    skew = np.array(
        [
            [0.0, -z_axis, y_axis],
            [z_axis, 0.0, -x_axis],
            [-y_axis, x_axis, 0.0],
        ],
        dtype=float,
    )
    return np.eye(3, dtype=float) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)


def apply_sensor_fidelity(
    position: np.ndarray,
    linear_velocity: np.ndarray,
    angular_velocity_world: np.ndarray,
    rotation_matrix: np.ndarray,
    fidelity: FidelityConfig,
    rng: np.random.Generator | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if fidelity.mode != "hil":
        return position, linear_velocity, angular_velocity_world, rotation_matrix

    sensor_fidelity = fidelity.sensor_fidelity
    measured_position = np.array(position, copy=True)
    measured_linear_velocity = np.array(linear_velocity, copy=True)
    measured_angular_velocity_world = np.array(angular_velocity_world, copy=True)
    measured_rotation_matrix = np.array(rotation_matrix, copy=True)
    if rng is None:
        return measured_position, measured_linear_velocity, measured_angular_velocity_world, measured_rotation_matrix

    if sensor_fidelity.position_noise_std_m > 0.0:
        measured_position += rng.normal(0.0, sensor_fidelity.position_noise_std_m, size=3)
    if sensor_fidelity.velocity_noise_std_m_per_s > 0.0:
        measured_linear_velocity += rng.normal(0.0, sensor_fidelity.velocity_noise_std_m_per_s, size=3)
    if sensor_fidelity.angular_velocity_noise_std_rad_per_s > 0.0:
        measured_angular_velocity_world += rng.normal(0.0, sensor_fidelity.angular_velocity_noise_std_rad_per_s, size=3)
    if sensor_fidelity.attitude_noise_std_rad > 0.0:
        noise_rotvec = rng.normal(0.0, sensor_fidelity.attitude_noise_std_rad, size=3)
        measured_rotation_matrix = measured_rotation_matrix @ _rotation_matrix_from_rotvec(noise_rotvec)
    return measured_position, measured_linear_velocity, measured_angular_velocity_world, measured_rotation_matrix


def apply_actuator_dynamics_step(
    requested_ctrl: np.ndarray,
    applied_ctrl: np.ndarray,
    timestep: float,
    fidelity: FidelityConfig,
    thrust_coefficient: float,
) -> np.ndarray:
    if fidelity.mode != "hil":
        return np.array(requested_ctrl, copy=True)

    actuator_dynamics = fidelity.actuator_dynamics
    updated_ctrl = np.array(requested_ctrl, copy=True)
    if actuator_dynamics.motor_tau_ms > 0.0:
        tau_seconds = actuator_dynamics.motor_tau_ms / 1.0e3
        alpha = min(1.0, timestep / (tau_seconds + timestep))
        updated_ctrl = applied_ctrl + alpha * (updated_ctrl - applied_ctrl)

    if actuator_dynamics.omega_rate_limit_rad_per_s is not None:
        if thrust_coefficient <= 0.0:
            raise ValueError("actuation.thrust_coefficient must be positive when omega_rate_limit_rad_per_s is set")
        requested_omega = np.sqrt(np.maximum(0.0, updated_ctrl) / thrust_coefficient)
        applied_omega = np.sqrt(np.maximum(0.0, applied_ctrl) / thrust_coefficient)
        omega_step_limit = actuator_dynamics.omega_rate_limit_rad_per_s * timestep
        clipped_omega = applied_omega + np.clip(requested_omega - applied_omega, -omega_step_limit, omega_step_limit)
        updated_ctrl = thrust_coefficient * clipped_omega * clipped_omega

    if actuator_dynamics.thrust_rate_limit_n_per_s is not None:
        thrust_step_limit = actuator_dynamics.thrust_rate_limit_n_per_s * timestep
        updated_ctrl = applied_ctrl + np.clip(updated_ctrl - applied_ctrl, -thrust_step_limit, thrust_step_limit)

    return np.maximum(0.0, updated_ctrl)


class ActuatorModel:
    """Applies actuator dynamics between the commanded and effective ctrl."""

    def __init__(self, model: mujoco.MjModel, fidelity: FidelityConfig, thrust_coefficient: float):
        self._fidelity = fidelity
        self._timestep = float(model.opt.timestep)
        self._thrust_coefficient = float(thrust_coefficient)
        self._applied_ctrl = np.zeros(model.nu, dtype=float)
        self._requested_ctrl = np.zeros(model.nu, dtype=float)
        self._initialized = False

    def apply(self, data: mujoco.MjData) -> None:
        self._requested_ctrl = np.array(data.ctrl, copy=True)
        if not self._initialized:
            self._applied_ctrl = np.array(data.ctrl, copy=True)
            self._initialized = True
        self._applied_ctrl = apply_actuator_dynamics_step(
            self._requested_ctrl,
            self._applied_ctrl,
            self._timestep,
            self._fidelity,
            self._thrust_coefficient,
        )
        data.ctrl[:] = self._applied_ctrl

    def snapshot(self) -> ActuatorSnapshot:
        return ActuatorSnapshot(
            requested_ctrl=np.array(self._requested_ctrl, copy=True),
            applied_ctrl=np.array(self._applied_ctrl, copy=True),
        )
