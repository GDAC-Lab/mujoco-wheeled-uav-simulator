"""What the controller measures: delayed, noisy, biased, sometimes stale states."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..types import SensorLayout
from .config import PlantSensorConfig

__all__ = ["MeasuredState", "SensorModel", "rotation_from_rotvec"]


def rotation_from_rotvec(rotvec: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(rotvec))
    if angle < 1.0e-12:
        return np.eye(3)
    x, y, z = rotvec / angle
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)


@dataclass(frozen=True)
class MeasuredState:
    position: np.ndarray
    velocity: np.ndarray
    angular_velocity_world: np.ndarray
    rotation_matrix: np.ndarray
    # Age of the position and attitude: the pose delay, plus the time a dropout has held them.
    measurement_age_s: float


class SensorModel:
    """Keeps a short history of the true sensor readings and draws measurements from it.

    ``record`` is called after every physics step with the step's sensor
    readings; ``measure`` is called once per published state. Delays are
    rounded to physics steps; at the start of a run, before a full delay has
    passed, the oldest reading stands in.
    """

    def __init__(self, config: PlantSensorConfig, *, timestep: float, sensor_layouts: list[SensorLayout], rng: np.random.Generator):
        self.config = config
        self._timestep = float(timestep)
        self._layouts = sensor_layouts
        self._rng = rng
        self._pose_lag = int(round(config.pose_delay_s / self._timestep))
        self._velocity_lag = int(round(config.velocity_delay_s / self._timestep))
        self._angular_lag = int(round(config.angular_velocity_delay_s / self._timestep))
        self._capacity = max(self._pose_lag, self._velocity_lag, self._angular_lag) + 1
        self._times = np.zeros(self._capacity)
        self._readings: np.ndarray | None = None
        self._count = 0
        self._head = -1
        num_uavs = len(sensor_layouts)
        bias_std = config.angular_velocity_bias_std_rad_per_s
        self.gyro_bias = rng.normal(0.0, bias_std, size=(num_uavs, 3)) if bias_std > 0.0 else np.zeros((num_uavs, 3))
        self._last_measure_time: float | None = None
        self._dropout_until = np.full(num_uavs, -math.inf)
        self._held: list[tuple[float, np.ndarray, np.ndarray, np.ndarray] | None] = [None] * num_uavs

    def record(self, time: float, sensordata: np.ndarray) -> None:
        if self._readings is None:
            self._readings = np.zeros((self._capacity, np.asarray(sensordata).size))
        self._head = (self._head + 1) % self._capacity
        self._times[self._head] = time
        self._readings[self._head] = sensordata
        self._count = min(self._count + 1, self._capacity)

    def _reading(self, lag: int, fallback: np.ndarray, fallback_time: float) -> tuple[float, np.ndarray]:
        if self._count == 0:
            return fallback_time, fallback
        lag = min(lag, self._count - 1)
        index = (self._head - lag) % self._capacity
        return float(self._times[index]), self._readings[index]

    def measure(self, time: float, sensordata: np.ndarray) -> list[MeasuredState]:
        """One measurement per vehicle at simulation time ``time`` (``sensordata``: the current readings)."""
        config = self.config
        rng = self._rng
        dt = 0.0 if self._last_measure_time is None else max(0.0, time - self._last_measure_time)
        self._last_measure_time = time
        walk = config.angular_velocity_bias_walk_rad_per_s_per_sqrt_s
        if walk > 0.0 and dt > 0.0:
            self.gyro_bias = self.gyro_bias + rng.normal(0.0, walk * math.sqrt(dt), size=self.gyro_bias.shape)
        pose_time, pose_reading = self._reading(self._pose_lag, sensordata, time)
        _, velocity_reading = self._reading(self._velocity_lag, sensordata, time)
        _, angular_reading = self._reading(self._angular_lag, sensordata, time)
        measurements = []
        for uav_index, layout in enumerate(self._layouts):
            held = self._held[uav_index]
            # Pose dropouts: the last delivered pose and velocity (noise and all) stand until the dropout ends.
            dropped = held is not None and config.pose_dropout_probability > 0.0 and (
                time < self._dropout_until[uav_index] or self._start_dropout(uav_index, time)
            )
            if dropped:
                sample_time, position, velocity, rotation = held
            else:
                sample_time = pose_time
                rotation = _rotation(pose_reading, layout)
                position = np.array(pose_reading[layout.position], dtype=float)
                velocity = np.array(velocity_reading[layout.linear_velocity], dtype=float)
                if config.position_noise_std_m > 0.0:
                    position = position + rng.normal(0.0, config.position_noise_std_m, size=3)
                if config.velocity_noise_std_m_per_s > 0.0:
                    velocity = velocity + rng.normal(0.0, config.velocity_noise_std_m_per_s, size=3)
                if config.attitude_noise_std_rad > 0.0:
                    rotation = rotation @ rotation_from_rotvec(rng.normal(0.0, config.attitude_noise_std_rad, size=3))
                self._held[uav_index] = (sample_time, position, velocity, rotation)
            gyro_rotation = _rotation(angular_reading, layout)
            angular_velocity_body = gyro_rotation.T @ np.asarray(angular_reading[layout.angular_velocity], dtype=float)
            angular_velocity_body = angular_velocity_body + self.gyro_bias[uav_index]
            if config.angular_velocity_noise_std_rad_per_s > 0.0:
                angular_velocity_body = angular_velocity_body + rng.normal(0.0, config.angular_velocity_noise_std_rad_per_s, size=3)
            measurements.append(
                MeasuredState(
                    position=position,
                    velocity=velocity,
                    # Expressed with the measured attitude, so that R_meas^T w = the measured body rate.
                    angular_velocity_world=rotation @ angular_velocity_body,
                    rotation_matrix=rotation,
                    measurement_age_s=max(0.0, time - sample_time),
                )
            )
        return measurements


    def _start_dropout(self, uav_index: int, time: float) -> bool:
        if self._rng.random() >= self.config.pose_dropout_probability:
            return False
        self._dropout_until[uav_index] = time + self._rng.exponential(self.config.pose_dropout_mean_duration_s)
        return True


def _rotation(reading: np.ndarray, layout: SensorLayout) -> np.ndarray:
    return np.column_stack((reading[layout.x_axis], reading[layout.y_axis], reading[layout.z_axis])).astype(float)
