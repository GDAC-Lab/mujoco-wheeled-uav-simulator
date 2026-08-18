from __future__ import annotations

import math
from typing import Any

import mujoco
import numpy as np

from ..types import AerodynamicsConfig, SensorLayout, UAVModelSpec

__all__ = ["AerodynamicsModel"]


def _rotation_from_euler_xyz_degrees(euler_degrees: list[float]) -> np.ndarray:
    roll, pitch, yaw = (math.radians(float(angle)) for angle in euler_degrees)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rotation_x = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    rotation_y = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rotation_z = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return rotation_x @ rotation_y @ rotation_z


class AerodynamicsModel:
    """Config-gated aerodynamic effect forces applied via ``xfrc_applied``.

    Currently implements a near-wall interaction model: a force along the
    outward wall normal (the ``-x`` face of the environment wall box in the
    wall frame) whose magnitude is a polynomial in the wall-tangential speed,
    decays exponentially with the clearance between the vehicle body origin
    and the wall face, and is clipped to a configured maximum. Positive
    coefficients emulate a pressing deficit (wake recirculation pushing the
    vehicle off the wall); negative coefficients emulate suction.

    The model is part of the plant physics: it applies in every fidelity mode
    and is controlled purely by the ``aerodynamics`` section of
    ``vehicle_params.json`` (disabled by default, in which case ``apply()`` is
    a no-op and ``xfrc_applied`` is left entirely to other writers).
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        params: dict[str, Any],
        config: AerodynamicsConfig,
        uav_specs: list[UAVModelSpec],
        sensor_layouts: list[SensorLayout],
    ):
        self._config = config
        self._sensor_layouts = sensor_layouts
        self._body_ids = []
        for uav_spec in uav_specs:
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, uav_spec.body_name)
            if body_id < 0:
                # -1 would silently index the LAST body via xfrc_applied[-1].
                raise ValueError(f"UAV body {uav_spec.body_name!r} not found in the compiled model")
            self._body_ids.append(body_id)
        self._last_forces = [np.zeros(3, dtype=float) for _ in uav_specs]
        # 0.0, not NaN: these are published in the state packet before the
        # first physics step in lockstep mode, and NaN is not valid JSON for
        # MATLAB's jsondecode.
        self._last_clearances = [0.0 for _ in uav_specs]
        self._last_tangential_speeds = [0.0 for _ in uav_specs]

        environment = params.get("environment", {})
        wall_position = environment.get("wall_position")
        wall_size = environment.get("wall_size")
        self._wall_available = wall_position is not None and wall_size is not None
        if self._wall_available:
            wall_rotation = _rotation_from_euler_xyz_degrees(environment.get("wall_euler", [0.0, 0.0, 0.0]))
            # The working face is the -x face of the wall box in the wall frame.
            self._wall_normal = wall_rotation @ np.array([-1.0, 0.0, 0.0])
            self._wall_face_point = np.array(wall_position, dtype=float) + wall_rotation @ np.array(
                [-float(wall_size[0]), 0.0, 0.0]
            )

    @property
    def active(self) -> bool:
        wall_effect = self._config.wall_effect
        return (
            self._config.enabled
            and wall_effect.enabled
            and self._wall_available
            and wall_effect.max_force_n > 0.0
        )

    def _wall_effect_force(self, position: np.ndarray, velocity: np.ndarray) -> tuple[np.ndarray, float, float]:
        wall_effect = self._config.wall_effect
        clearance = float(self._wall_normal @ (position - self._wall_face_point))
        clearance = max(clearance, 0.0)
        tangential_velocity = velocity - (self._wall_normal @ velocity) * self._wall_normal
        tangential_speed = float(np.linalg.norm(tangential_velocity))
        magnitude = (
            wall_effect.coeff_const_n
            + wall_effect.coeff_linear_n_per_mps * tangential_speed
            + wall_effect.coeff_quadratic_n_per_mps2 * tangential_speed * tangential_speed
        )
        magnitude *= math.exp(-max(0.0, clearance - wall_effect.reference_clearance_m) / wall_effect.decay_length_m)
        magnitude = min(max(magnitude, -wall_effect.max_force_n), wall_effect.max_force_n)
        return magnitude * self._wall_normal, clearance, tangential_speed

    def apply(self, data: mujoco.MjData) -> None:
        # Adds the aero force on top of the command-owned body-wrench rows.
        # While this model is active, the simulation loop makes
        # ControlCommandDispatcher.refresh_body_wrenches rebuild those rows at
        # the start of every physics step (an overwrite here would silently
        # drop the body_wrench command feature, and += without that per-step
        # rebase would accumulate).
        if not self.active:
            return
        for uav_index, body_id in enumerate(self._body_ids):
            sensor_layout = self._sensor_layouts[uav_index]
            position = np.asarray(data.sensordata[sensor_layout.position])
            velocity = np.asarray(data.sensordata[sensor_layout.linear_velocity])
            force, clearance, tangential_speed = self._wall_effect_force(position, velocity)
            data.xfrc_applied[body_id, 0:3] += force
            self._last_forces[uav_index] = force
            self._last_clearances[uav_index] = clearance
            self._last_tangential_speeds[uav_index] = tangential_speed

    def snapshot(self) -> list[dict[str, object]] | None:
        """Per-UAV state-payload fragment, or None when the model is inactive."""
        if not self.active:
            return None
        return [
            {
                "wall_effect_force": self._last_forces[uav_index].tolist(),
                "wall_clearance_m": self._last_clearances[uav_index],
                "tangential_speed_mps": self._last_tangential_speeds[uav_index],
            }
            for uav_index in range(len(self._body_ids))
        ]
