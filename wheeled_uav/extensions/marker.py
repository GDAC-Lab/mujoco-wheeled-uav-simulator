"""An example extension: a marker that shows a commanded point, such as a setpoint or a waypoint.

The marker is a small sphere on a mocap body: it is drawn by the viewer and in recordings, and takes no
part in the physics (no contacts, nothing attached to it). A controller moves it by sending
``"extensions": {"<name>": {"position": [x, y, z]}}`` with its commands; the state packets report where
it is and how far each vehicle's body origin is from it.

    "extensions": {
      "goal": {"class": "wheeled_uav.extensions.marker:Marker", "position": [1.0, 0.0, 1.2], "max_speed": 0.5}
    }

Configuration: ``position`` [m] (start; default [0, 0, 1]), ``radius`` [m] (0.05), ``rgba``
([1, 0.5, 0, 0.6]), ``max_speed`` [m/s] (default null: the marker jumps to each commanded point;
a number: it moves there at that speed, one physics step at a time).

State: ``{"position": [x, y, z], "target": [x, y, z], "distance": [one per vehicle, m]}``.
"""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

from . import Extension, ExtensionContext, ExtensionError

__all__ = ["Marker"]


def _vector3(value: Any, what: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        raise ExtensionError(f"{what} must be 3 numbers, got {value!r}") from None
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ExtensionError(f"{what} must be 3 finite numbers, got {value!r}")
    return array


class Marker(Extension):
    keys = frozenset({"position", "radius", "rgba", "max_speed"})

    def __init__(self, name: str, config: dict[str, Any]):
        super().__init__(name, config)
        self.start = _vector3(self.config.get("position", [0.0, 0.0, 1.0]), f"extensions.{name}.position")
        self.radius = float(self.config.get("radius", 0.05))
        if not np.isfinite(self.radius) or self.radius <= 0.0:
            raise ExtensionError(f"extensions.{name}.radius must be positive")
        rgba = np.asarray(self.config.get("rgba", [1.0, 0.5, 0.0, 0.6]), dtype=float).reshape(-1)
        if rgba.shape != (4,):
            raise ExtensionError(f"extensions.{name}.rgba must be 4 numbers")
        self.rgba = rgba
        max_speed = self.config.get("max_speed")
        self.max_speed = None if max_speed is None else float(max_speed)
        if self.max_speed is not None and (not np.isfinite(self.max_speed) or self.max_speed <= 0.0):
            raise ExtensionError(f"extensions.{name}.max_speed must be positive (or null)")
        self.target = self.start.copy()
        self._vehicle_bodies: list[str] = []
        self._vehicle_ids: list[int] = []
        self._mocap = -1

    @property
    def body_name(self) -> str:
        return f"{self.name}_marker"

    def edit_spec(self, spec: mujoco.MjSpec, context: ExtensionContext) -> None:
        body = spec.worldbody.add_body(name=self.body_name, pos=self.start.tolist(), mocap=True)
        body.add_geom(
            name=f"{self.name}_marker_geom",
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[self.radius, 0.0, 0.0],
            rgba=self.rgba.tolist(),
            contype=0,
            conaffinity=0,
        )
        self._vehicle_bodies = [vehicle.body for vehicle in context.vehicles]

    def bind(self, model: mujoco.MjModel) -> None:
        body_id = model.body(self.body_name).id
        self._mocap = int(model.body_mocapid[body_id])
        self._vehicle_ids = [model.body(name).id for name in self._vehicle_bodies]

    def reset(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self.target = self.start.copy()
        data.mocap_pos[self._mocap] = self.start

    def apply_command(self, command: Any) -> None:
        if not isinstance(command, dict) or set(command) - {"position"}:
            raise ExtensionError(f'extension {self.name!r} takes {{"position": [x, y, z]}}, got {command!r}')
        if "position" in command:
            self.target = _vector3(command["position"], f"extension {self.name!r}: position")

    def before_step(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        position = data.mocap_pos[self._mocap]
        if self.max_speed is None:
            position[:] = self.target
            return
        offset = self.target - position
        distance = float(np.linalg.norm(offset))
        step = self.max_speed * float(model.opt.timestep)
        position[:] = self.target if distance <= step else position + offset * (step / distance)

    def state(self, model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, Any]:
        position = np.array(data.mocap_pos[self._mocap])
        return {
            "position": position,
            "target": self.target,
            "distance": [float(np.linalg.norm(data.xpos[body_id] - position)) for body_id in self._vehicle_ids],
        }
