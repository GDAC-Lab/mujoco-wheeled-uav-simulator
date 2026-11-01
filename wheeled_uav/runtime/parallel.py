"""Per-vehicle physics worlds stepped in parallel (opt-in).

With parallel physics each vehicle is simulated in a world of its own: the
scene's scenery (terrain, wall, floor) with that vehicle alone. All worlds
share one compiled single-vehicle model and advance concurrently on a thread
pool -- MuJoCo releases the GIL inside mj_step -- so the physics cost of a
control period is spread over the cores instead of piling up on one.

The scene's combined model and data stay the one place the rest of the
runtime works with. Before a batch of steps each vehicle's controls and
external wrenches are copied from the combined data into its world; after the
batch the worlds' joint states, sensor readings and time are copied back.
Contact reports are built from the worlds.

Vehicles in different worlds do not collide with one another. A warning is
printed the first time two of them come close enough that they could have.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

from ..model.builder import render_model_xml
from ..paths import DEFAULT_PATH_RESOLVER
from ..types import SensorLayout, SurfaceEvaluator, UAVModelSpec
from .contact import build_contact_attribution, build_contact_reports, build_geom_name_lookup
from .scene import SimulationScene, build_sensor_layout

__all__ = ["ParallelPhysics"]

_JOINT_QPOS_SIZE = {
    int(mujoco.mjtJoint.mjJNT_FREE): 7,
    int(mujoco.mjtJoint.mjJNT_BALL): 4,
    int(mujoco.mjtJoint.mjJNT_SLIDE): 1,
    int(mujoco.mjtJoint.mjJNT_HINGE): 1,
}
_JOINT_DOF_SIZE = {
    int(mujoco.mjtJoint.mjJNT_FREE): 6,
    int(mujoco.mjtJoint.mjJNT_BALL): 3,
    int(mujoco.mjtJoint.mjJNT_SLIDE): 1,
    int(mujoco.mjtJoint.mjJNT_HINGE): 1,
}


@dataclass(frozen=True)
class _VehicleIndices:
    """Where one vehicle's quantities live in a model (combined or world)."""

    qpos: np.ndarray
    qvel: np.ndarray
    ctrl: np.ndarray
    bodies: np.ndarray
    sensors: np.ndarray


def _subtree_bodies(model: mujoco.MjModel, root_body_id: int) -> list[int]:
    # Bodies are numbered parents-first, so one sweep collects the subtree.
    members = {root_body_id}
    for body_id in range(root_body_id + 1, model.nbody):
        if int(model.body_parentid[body_id]) in members:
            members.add(body_id)
    return sorted(members)


def _sensor_indices(layout: SensorLayout) -> np.ndarray:
    slices = (layout.position, layout.linear_velocity, layout.angular_velocity, layout.x_axis, layout.y_axis, layout.z_axis)
    return np.concatenate([np.arange(part.start, part.stop) for part in slices])


def _vehicle_indices(model: mujoco.MjModel, uav_spec: UAVModelSpec, layout: SensorLayout) -> _VehicleIndices:
    bodies = _subtree_bodies(model, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, uav_spec.body_name))
    qpos: list[int] = []
    qvel: list[int] = []
    for joint_id in range(model.njnt):
        if int(model.jnt_bodyid[joint_id]) not in bodies:
            continue
        joint_type = int(model.jnt_type[joint_id])
        qpos_address, dof_address = int(model.jnt_qposadr[joint_id]), int(model.jnt_dofadr[joint_id])
        qpos.extend(range(qpos_address, qpos_address + _JOINT_QPOS_SIZE[joint_type]))
        qvel.extend(range(dof_address, dof_address + _JOINT_DOF_SIZE[joint_type]))
    ctrl = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in uav_spec.actuator_names]
    return _VehicleIndices(
        qpos=np.asarray(qpos, dtype=int),
        qvel=np.asarray(qvel, dtype=int),
        ctrl=np.asarray(ctrl, dtype=int),
        bodies=np.asarray(bodies, dtype=int),
        sensors=_sensor_indices(layout),
    )


def _step_worlds(model: mujoco.MjModel, datas: list[mujoco.MjData], nstep: int) -> None:
    for data in datas:
        mujoco.mj_step(model, data, nstep=nstep)


class ParallelPhysics:
    """Steps one world per vehicle on a pool of threads."""

    def __init__(self, scene: SimulationScene, threads: int):
        if scene.model.na:
            raise ValueError("parallel physics does not support actuators with activation state")
        request = scene.request
        resolver = scene.path_resolver or DEFAULT_PATH_RESOLVER
        # The world is the scene's model with its first vehicle only; where that
        # vehicle spawns does not matter, load_state() moves it.
        world_xml_path, _, (world_spec,) = render_model_xml(
            scene.params,
            instance_id=request.instance_id,
            output_path=scene.model_xml_path.with_name(f"{scene.model_xml_path.stem}.vehicle_world.xml"),
            num_uavs=1,
            spawn_radius=request.spawn_radius,
            template_path=request.template_path,
            path_resolver=resolver,
            params_dir=resolver.get_params_path(request.params_path).parent,
        )
        self._model = mujoco.MjModel.from_xml_path(str(world_xml_path))
        self._world_spec = world_spec
        world_indices = _vehicle_indices(self._model, world_spec, build_sensor_layout(self._model, world_spec.sensor_names))
        self._combined = [
            _vehicle_indices(scene.model, uav_spec, layout)
            for uav_spec, layout in zip(scene.uav_specs, scene.sensor_layouts, strict=True)
        ]
        if any(indices.qpos.size != world_indices.qpos.size or indices.bodies.size != world_indices.bodies.size for indices in self._combined):
            raise ValueError("every vehicle of the scene must match the single-vehicle world")
        self._world = world_indices
        self._position_slices = [layout.position for layout in scene.sensor_layouts]
        self._names = [uav_spec.name for uav_spec in scene.uav_specs]
        self._attribution = build_contact_attribution(self._model, [world_spec])
        # Contact details carry the combined model's geom names (uav_3_left_wheel_geom, ...).
        world_geom_names = build_geom_name_lookup(self._model)
        world_bodies = set(int(body_id) for body_id in world_indices.bodies)
        vehicle_geoms = {
            geom_id for geom_id in range(self._model.ngeom) if int(self._model.geom_bodyid[geom_id]) in world_bodies
        }
        self._geom_names = [
            tuple(
                f"{uav_spec.contact_prefix}{name}" if geom_id in vehicle_geoms else name
                for geom_id, name in enumerate(world_geom_names)
            )
            for uav_spec in scene.uav_specs
        ]
        self._datas = [mujoco.MjData(self._model) for _ in scene.uav_specs]
        self.load_state(scene.data)
        self._separation_m = 2.0 * self._vehicle_radius()
        self._warned_proximity = False
        thread_count = max(1, min(int(threads), len(self._datas)))
        self._chunks = [self._datas[index::thread_count] for index in range(thread_count)]
        self._pool = ThreadPoolExecutor(max_workers=thread_count, thread_name_prefix="physics")

    @property
    def thread_count(self) -> int:
        return len(self._chunks)

    def _vehicle_radius(self) -> float:
        # Largest distance from the vehicle body origin to the far side of any of its colliding geoms.
        data = self._datas[0]
        root = int(self._world.bodies[0])
        vehicle_bodies = set(int(body_id) for body_id in self._world.bodies)
        radius = 0.0
        for geom_id in range(self._model.ngeom):
            if int(self._model.geom_bodyid[geom_id]) not in vehicle_bodies:
                continue
            if not (self._model.geom_contype[geom_id] or self._model.geom_conaffinity[geom_id]):
                continue
            offset = float(np.linalg.norm(data.geom_xpos[geom_id] - data.xpos[root]))
            radius = max(radius, offset + float(self._model.geom_rbound[geom_id]))
        return radius

    def load_state(self, data: mujoco.MjData) -> None:
        """Copy every vehicle's joint state and the time from the combined data into its world."""
        for world_data, combined in zip(self._datas, self._combined):
            world_data.qpos[self._world.qpos] = data.qpos[combined.qpos]
            world_data.qvel[self._world.qvel] = data.qvel[combined.qvel]
            world_data.time = data.time
            mujoco.mj_forward(self._model, world_data)

    def step(self, data: mujoco.MjData, nstep: int = 1) -> None:
        """Advance every world by `nstep` steps under the combined data's controls and wrenches."""
        world = self._world
        for world_data, combined in zip(self._datas, self._combined):
            world_data.ctrl[world.ctrl] = data.ctrl[combined.ctrl]
            world_data.xfrc_applied[world.bodies] = data.xfrc_applied[combined.bodies]
        for future in [self._pool.submit(_step_worlds, self._model, chunk, nstep) for chunk in self._chunks]:
            future.result()
        for world_data, combined in zip(self._datas, self._combined):
            data.qpos[combined.qpos] = world_data.qpos[world.qpos]
            data.qvel[combined.qvel] = world_data.qvel[world.qvel]
            data.sensordata[combined.sensors] = world_data.sensordata[world.sensors]
        data.time = self._datas[0].time
        self._check_separation(data)

    def contact_reports(self, surface_evaluator: SurfaceEvaluator | None, include_details: bool) -> list[dict[str, Any]]:
        return [
            build_contact_reports(
                self._model,
                world_data,
                surface_evaluator,
                [self._world_spec],
                geom_names=geom_names,
                include_details=include_details,
                attribution=self._attribution,
            )[0]
            for world_data, geom_names in zip(self._datas, self._geom_names)
        ]

    def _check_separation(self, data: mujoco.MjData) -> None:
        if self._warned_proximity or len(self._datas) < 2:
            return
        positions = np.array([data.sensordata[position] for position in self._position_slices])
        distances = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
        np.fill_diagonal(distances, np.inf)
        first, second = np.unravel_index(int(np.argmin(distances)), distances.shape)
        if distances[first, second] < self._separation_m:
            print(
                f"parallel physics: {self._names[first]} and {self._names[second]} are "
                f"{distances[first, second]:.2f} m apart, close enough to touch; vehicles in "
                "separate worlds do not collide with each other"
            )
            self._warned_proximity = True

    def close(self) -> None:
        self._pool.shutdown(wait=True)
