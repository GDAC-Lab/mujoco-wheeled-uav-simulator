"""Run the simulator in-process: a Python controller steps it directly, without UDP or pacing.

The same scene, plant model and state fields as the UDP simulator (one
control period per ``step`` and the state packet as a dict), for scripted
studies that need no separate controller process: system identification
checks, batch evaluations over seeds, regression tests::

    from wheeled_uav.sim import Simulator

    with Simulator(overrides={"plant": {"actuation": {"motor_time_constant_up_s": 0.03}}}) as sim:
        state = sim.reset()
        while state["time"] < 2.0:
            state = sim.step([2.2, 2.2, 2.2, 2.2])

A step applies the command as a fresh command packet would, advances one
control period (``simulation.state_publish_every_n_steps`` physics steps),
and returns the state packet the UDP simulator would publish then (parsed
JSON; for several UAVs, the multi-UAV packet with its ``uavs`` list). The HIL
network model is not used (there is no network); the plant model is.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from .clock import wall_time_ns
from .config import load_vehicle_params, merge_params
from .runtime.loop import SimulationRuntime
from .runtime.scene import SimulationRequest, SimulationScene, load_simulation_scene, resolve_timing_config

__all__ = ["Simulator"]


class Simulator:
    def __init__(
        self,
        params_path: str | Path | None = None,
        *,
        params: dict[str, Any] | None = None,
        overrides: dict[str, Any] | None = None,
        num_uavs: int = 1,
        spawn_radius: float = 1.5,
        seed: int | None = None,
        fidelity_mode: str | None = None,
        generated_xml_dir: str | Path | None = None,
        instance_id: int = 0,
    ):
        """``params`` replaces loading ``params_path`` (whose directory still anchors relative paths);
        ``overrides`` is merged into the params (objects key by key)."""
        self._base_params = params if params is not None else load_vehicle_params(params_path)
        self._overrides = overrides
        self._temporary_dir = None
        if generated_xml_dir is None:
            self._temporary_dir = tempfile.TemporaryDirectory(prefix="wheeled_uav_sim_")
            generated_xml_dir = self._temporary_dir.name
        self._request = SimulationRequest(
            instance_id=instance_id,
            num_uavs=num_uavs,
            spawn_radius=spawn_radius,
            params_path=params_path,
            generated_xml_dir=generated_xml_dir,
            fidelity_mode=fidelity_mode,
            pacing_mode="lockstep",
            headless=True,
            state_format="json",
            seed=seed,
        )
        self._runtime: SimulationRuntime | None = None
        self._sequence = 0
        self._state: dict[str, Any] | None = None

    # -- life cycle ---------------------------------------------------------
    def reset(self, *, seed: int | None = None) -> dict[str, Any]:
        """(Re)build the scene, with new randomization and noise draws when ``seed`` is given; return the first state."""
        self.close_runtime()
        if seed is not None:
            self._request = replace(self._request, seed=int(seed))
        params = merge_params(self._base_params, self._overrides)
        scene = load_simulation_scene(self._request, params=params)
        self._runtime = SimulationRuntime(scene)
        self._sequence = 0
        self._state = self._build_state()
        return self._state

    def close_runtime(self) -> None:
        if self._runtime is not None:
            self._runtime.close()
            self._runtime = None

    def close(self) -> None:
        self.close_runtime()
        if self._temporary_dir is not None:
            self._temporary_dir.cleanup()
            self._temporary_dir = None

    def __enter__(self) -> Simulator:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- what a controller uses ---------------------------------------------
    @property
    def runtime(self) -> SimulationRuntime:
        if self._runtime is None:
            raise RuntimeError("call reset() first")
        return self._runtime

    @property
    def scene(self) -> SimulationScene:
        return self.runtime.scene

    @property
    def time(self) -> float:
        return float(self.scene.data.time)

    @property
    def steps_per_control_period(self) -> int:
        return resolve_timing_config(self.scene).state_publish_every_n_steps

    @property
    def control_period(self) -> float:
        return self.steps_per_control_period * float(self.scene.model.opt.timestep)

    @property
    def state(self) -> dict[str, Any]:
        """The state returned by the last reset() or step()."""
        if self._state is None:
            raise RuntimeError("call reset() first")
        return self._state

    def step(
        self,
        rotor_thrusts: Any,
        *,
        steps: int | None = None,
        body_wrenches: Any = None,
    ) -> dict[str, Any]:
        """Apply ``rotor_thrusts`` [N] (4 values, or one list of 4 per UAV) and advance; return the new state.

        ``steps`` physics steps (default: one control period). ``body_wrenches``:
        an optional world-frame [fx, fy, fz, tx, ty, tz] per UAV, as in a command packet.
        """
        runtime = self.runtime
        thrusts = np.asarray(rotor_thrusts, dtype=float)
        if thrusts.ndim == 1:
            thrusts = thrusts[None, :]
        wrenches = None
        if body_wrenches is not None:
            wrenches = np.asarray(body_wrenches, dtype=float)
            wrenches = (wrenches[None, :] if wrenches.ndim == 1 else wrenches).tolist()
        runtime.command_dispatcher.apply_command(runtime.scene.data, thrusts.tolist(), wrenches)
        count = self.steps_per_control_period if steps is None else int(steps)
        if count < 1:
            raise ValueError("steps must be at least 1")
        for _ in range(count):
            runtime.step_physics_once()
        self._state = self._build_state()
        return self._state

    def set_vehicle_state(
        self,
        position,
        quaternion,
        velocity=(0.0, 0.0, 0.0),
        angular_velocity_body=(0.0, 0.0, 0.0),
        *,
        rotor_thrusts=None,
        settle: bool = True,
        uav_index: int = 0,
    ) -> None:
        """Put a vehicle in a given state (wheels at rest relative to the body), e.g. a logged one.

        With ``rotor_thrusts`` the command is applied and (``settle``) the
        plant's rotors settle in its steady state, as for a vehicle that has
        been flying with it; see also warm_start_rotors.
        """
        scene = self.scene
        model, data = scene.model, scene.data
        spec = scene.uav_specs[uav_index]
        body_id = model.body(spec.body_name).id
        joint_id = int(model.body_jntadr[body_id])
        qpos_address, dof_address = int(model.jnt_qposadr[joint_id]), int(model.jnt_dofadr[joint_id])
        quaternion = np.asarray(quaternion, dtype=float)
        data.qpos[qpos_address : qpos_address + 3] = position
        data.qpos[qpos_address + 3 : qpos_address + 7] = quaternion / np.linalg.norm(quaternion)
        data.qvel[dof_address : dof_address + 3] = velocity
        data.qvel[dof_address + 3 : dof_address + 6] = angular_velocity_body
        for suffix in ("lw", "rw"):
            wheel = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{spec.contact_prefix}joint_{suffix}")
            if wheel >= 0:
                data.qvel[int(model.jnt_dofadr[wheel])] = 0.0
        mujoco.mj_forward(model, data)
        runtime = self.runtime
        if runtime.parallel_physics is not None:
            runtime.parallel_physics.load_state(data)
        if rotor_thrusts is not None:
            thrusts = np.asarray(rotor_thrusts, dtype=float)
            commands = np.array(data.ctrl, dtype=float).reshape(len(scene.uav_specs), -1)
            commands[uav_index] = thrusts
            runtime.command_dispatcher.apply_command(data, commands.tolist())
            if settle and runtime.plant is not None:
                runtime.plant.settle(data)

    def warm_start_rotors(self, command_history) -> None:
        """Run the plant's rotor model (delay line and lag) through past commands, one row per physics step.

        Rows hold the commands of every rotor of the scene (or the four of a
        single UAV). The rotors then are where that history leaves them,
        instead of in a steady state; the vehicles and the battery are not
        moved.
        """
        plant = self.runtime.plant
        if plant is None or plant.actuation is None:
            return
        history = np.asarray(command_history, dtype=float)
        history = history.reshape(history.shape[0], -1)
        ratio = None if plant.battery is None else plant.battery.voltage_ratio()
        plant.actuation.settle(history[0], ratio)
        for row in history[1:]:
            plant.actuation.step(row, ratio)

    def truth(self, uav_index: int = 0) -> dict[str, np.ndarray]:
        """The true state of a vehicle now (from the joint state, not the sensors)."""
        scene = self.scene
        data, model = scene.data, scene.model
        body_id = model.body(scene.uav_specs[uav_index].body_name).id
        joint_id = int(model.body_jntadr[body_id])
        qpos_address, dof_address = int(model.jnt_qposadr[joint_id]), int(model.jnt_dofadr[joint_id])
        quaternion = np.array(data.qpos[qpos_address + 3 : qpos_address + 7])
        rotation = np.zeros(9)
        mujoco.mju_quat2Mat(rotation, quaternion)
        return {
            "time": np.array(data.time),
            "position": np.array(data.qpos[qpos_address : qpos_address + 3]),
            "quaternion": quaternion,
            "rotation_matrix": rotation.reshape(3, 3),
            "velocity": np.array(data.qvel[dof_address : dof_address + 3]),
            "angular_velocity_body": np.array(data.qvel[dof_address + 3 : dof_address + 6]),
        }

    def _build_state(self) -> dict[str, Any]:
        runtime = self.runtime
        self._sequence += 1
        contact_reports = None
        if runtime.parallel_physics is not None:
            contact_reports = runtime.parallel_physics.contact_reports(
                runtime.scene.surface_evaluator, runtime.state_publisher.include_contact_details
            )
        payload = runtime.state_publisher.build_payload(
            1.0,
            self._sequence,
            wall_time_ns(),
            runtime.actuator_snapshot(),
            aero_snapshot=runtime.aerodynamics_model.snapshot(),
            contact_reports=contact_reports,
        )
        return json.loads(payload)
