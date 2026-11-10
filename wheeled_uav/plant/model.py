"""The plant model: what happens between a controller's command and the physics, and between the physics and the controller.

Applied every physics step around ``mj_step`` (``apply`` before, ``restore``
and ``record`` after), in every fidelity mode: it is part of the plant, like
the aerodynamics section. Without a ``plant`` section nothing here runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

from ..model.surface import evaluate_height_function
from ..types import SurfaceEvaluator, UAVModelSpec
from .actuation import BatteryPacks, RotorActuation
from .config import ActuationConfig, PlantConfig
from .correction import CorrectionModel
from .sensors import MeasuredState, SensorModel

__all__ = ["PlantModel", "VehicleKinematics", "plant_random_generator"]

# Stream ids mixed into the run's seed, so each source of randomness draws
# independently of the others and of the HIL network model.
_SENSOR_STREAM = 0x504C4E54  # "PLNT"


def plant_random_generator(seed: int, stream: int = _SENSOR_STREAM) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([int(seed), int(stream)]))


@dataclass(frozen=True)
class _Vehicle:
    body_id: int
    qpos_address: int
    dof_address: int
    actuator_ids: np.ndarray
    # Rotor positions in the body frame (the prop sites the thrust acts at).
    rotor_positions: np.ndarray


@dataclass(frozen=True)
class VehicleKinematics:
    """The current state of one vehicle, read straight from qpos/qvel (not from the one-step-old sensor readings)."""

    position: np.ndarray
    rotation: np.ndarray
    velocity_world: np.ndarray
    angular_velocity_body: np.ndarray


def _vehicle(model: mujoco.MjModel, spec: UAVModelSpec) -> _Vehicle:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, spec.body_name)
    if body_id < 0:
        raise ValueError(f"UAV body {spec.body_name!r} not found in the compiled model")
    joint_id = int(model.body_jntadr[body_id])
    if joint_id < 0 or model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError(f"the plant model needs a free joint on UAV body {spec.body_name!r}")
    actuator_ids = []
    rotor_positions = []
    for name in spec.actuator_names:
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if actuator_id < 0:
            raise ValueError(f"rotor actuator {name!r} not found in the compiled model")
        site_id = int(model.actuator_trnid[actuator_id, 0])
        if int(model.site_bodyid[site_id]) != body_id:
            raise ValueError(f"rotor site of {name!r} is not on the UAV body {spec.body_name!r}")
        actuator_ids.append(actuator_id)
        rotor_positions.append(np.array(model.site_pos[site_id], dtype=float))
    return _Vehicle(
        body_id=body_id,
        qpos_address=int(model.jnt_qposadr[joint_id]),
        dof_address=int(model.jnt_dofadr[joint_id]),
        actuator_ids=np.asarray(actuator_ids, dtype=int),
        rotor_positions=np.asarray(rotor_positions),
    )


class PlantModel:
    def __init__(
        self,
        model: mujoco.MjModel,
        params: dict[str, Any],
        config: PlantConfig,
        uav_specs: list[UAVModelSpec],
        sensor_layouts,
        *,
        seed: int = 0,
        surface_evaluator: SurfaceEvaluator | None = None,
    ):
        self.config = config
        self._model = model
        self._timestep = float(model.opt.timestep)
        self._vehicles = [_vehicle(model, spec) for spec in uav_specs]
        self._num_uavs = len(self._vehicles)
        self._rotors_per_uav = len(uav_specs[0].actuator_names)
        self._ctrl_index = np.concatenate([vehicle.actuator_ids for vehicle in self._vehicles])
        # MuJoCo clamps a limited actuator's control to its range: what the model reports as applied stays inside it.
        limited = np.asarray(model.actuator_ctrllimited[self._ctrl_index], dtype=bool)
        self._ctrl_limit = np.where(limited, model.actuator_ctrlrange[self._ctrl_index, 1], np.inf)
        aerodynamics = config.aerodynamics
        self._ground_effect = None if aerodynamics is None else aerodynamics.ground_effect
        self._surface_evaluator = surface_evaluator

        actuation_config = config.actuation
        if actuation_config is None and (config.battery is not None or self._ground_effect is not None):
            actuation_config = ActuationConfig()  # the battery and ground effect act on the rotor thrusts
        self.battery = None if config.battery is None else BatteryPacks(config.battery, self._num_uavs)
        self.actuation = None
        if actuation_config is not None:
            ctrl_limit = float(params["actuation"]["max_rotor_thrust"])
            self.actuation = RotorActuation(
                actuation_config,
                timestep=self._timestep,
                num_uavs=self._num_uavs,
                rotors_per_uav=self._rotors_per_uav,
                ctrl_limit=ctrl_limit,
                thrust_coefficient=float(params["actuation"].get("thrust_coefficient", 0.0)),
                voltage_compensation=True if config.battery is None else config.battery.voltage_compensation,
            )
        if self._ground_effect is not None:
            radius = self._ground_effect.rotor_radius_m
            if radius is None:
                radius = float(params.get("drone", {}).get("propeller", {}).get("radius", 0.0))
            if radius <= 0.0:
                raise ValueError("plant.aerodynamics.ground_effect needs rotor_radius_m (or drone.propeller.radius)")
            self._rotor_radius = radius
        self.correction = None if config.correction is None else CorrectionModel.from_file(config.correction.path)
        self.sensors = None
        if config.sensors is not None:
            self.sensors = SensorModel(
                config.sensors, timestep=self._timestep, sensor_layouts=list(sensor_layouts), rng=plant_random_generator(seed)
            )
        self._saved_ctrl: np.ndarray | None = None
        self._ground_gain = np.ones(self._num_uavs * self._rotors_per_uav)
        self._last_forces = np.zeros((self._num_uavs, 6))

    # -- what the runtime needs to know ---------------------------------------
    @property
    def applies_body_forces(self) -> bool:
        """True when the model adds to xfrc_applied (the runtime then rebuilds those rows every step)."""
        aerodynamics = self.config.aerodynamics
        has_drag = aerodynamics is not None and (
            any(aerodynamics.rotor_drag_coefficients_n_s_per_m) or any(aerodynamics.body_drag_area_m2)
        )
        return has_drag or self.correction is not None

    def kinematics(self, data: mujoco.MjData, uav_index: int) -> VehicleKinematics:
        vehicle = self._vehicles[uav_index]
        qpos = data.qpos[vehicle.qpos_address : vehicle.qpos_address + 7]
        rotation = np.zeros(9)
        mujoco.mju_quat2Mat(rotation, qpos[3:7])
        dof = vehicle.dof_address
        return VehicleKinematics(
            position=np.array(qpos[0:3], dtype=float),
            rotation=rotation.reshape(3, 3),
            velocity_world=np.array(data.qvel[dof : dof + 3], dtype=float),
            angular_velocity_body=np.array(data.qvel[dof + 3 : dof + 6], dtype=float),
        )

    def ground_height(self, x: float, y: float) -> float:
        """Height of the ground under (x, y): ground_effect.ground_height_m, else the surface height function, else 0."""
        if self._ground_effect is not None and self._ground_effect.ground_height_m is not None:
            return self._ground_effect.ground_height_m
        if self._surface_evaluator is not None:
            return float(evaluate_height_function(self._surface_evaluator, x, y))
        return 0.0

    def _update_ground_gain(self, data: mujoco.MjData) -> None:
        radius, cap = self._rotor_radius, self._ground_effect.max_gain
        gains = []
        for uav_index, vehicle in enumerate(self._vehicles):
            state = self.kinematics(data, uav_index)
            for offset in vehicle.rotor_positions:
                rotor = state.position + state.rotation @ offset
                height = rotor[2] - self.ground_height(rotor[0], rotor[1])
                ratio = radius / (4.0 * height) if height > 0.0 else np.inf
                gains.append(cap if ratio >= 1.0 else min(cap, 1.0 / (1.0 - ratio * ratio)))
        self._ground_gain = np.asarray(gains)

    # -- per physics step ------------------------------------------------
    def settle(self, data: mujoco.MjData) -> None:
        """Put the rotors in the steady state of the current command (a vehicle already flying)."""
        if self.actuation is not None:
            ratio = None if self.battery is None else self.battery.voltage_ratio()
            self.actuation.settle(data.ctrl[self._ctrl_index], ratio)

    def apply(self, data: mujoco.MjData) -> None:
        """Before mj_step: replace the commanded rotor thrusts with the delivered ones, add the plant's forces."""
        if self.actuation is not None:
            self._saved_ctrl = np.array(data.ctrl, copy=True)
            ratio = None if self.battery is None else self.battery.voltage_ratio()
            thrust = self.actuation.step(data.ctrl[self._ctrl_index], ratio)
            evaluated = None
            if self.battery is not None:
                thrust, evaluated = self._thrust_at_consistent_voltage(thrust)
            if self._ground_effect is not None:
                self._update_ground_gain(data)
            data.ctrl[self._ctrl_index] = self._applied_thrusts()
            if self.battery is not None:
                self.battery.update(thrust.reshape(self._num_uavs, self._rotors_per_uav), self._timestep, evaluated)
        if self.applies_body_forces:
            for uav_index, vehicle in enumerate(self._vehicles):
                wrench_body = self._body_wrench(data, uav_index)
                state = self.kinematics(data, uav_index)
                force_world = state.rotation @ wrench_body[:3]
                torque_world = state.rotation @ wrench_body[3:]
                data.xfrc_applied[vehicle.body_id, 0:3] += force_world
                data.xfrc_applied[vehicle.body_id, 3:6] += torque_world
                self._last_forces[uav_index] = wrench_body

    def _thrust_at_consistent_voltage(self, thrust: np.ndarray) -> tuple[np.ndarray, tuple]:
        """The step's thrust at the terminal voltage that this thrust itself produces, and the battery's evaluation of it.

        ``thrust`` was computed with the previous step's voltage. Where the
        thrust depends on the voltage (at the voltage-dependent limit, or
        without voltage compensation) that one-step lag can alternate between
        two states instead of settling, so there the two are solved together:
        a few substitutions when they converge, else bisection (more voltage
        allows more thrust, which lowers the voltage, so the consistent value
        is unique).
        """
        battery, actuation = self.battery, self.actuation
        shape = (self._num_uavs, self._rotors_per_uav)
        reference = battery.config.reference_voltage_v
        tolerance = 1.0e-6 * reference
        used = battery.voltage
        evaluated = battery.evaluate(thrust.reshape(shape))
        # The limit is lowest at the lower voltage: one check covers both.
        if not actuation.depends_on_voltage(np.minimum(used, evaluated[3]) / reference):
            return thrust, evaluated
        for _ in range(3):
            if np.all(np.abs(evaluated[3] - used) <= tolerance):
                return thrust, evaluated
            used = evaluated[3]
            thrust = actuation.deliver(used / reference)
            evaluated = battery.evaluate(thrust.reshape(shape))
        low, high = 0.5 * evaluated[1], evaluated[1].copy()
        for _ in range(30):
            middle = 0.5 * (low + high)
            implied = battery.terminal_voltage(actuation.deliver(middle / reference).reshape(shape))
            above = implied < middle  # the consistent voltage is below the trial value
            high = np.where(above, middle, high)
            low = np.where(above, low, middle)
        thrust = actuation.deliver(0.5 * (low + high) / reference)
        return thrust, battery.evaluate(thrust.reshape(shape))

    def restore(self, data: mujoco.MjData) -> None:
        """After mj_step: put the commanded thrusts back (the command persists between steps)."""
        if self._saved_ctrl is not None:
            data.ctrl[:] = self._saved_ctrl
            self._saved_ctrl = None

    def record(self, data: mujoco.MjData) -> None:
        """After mj_step: keep the step's sensor readings for delayed measurements."""
        if self.sensors is not None:
            self.sensors.record(data.time, data.sensordata)

    # -- forces -----------------------------------------------------------
    def airspeed_body(self, state: VehicleKinematics) -> np.ndarray:
        wind = np.zeros(3) if self.config.aerodynamics is None else np.asarray(self.config.aerodynamics.wind_m_per_s)
        return state.rotation.T @ (state.velocity_world - wind)

    def drag_force_body(self, airspeed_body: np.ndarray) -> np.ndarray:
        aerodynamics = self.config.aerodynamics
        if aerodynamics is None:
            return np.zeros(3)
        linear = np.asarray(aerodynamics.rotor_drag_coefficients_n_s_per_m) * airspeed_body
        quadratic = 0.5 * aerodynamics.air_density_kg_per_m3 * np.asarray(aerodynamics.body_drag_area_m2) * np.abs(airspeed_body) * airspeed_body
        return -(linear + quadratic)

    def _applied_thrusts(self) -> np.ndarray:
        """What the actuators receive: the delivered thrust times the ground-effect gain, within their range."""
        return np.minimum(self.actuation.delivered * self._ground_gain, self._ctrl_limit)

    def delivered_thrusts(self, uav_index: int) -> np.ndarray:
        """Thrust of each rotor of the vehicle as applied in the last step (ground effect included)."""
        if self.actuation is None:
            return np.full(self._rotors_per_uav, np.nan)
        rotors = slice(uav_index * self._rotors_per_uav, (uav_index + 1) * self._rotors_per_uav)
        return self._applied_thrusts()[rotors]

    def correction_features(self, data: mujoco.MjData, uav_index: int) -> np.ndarray:
        state = self.kinematics(data, uav_index)
        airspeed = self.airspeed_body(state)
        if self.actuation is not None:
            thrusts = self.delivered_thrusts(uav_index)
        else:
            vehicle = self._vehicles[uav_index]
            thrusts = np.array(data.ctrl[vehicle.actuator_ids], dtype=float)
        height = state.position[2] - self.ground_height(state.position[0], state.position[1])
        return np.concatenate([airspeed, state.angular_velocity_body, thrusts, [np.sum(thrusts), np.linalg.norm(airspeed), height]])

    def _body_wrench(self, data: mujoco.MjData, uav_index: int) -> np.ndarray:
        state = self.kinematics(data, uav_index)
        wrench = np.zeros(6)
        wrench[:3] = self.drag_force_body(self.airspeed_body(state))
        if self.correction is not None:
            wrench += self.config.correction.scale * self.correction.evaluate(self.correction_features(data, uav_index))
        return wrench

    # -- what the controller and the logs see -----------------------------
    def measure(self, data: mujoco.MjData) -> list[MeasuredState] | None:
        if self.sensors is None:
            return None
        return self.sensors.measure(data.time, data.sensordata)

    def requested_and_applied(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Commanded and applied thrust of every rotor (actuator log), or None without the rotor model."""
        if self.actuation is None:
            return None
        return self.actuation.requested.copy(), self._applied_thrusts()

    def snapshot(self) -> list[dict[str, Any]]:
        """Per-vehicle JSON fragment ``plant`` of the state packet."""
        entries: list[dict[str, Any]] = [{} for _ in self._vehicles]
        speeds = None if self.actuation is None else self.actuation.rotor_speeds()
        for uav_index, entry in enumerate(entries):
            if speeds is not None:
                rotors = slice(uav_index * self._rotors_per_uav, (uav_index + 1) * self._rotors_per_uav)
                entry["rotor_speeds_rad_s"] = speeds[rotors].tolist()
            if self.battery is not None:
                entry["battery_voltage_v"] = float(self.battery.voltage[uav_index])
                entry["battery_current_a"] = float(self.battery.current[uav_index])
                entry["battery_charge"] = float(self.battery.charge[uav_index])
            if self.applies_body_forces:
                entry["plant_force_body"] = self._last_forces[uav_index, :3].tolist()
                entry["plant_torque_body"] = self._last_forces[uav_index, 3:].tolist()
        return entries
