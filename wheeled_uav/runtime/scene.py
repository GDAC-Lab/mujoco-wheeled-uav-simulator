"""Simulation scene assembly: request -> rendered XML -> loaded MuJoCo scene."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from ..calibration import apply_calibration_file
from ..config import build_fidelity_config, load_vehicle_params
from ..extensions import SceneExtensions, VehicleInfo, apply_spec_edits, load_extensions
from ..model.builder import render_model_xml
from ..plant import PlantConfig, apply_randomization, parse_plant_config, write_randomization_record
from ..model.surface import get_surface_config
from ..paths import DEFAULT_PATH_RESOLVER, PathResolver
from ..protocol import UDP_IP, get_instance_ports
from ..timing import PacingMode, SimulationTimingConfig, parse_simulation_timing
from ..types import FidelityConfig, SensorLayout, SensorNames, SurfaceEvaluator, UAVModelSpec
from .contact import ContactAttribution, build_contact_attribution, build_geom_name_lookup

__all__ = [
    "RenderedModelArtifacts",
    "SimulationRequest",
    "SimulationScene",
    "apply_initial_velocities",
    "build_sensor_layout",
    "build_sensor_layouts",
    "load_simulation_scene",
    "render_simulation_model",
    "resolve_physics_threads",
    "resolve_seed",
    "resolve_seed_from_params",
    "resolve_state_format",
    "resolve_timing_config",
    "validate_model",
]


@dataclass(frozen=True)
class SimulationRequest:
    """Everything needed to build and run one simulator process."""

    instance_id: int = 0
    recv_port: int | None = None
    send_port: int | None = None
    bind_ip: str = UDP_IP
    state_target_ip: str = UDP_IP
    num_uavs: int = 1
    spawn_radius: float = 1.5
    params_path: str | Path | None = None
    template_path: str | Path | None = None
    generated_xml_dir: str | Path | None = None
    # None -> use fidelity_mode from the params file (same tri-state convention
    # as pacing_mode and hold_until_first_command below).
    fidelity_mode: str | None = None
    pacing_mode: PacingMode | None = None
    headless: bool = False
    duration_seconds: float | None = None
    record_path: str | Path | None = None
    # None -> use simulation.hold_until_first_command from the params file.
    hold_until_first_command: bool | None = None
    # None -> use simulation.physics_threads from the params file.
    physics_threads: int | None = None
    # None -> use simulation.state_format from the params file.
    state_format: str | None = None
    # None -> use simulation.seed from the params file (0 when absent).
    seed: int | None = None

    def resolved_ports(self) -> tuple[int, int]:
        default_recv_port, default_send_port = get_instance_ports(self.instance_id)
        recv_port = default_recv_port if self.recv_port is None else self.recv_port
        send_port = default_send_port if self.send_port is None else self.send_port
        return recv_port, send_port


@dataclass(frozen=True)
class RenderedModelArtifacts:
    params: dict[str, Any]
    model_xml_path: Path
    surface_evaluator: SurfaceEvaluator | None
    uav_specs: list[UAVModelSpec]
    # The values the randomization section drew (empty without one), and the file they were written to.
    randomization_draws: tuple[dict[str, Any], ...] = ()
    randomization_record_path: Path | None = None
    # The extensions section, constructed (None: no extensions), and the spec they edited: the scene's
    # model is compiled from it (model_xml_path is then MuJoCo's XML of it, for viewing).
    extensions: SceneExtensions | None = None
    spec: mujoco.MjSpec | None = None


@dataclass(frozen=True)
class SimulationScene:
    request: SimulationRequest
    params: dict[str, Any]
    fidelity: FidelityConfig
    model_xml_path: Path
    model: mujoco.MjModel
    data: mujoco.MjData
    sensor_layouts: list[SensorLayout]
    surface_evaluator: SurfaceEvaluator | None
    uav_specs: list[UAVModelSpec]
    geom_names: tuple[str, ...]
    # Which UAV owns each geom; computed once so the per-packet contact
    # reports need only a single pass over the contacts.
    contact_attribution: ContactAttribution | None = None
    # The resolver the scene was rendered with (None: the default one), for
    # models rendered later from the same params (parallel physics).
    path_resolver: PathResolver | None = None
    # The plant section, parsed (None: no plant model).
    plant: PlantConfig | None = None
    # The values the randomization section drew for this run (already in params).
    randomization_draws: tuple[dict[str, Any], ...] = ()
    # The extensions section, bound to the model and reset (None: no extensions; wheeled_uav.extensions).
    extensions: SceneExtensions | None = None


def _get_sensor_slice(model: mujoco.MjModel, sensor_name: str) -> slice:
    sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
    sensor_address = model.sensor_adr[sensor_id]
    sensor_dimension = model.sensor_dim[sensor_id]
    return slice(sensor_address, sensor_address + sensor_dimension)


def build_sensor_layout(model: mujoco.MjModel, sensor_names: SensorNames) -> SensorLayout:
    return SensorLayout(
        position=_get_sensor_slice(model, sensor_names.position),
        linear_velocity=_get_sensor_slice(model, sensor_names.linear_velocity),
        angular_velocity=_get_sensor_slice(model, sensor_names.angular_velocity),
        x_axis=_get_sensor_slice(model, sensor_names.x_axis),
        y_axis=_get_sensor_slice(model, sensor_names.y_axis),
        z_axis=_get_sensor_slice(model, sensor_names.z_axis),
    )


def build_sensor_layouts(model: mujoco.MjModel, uav_specs: list[UAVModelSpec]) -> list[SensorLayout]:
    return [build_sensor_layout(model, uav_spec.sensor_names) for uav_spec in uav_specs]


def apply_initial_velocities(model: mujoco.MjModel, data: mujoco.MjData, uav_specs: list[UAVModelSpec]) -> None:
    """Write each vehicle's spawn velocity into ``data.qvel``.

    Call on freshly created data (qpos at the spawn pose), before mj_forward.
    """
    for uav_spec in uav_specs:
        velocity = uav_spec.initial_velocity
        if velocity is None:
            continue
        joint_id = int(model.body_jntadr[model.body(uav_spec.body_name).id])
        if joint_id < 0 or model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError(f"UAV body {uav_spec.body_name!r} has no free joint to carry its initial velocity")
        qpos_address = int(model.jnt_qposadr[joint_id])
        dof_address = int(model.jnt_dofadr[joint_id])
        rotation = np.zeros(9)
        mujoco.mju_quat2Mat(rotation, data.qpos[qpos_address + 3 : qpos_address + 7])
        # A free joint takes its linear velocity in world coordinates and its
        # angular velocity in body coordinates.
        data.qvel[dof_address : dof_address + 3] = velocity.linear
        data.qvel[dof_address + 3 : dof_address + 6] = rotation.reshape(3, 3).T @ np.asarray(velocity.angular)
        for joint_suffix, wheel_rate in (("lw", velocity.left_wheel_rate), ("rw", velocity.right_wheel_rate)):
            wheel_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{uav_spec.contact_prefix}joint_{joint_suffix}")
            if wheel_joint_id >= 0:
                data.qvel[int(model.jnt_dofadr[wheel_joint_id])] = wheel_rate


def render_simulation_model(
    request: SimulationRequest,
    *,
    path_resolver: PathResolver | None = None,
    params: dict[str, Any] | None = None,
) -> RenderedModelArtifacts:
    """Render the model XML for ``request``.

    ``params``, when given, is used instead of loading ``request.params_path``
    (a copy is taken; relative paths in it still resolve against the
    directory of ``request.params_path``, the repository default without one).
    """
    resolver = path_resolver or DEFAULT_PATH_RESOLVER
    if params is None:
        params = load_vehicle_params(params_path=request.params_path, path_resolver=resolver)
    else:
        params = apply_calibration_file(copy.deepcopy(params), resolver.get_params_path(request.params_path))
    # Draw the randomized parameters before anything reads them: mass, inertia
    # and geometry go into the XML, the plant parameters into the plant model.
    seed = resolve_seed_from_params(params, request.seed)
    params, draws = apply_randomization(params, seed=seed)
    params_dir = resolver.get_params_path(request.params_path).parent
    model_xml_path, surface_evaluator, uav_specs = render_model_xml(
        params,
        instance_id=request.instance_id,
        num_uavs=request.num_uavs,
        spawn_radius=request.spawn_radius,
        template_path=request.template_path,
        generated_xml_dir=request.generated_xml_dir,
        path_resolver=resolver,
        params_dir=params_dir,
    )
    record_path = write_randomization_record(model_xml_path, seed, draws) if draws else None
    # Extensions edit the rendered model (after randomization: their own settings may be drawn too).
    extensions = load_extensions(params, params_dir)
    extended_spec = None
    if extensions is not None:
        extended_xml_path = model_xml_path.with_name(f"{model_xml_path.stem}.extended.xml")
        extended_spec = apply_spec_edits(
            extensions,
            model_xml_path,
            vehicles=[VehicleInfo(index, uav.name, uav.body_name) for index, uav in enumerate(uav_specs)],
            params=params,
            params_dir=params_dir,
            seed=seed,
            output_path=extended_xml_path,
        )
        model_xml_path = extended_xml_path
    return RenderedModelArtifacts(
        params=params,
        model_xml_path=model_xml_path,
        surface_evaluator=surface_evaluator,
        uav_specs=uav_specs,
        randomization_draws=tuple(draws),
        randomization_record_path=record_path,
        extensions=extensions,
        spec=extended_spec,
    )


def _compile(rendered_model: RenderedModelArtifacts) -> mujoco.MjModel:
    if rendered_model.spec is not None:
        return rendered_model.spec.compile()
    return mujoco.MjModel.from_xml_path(str(rendered_model.model_xml_path))


def _check_plant_against_hil_effects(plant: PlantConfig | None, fidelity: FidelityConfig) -> None:
    """The plant model replaces the HIL-only actuator lag and sensor noise; both at once would apply twice."""
    if plant is None or fidelity.mode != "hil":
        return
    dynamics = fidelity.actuator_dynamics
    if plant.actuation is not None and (
        dynamics.motor_tau_ms > 0.0 or dynamics.thrust_rate_limit_n_per_s is not None or dynamics.omega_rate_limit_rad_per_s is not None
    ):
        raise ValueError(
            "plant.actuation and the HIL actuator_dynamics both model the rotors; "
            "set the motor time constants in plant.actuation and clear actuator_dynamics"
        )
    noise = fidelity.sensor_fidelity
    if plant.sensors is not None and any(
        value > 0.0
        for value in (
            noise.position_noise_std_m,
            noise.velocity_noise_std_m_per_s,
            noise.angular_velocity_noise_std_rad_per_s,
            noise.attitude_noise_std_rad,
        )
    ):
        raise ValueError(
            "plant.sensors and the HIL sensor_fidelity both add sensor noise; set the noise in plant.sensors and clear sensor_fidelity"
        )


def load_simulation_scene(
    request: SimulationRequest,
    *,
    path_resolver: PathResolver | None = None,
    params: dict[str, Any] | None = None,
) -> SimulationScene:
    """Build the scene for ``request`` (``params``: see render_simulation_model)."""
    rendered_model = render_simulation_model(request, path_resolver=path_resolver, params=params)
    fidelity = build_fidelity_config(rendered_model.params)
    if request.fidelity_mode is not None and request.fidelity_mode != fidelity.mode:
        fidelity = replace(fidelity, mode=request.fidelity_mode)  # type: ignore[arg-type]
    resolver = path_resolver or DEFAULT_PATH_RESOLVER
    plant = parse_plant_config(rendered_model.params, params_dir=resolver.get_params_path(request.params_path).parent)
    _check_plant_against_hil_effects(plant, fidelity)
    model = _compile(rendered_model)
    data = mujoco.MjData(model)
    apply_initial_velocities(model, data, rendered_model.uav_specs)
    extensions = rendered_model.extensions
    if extensions is not None:
        extensions.bind(model)
        extensions.reset(model, data)
    mujoco.mj_forward(model, data)
    return SimulationScene(
        request=request,
        params=rendered_model.params,
        fidelity=fidelity,
        model_xml_path=rendered_model.model_xml_path,
        model=model,
        data=data,
        sensor_layouts=build_sensor_layouts(model, rendered_model.uav_specs),
        surface_evaluator=rendered_model.surface_evaluator,
        uav_specs=rendered_model.uav_specs,
        geom_names=build_geom_name_lookup(model),
        contact_attribution=build_contact_attribution(model, rendered_model.uav_specs),
        path_resolver=path_resolver,
        plant=plant,
        randomization_draws=rendered_model.randomization_draws,
        extensions=extensions,
    )


def resolve_physics_threads(scene: SimulationScene) -> int:
    """0: one world holding every vehicle; n >= 1: a world per vehicle, stepped on n threads."""
    if scene.request.physics_threads is not None:
        threads = int(scene.request.physics_threads)
    else:
        threads = int(scene.params.get("simulation", {}).get("physics_threads", 0))
    if threads < 0:
        raise ValueError("simulation.physics_threads must be non-negative")
    return threads


def resolve_seed(scene: SimulationScene) -> int:
    """The run's random seed: --seed, else simulation.seed, else 0.

    Every random draw of a run (HIL network delays and losses, sensor noise,
    the plant model's sensor noise and dropouts) comes from generators seeded
    with it, so two runs with the same seed and inputs draw the same values.
    """
    return resolve_seed_from_params(scene.params, getattr(scene.request, "seed", None))


def resolve_seed_from_params(params: dict[str, Any], override: int | None = None) -> int:
    seed = override if override is not None else params.get("simulation", {}).get("seed", 0)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(f"simulation.seed must be a non-negative integer, got {seed!r}")
    return seed


STATE_FORMATS = ("json", "binary")


def resolve_state_format(scene: SimulationScene) -> str:
    """How state packets are encoded: "json" (default) or "binary" (see wheeled_uav.state_packets)."""
    state_format = scene.request.state_format
    if state_format is None:
        state_format = scene.params.get("simulation", {}).get("state_format", "json")
    state_format = str(state_format).strip().lower()
    if state_format not in STATE_FORMATS:
        raise ValueError(f"simulation.state_format must be one of {STATE_FORMATS}")
    return state_format


def resolve_timing_config(scene: SimulationScene) -> SimulationTimingConfig:
    timing_config = parse_simulation_timing(scene.params)
    if scene.request.pacing_mode is not None:
        timing_config = replace(timing_config, pacing_mode=scene.request.pacing_mode)
    return timing_config


def validate_model(
    request: SimulationRequest,
    *,
    path_resolver: PathResolver | None = None,
) -> int:
    """Render the XML, compile the MuJoCo model, and print a JSON summary."""
    rendered_model = render_simulation_model(request, path_resolver=path_resolver)
    model = _compile(rendered_model)
    effective_fidelity_mode = request.fidelity_mode or build_fidelity_config(rendered_model.params).mode

    print(
        json.dumps(
            {
                "instance_id": request.instance_id,
                "fidelity_mode": effective_fidelity_mode,
                "num_uavs": request.num_uavs,
                "xml_path": str(rendered_model.model_xml_path),
                "surface_type": get_surface_config(rendered_model.params["environment"])["type"],
                "surface_function": None if rendered_model.surface_evaluator is None else rendered_model.surface_evaluator.kind,
                "ngeom": int(model.ngeom),
                "nsensor": int(model.nsensor),
                "nu": int(model.nu),
                "plant": _plant_summary(rendered_model, path_resolver, request),
                "randomized_parameters": [draw["path"] for draw in rendered_model.randomization_draws],
                "randomization_record": None
                if rendered_model.randomization_record_path is None
                else str(rendered_model.randomization_record_path),
                "extensions": [] if rendered_model.extensions is None else list(rendered_model.extensions.names),
            },
            ensure_ascii=False,
        )
    )
    return 0


def _plant_summary(rendered_model: RenderedModelArtifacts, path_resolver: PathResolver | None, request: SimulationRequest) -> list[str]:
    """The plant subsections in effect (and a check that the section parses)."""
    resolver = path_resolver or DEFAULT_PATH_RESOLVER
    plant = parse_plant_config(rendered_model.params, params_dir=resolver.get_params_path(request.params_path).parent)
    if plant is None:
        return []
    return [name for name in ("actuation", "battery", "aerodynamics", "correction", "sensors") if getattr(plant, name) is not None]
