"""Simulation scene assembly: request -> rendered XML -> loaded MuJoCo scene."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import mujoco

from ..config import build_fidelity_config, load_vehicle_params
from ..model.builder import render_model_xml
from ..model.surface import get_surface_config
from ..paths import DEFAULT_PATH_RESOLVER, PathResolver
from ..protocol import UDP_IP, get_instance_ports
from ..timing import PacingMode, SimulationTimingConfig, parse_simulation_timing
from ..types import FidelityConfig, SensorLayout, SensorNames, SurfaceEvaluator, UAVModelSpec
from .contact import build_geom_name_lookup

__all__ = [
    "RenderedModelArtifacts",
    "SimulationRequest",
    "SimulationScene",
    "build_sensor_layout",
    "build_sensor_layouts",
    "load_simulation_scene",
    "render_simulation_model",
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


def render_simulation_model(
    request: SimulationRequest,
    *,
    path_resolver: PathResolver | None = None,
) -> RenderedModelArtifacts:
    resolver = path_resolver or DEFAULT_PATH_RESOLVER
    params = load_vehicle_params(params_path=request.params_path, path_resolver=resolver)
    model_xml_path, surface_evaluator, uav_specs = render_model_xml(
        params,
        instance_id=request.instance_id,
        num_uavs=request.num_uavs,
        spawn_radius=request.spawn_radius,
        template_path=request.template_path,
        generated_xml_dir=request.generated_xml_dir,
        path_resolver=resolver,
        params_dir=resolver.get_params_path(request.params_path).parent,
    )
    return RenderedModelArtifacts(
        params=params,
        model_xml_path=model_xml_path,
        surface_evaluator=surface_evaluator,
        uav_specs=uav_specs,
    )


def load_simulation_scene(
    request: SimulationRequest,
    *,
    path_resolver: PathResolver | None = None,
) -> SimulationScene:
    rendered_model = render_simulation_model(request, path_resolver=path_resolver)
    fidelity = build_fidelity_config(rendered_model.params)
    if request.fidelity_mode is not None and request.fidelity_mode != fidelity.mode:
        fidelity = replace(fidelity, mode=request.fidelity_mode)  # type: ignore[arg-type]
    model = mujoco.MjModel.from_xml_path(str(rendered_model.model_xml_path))
    data = mujoco.MjData(model)
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
    )


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
    model = mujoco.MjModel.from_xml_path(str(rendered_model.model_xml_path))
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
            },
            ensure_ascii=False,
        )
    )
    return 0
