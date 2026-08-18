from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from ..paths import DEFAULT_PATH_RESOLVER, PathResolver
from ..types import InitialPoseSpec, RotorSpec, SensorNames, SurfaceEvaluator, UAVModelSpec
from .poses import build_initial_poses, build_surface_model_spec
from .xml_format import format_scalar, format_vector

__all__ = [
    "build_allocation_matrix",
    "build_sensor_names",
    "build_rotor_specs",
    "build_uav_model_specs",
    "build_sensor_block",
    "build_xml_replacements",
    "render_model_xml",
]


def build_allocation_matrix(rotor_specs: list[RotorSpec]) -> np.ndarray:
    # Maps rotor thrusts to the body wrench [f_z; M_x; M_y; M_z]:
    #   column_i = [axis_i.z; p_i x axis_i + spin_i * kappa_i * axis_i]
    # derived from each rotor's actual geometry, so any rotor ordering,
    # placement, or tilt in the JSON stays consistent with the model actuators.
    columns = []
    for rotor_spec in rotor_specs:
        axis = np.asarray(rotor_spec.thrust_axis, dtype=float)
        position = np.asarray(rotor_spec.position, dtype=float)
        moment = np.cross(position, axis) + rotor_spec.spin_sign * rotor_spec.yaw_moment_ratio * axis
        columns.append(np.concatenate(([axis[2]], moment)))
    return np.column_stack(columns)


def _normalize_vector(values: list[float], field_name: str) -> tuple[float, float, float]:
    vector = np.asarray([float(value) for value in values], dtype=float)
    vector_norm = float(np.linalg.norm(vector))
    if vector.shape != (3,):
        raise ValueError(f"{field_name} must have exactly 3 elements")
    if vector_norm <= 1.0e-9:
        raise ValueError(f"{field_name} must not be the zero vector")
    normalized_vector = vector / vector_norm
    return float(normalized_vector[0]), float(normalized_vector[1]), float(normalized_vector[2])


def _parse_rotor_spec(rotor_config: dict[str, Any], rotor_index: int, default_yaw_moment_ratio: float) -> RotorSpec:
    rotor_suffix = str(rotor_config.get("name", "")).strip().lower()
    if not rotor_suffix:
        raise ValueError(f"actuation.rotors[{rotor_index}] must define a non-empty name")

    raw_position = rotor_config.get("position_body")
    if not isinstance(raw_position, list):
        raise ValueError(f"actuation.rotors[{rotor_index}].position_body must be a 3-element list")

    raw_thrust_axis = rotor_config.get("thrust_axis_body")
    if not isinstance(raw_thrust_axis, list):
        raise ValueError(f"actuation.rotors[{rotor_index}].thrust_axis_body must be a 3-element list")

    position = tuple(float(value) for value in raw_position)
    if len(position) != 3:
        raise ValueError(f"actuation.rotors[{rotor_index}].position_body must have exactly 3 elements")

    thrust_axis = _normalize_vector(raw_thrust_axis, f"actuation.rotors[{rotor_index}].thrust_axis_body")
    yaw_moment_ratio = float(rotor_config.get("yaw_moment_ratio", default_yaw_moment_ratio))
    spin_sign = float(rotor_config.get("spin_sign", 1.0))
    if abs(spin_sign) <= 1.0e-9:
        raise ValueError(f"actuation.rotors[{rotor_index}].spin_sign must be non-zero")

    return RotorSpec(
        suffix=rotor_suffix,
        position=position,
        thrust_axis=thrust_axis,
        yaw_moment_ratio=yaw_moment_ratio,
        spin_sign=1.0 if spin_sign > 0.0 else -1.0,
    )


def build_rotor_specs(params: dict[str, Any]) -> list[RotorSpec]:
    actuation = params["actuation"]
    default_yaw_moment_ratio = float(actuation["yaw_moment_ratio"])
    rotor_configs = actuation.get("rotors")
    if isinstance(rotor_configs, list) and len(rotor_configs) > 0:
        rotor_specs = [_parse_rotor_spec(rotor_config, rotor_index, default_yaw_moment_ratio) for rotor_index, rotor_config in enumerate(rotor_configs)]
        if len(rotor_specs) != 4:
            raise ValueError("actuation.rotors must define exactly 4 rotors to preserve the current controller interface")
        rotor_suffixes = [rotor_spec.suffix for rotor_spec in rotor_specs]
        if len(set(rotor_suffixes)) != len(rotor_suffixes):
            raise ValueError("actuation.rotors names must be unique")
        return rotor_specs

    drone = params["drone"]
    arm_x = float(drone["arm"]["x"])
    arm_y = float(drone["arm"]["y"])
    propeller_z = float(drone.get("propeller", {}).get("z", 0.0))
    # spin_sign は PX4 の Quad-X 機体規約（CA_ROTOR*_KM の符号）に合わせる。
    # PX4 は FRD(z下)、本モデルは NWU(z上) なので符号は見た目が反対になるのが正しい。
    #   PX4(FRD): FR=+ BL=+ FL=- BR=-   ->   本モデル(z上): FR=- BL=- FL=+ BR=+
    # この符号を反転させると閉ループでヨーが正帰還となり発散する。変更しないこと。
    return [
        RotorSpec("fr", (arm_x, -arm_y, propeller_z), (0.0, 0.0, 1.0), default_yaw_moment_ratio, -1.0),
        RotorSpec("fl", (arm_x, arm_y, propeller_z), (0.0, 0.0, 1.0), default_yaw_moment_ratio, 1.0),
        RotorSpec("br", (-arm_x, -arm_y, propeller_z), (0.0, 0.0, 1.0), default_yaw_moment_ratio, 1.0),
        RotorSpec("bl", (-arm_x, arm_y, propeller_z), (0.0, 0.0, 1.0), default_yaw_moment_ratio, -1.0),
    ]


def build_sensor_names(params: dict[str, Any], body_name: str | None = None, sensor_prefix: str | None = None) -> SensorNames:
    sensor_items = params["sensors"]["items"]
    if body_name is None and sensor_prefix is None:
        return SensorNames(
            position=sensor_items["position"]["name"],
            linear_velocity=sensor_items["linear_velocity"]["name"],
            angular_velocity=sensor_items["angular_velocity"]["name"],
            x_axis=sensor_items["x_axis"]["name"],
            y_axis=sensor_items["y_axis"]["name"],
            z_axis=sensor_items["z_axis"]["name"],
        )

    prefix = sensor_prefix or body_name or "uav"
    return SensorNames(
        position=f"{prefix}_position",
        linear_velocity=f"{prefix}_linear_velocity",
        angular_velocity=f"{prefix}_angular_velocity",
        x_axis=f"{prefix}_x_axis",
        y_axis=f"{prefix}_y_axis",
        z_axis=f"{prefix}_z_axis",
    )


def build_uav_model_specs(params: dict[str, Any], num_uavs: int) -> list[UAVModelSpec]:
    base_name = str(params["drone"]["name"])
    rotor_suffixes = tuple(rotor_spec.suffix for rotor_spec in build_rotor_specs(params))
    specs: list[UAVModelSpec] = []
    for uav_index in range(num_uavs):
        if num_uavs == 1:
            body_name = base_name
            sensor_prefix = base_name
            contact_prefix = ""
        else:
            body_name = f"uav_{uav_index + 1}"
            sensor_prefix = body_name
            contact_prefix = f"{body_name}_"
        specs.append(
            UAVModelSpec(
                name=body_name,
                body_name=body_name,
                actuator_names=tuple(f"{body_name}_thrust_{suffix}" for suffix in rotor_suffixes),
                sensor_names=build_sensor_names(params, body_name=body_name, sensor_prefix=sensor_prefix),
                contact_prefix=contact_prefix,
            )
        )
    return specs


def _build_rotor_site_lines(
    rotor_specs: list[RotorSpec],
    prefix: str,
    propeller_radius: float,
    propeller_thickness: float,
    prop_rgba: list[float],
) -> list[str]:
    return [
        f'      <site name="{prefix}prop_{rotor_spec.suffix}" pos="{format_vector(list(rotor_spec.position))}" zaxis="{format_vector(list(rotor_spec.thrust_axis))}" type="cylinder" size="{format_vector([propeller_radius, propeller_thickness])}" rgba="{format_vector(prop_rgba)}"/>'
        for rotor_spec in rotor_specs
    ]


def _build_wheel_body_block(drone: dict[str, Any], prefix: str, side_name: str, wheel_offset_y: float) -> list[str]:
    wheel_direction = -1.0 if side_name == "right" else 1.0
    wheel_body_name = f"{prefix}{side_name}_wheel"
    wheel_joint_name = f"{prefix}joint_{'rw' if side_name == 'right' else 'lw'}"
    wheel_geom_name = f"{prefix}{side_name}_wheel_geom"
    wheels = drone["wheels"]
    wheel_friction = format_vector(wheels["friction"])
    optional_attributes = ""
    if wheels.get("solimp") is not None:
        optional_attributes += f' solimp="{format_vector(wheels["solimp"])}"'
    if wheels.get("condim") is not None:
        optional_attributes += f' condim="{format_scalar(int(wheels["condim"]))}"'
    return [
        f'      <body name="{wheel_body_name}" pos="{format_vector([0.0, wheel_direction * wheel_offset_y, 0.0])}">',
        f'        <joint name="{wheel_joint_name}" type="hinge" axis="0 1 0" damping="{format_scalar(wheels["joint_damping"])}"/>',
        f'        <geom name="{wheel_geom_name}" type="cylinder" size="{format_vector([wheels["radius"], wheels["thickness"]])}" euler="90 0 0" mass="{format_scalar(wheels["mass"])}" rgba="{format_vector(wheels["rgba"])}" solref="{format_vector(wheels["solref"])}"{optional_attributes} friction="{wheel_friction}" contype="{format_scalar(wheels["contact"]["contype"])}" conaffinity="{format_scalar(wheels["contact"]["conaffinity"])}"/>',
        "      </body>",
    ]


def _get_inertial_reference(drone: dict[str, Any]) -> str:
    inertial_reference = str(drone.get("inertial_reference", "body_only")).strip().lower()
    if inertial_reference not in {"body_only", "total_vehicle"}:
        raise ValueError('drone.inertial_reference must be "body_only" or "total_vehicle"')
    return inertial_reference


def _wheel_pair_inertia_contribution(drone: dict[str, Any]) -> tuple[float, tuple[float, float, float]]:
    # Both wheels modeled as solid cylinders (axis along body y) offset +/-offset_y from the COM.
    wheels = drone["wheels"]
    wheel_mass = float(wheels["mass"])
    radius = float(wheels["radius"])
    full_height = 2.0 * float(wheels["thickness"])
    offset_y = float(wheels["offset_y"])

    inertia_about_axis = 0.5 * wheel_mass * radius * radius
    inertia_perpendicular = wheel_mass * (3.0 * radius * radius + full_height * full_height) / 12.0
    parallel_axis_term = wheel_mass * offset_y * offset_y

    pair_ixx = 2.0 * (inertia_perpendicular + parallel_axis_term)
    pair_iyy = 2.0 * inertia_about_axis
    pair_izz = pair_ixx
    return 2.0 * wheel_mass, (pair_ixx, pair_iyy, pair_izz)


def _get_drone_mass(drone: dict[str, Any]) -> float:
    # Mass assigned to the central body <inertial>. With inertial_reference
    # "total_vehicle", drone.mass is the whole-vehicle mass and the wheel masses
    # are subtracted here so the assembled MuJoCo model matches drone.mass exactly.
    total_or_body_mass = float(drone.get("mass", drone["body_box"]["mass"]))
    if _get_inertial_reference(drone) == "body_only":
        return total_or_body_mass

    wheel_pair_mass, _ = _wheel_pair_inertia_contribution(drone)
    body_mass = total_or_body_mass - wheel_pair_mass
    if body_mass <= 0.0:
        raise ValueError("drone.mass (total_vehicle) must exceed the combined wheel mass")
    return body_mass


def _get_raw_diaginertia(drone: dict[str, Any]) -> tuple[float, float, float]:
    raw_inertia = drone.get("inertia")
    if raw_inertia is not None:
        inertia_matrix = np.asarray(raw_inertia, dtype=float)
        if inertia_matrix.shape == (3, 3):
            if not np.allclose(inertia_matrix, np.diag(np.diag(inertia_matrix)), atol=1.0e-12):
                raise ValueError("drone.inertia must be diagonal for MuJoCo inertial output")
            return tuple(float(value) for value in np.diag(inertia_matrix))
        if inertia_matrix.shape == (3,):
            return tuple(float(value) for value in inertia_matrix)
        raise ValueError("drone.inertia must be a 3-element vector or 3x3 matrix")

    mass = float(drone.get("mass", drone["body_box"]["mass"]))
    box_size = np.asarray(drone["body_box"]["size"], dtype=float)
    ixx = mass * (box_size[1] ** 2 + box_size[2] ** 2) / 3.0
    iyy = mass * (box_size[0] ** 2 + box_size[2] ** 2) / 3.0
    izz = mass * (box_size[0] ** 2 + box_size[1] ** 2) / 3.0
    return float(ixx), float(iyy), float(izz)


def _get_drone_diaginertia(drone: dict[str, Any]) -> tuple[float, float, float]:
    # With inertial_reference "total_vehicle", drone.inertia is the whole-vehicle
    # inertia about the COM; the analytically-known wheel cylinder contributions are
    # subtracted so the assembled model reproduces drone.inertia.
    diaginertia = _get_raw_diaginertia(drone)
    if _get_inertial_reference(drone) == "body_only":
        return diaginertia

    _, wheel_pair_inertia = _wheel_pair_inertia_contribution(drone)
    body_diaginertia = tuple(total - wheels for total, wheels in zip(diaginertia, wheel_pair_inertia, strict=True))
    if any(value <= 0.0 for value in body_diaginertia):
        raise ValueError(
            "drone.inertia (total_vehicle) minus the wheel contributions "
            f"{tuple(round(value, 8) for value in wheel_pair_inertia)} must stay positive; "
            "reduce drone.wheels.mass or increase drone.inertia"
        )
    return body_diaginertia  # type: ignore[return-value]


def build_sensor_block(params: dict[str, Any], uav_specs: list[UAVModelSpec]) -> str:
    sensor_config = params["sensors"]
    sensor_lines: list[str] = []
    item_types = {item_name: item["type"] for item_name, item in sensor_config["items"].items()}
    for uav_spec in uav_specs:
        sensor_lines.extend(
            [
                f'    <{item_types["position"]} name="{uav_spec.sensor_names.position}" objtype="xbody" objname="{uav_spec.body_name}"/>',
                f'    <{item_types["linear_velocity"]} name="{uav_spec.sensor_names.linear_velocity}" objtype="xbody" objname="{uav_spec.body_name}"/>',
                f'    <{item_types["angular_velocity"]} name="{uav_spec.sensor_names.angular_velocity}" objtype="xbody" objname="{uav_spec.body_name}"/>',
                f'    <{item_types["x_axis"]} name="{uav_spec.sensor_names.x_axis}" objtype="xbody" objname="{uav_spec.body_name}"/>',
                f'    <{item_types["y_axis"]} name="{uav_spec.sensor_names.y_axis}" objtype="xbody" objname="{uav_spec.body_name}"/>',
                f'    <{item_types["z_axis"]} name="{uav_spec.sensor_names.z_axis}" objtype="xbody" objname="{uav_spec.body_name}"/>',
            ]
        )
    return "\n".join(sensor_lines)


def _build_uav_color(uav_index: int, alpha: float) -> list[float]:
    palette = (
        [0.92, 0.24, 0.24],
        [0.18, 0.54, 0.95],
        [0.20, 0.72, 0.38],
        [0.94, 0.66, 0.18],
        [0.61, 0.31, 0.88],
    )
    red, green, blue = palette[uav_index % len(palette)]
    return [red, green, blue, alpha]


def _mesh_config(params: dict[str, Any]) -> dict[str, Any] | None:
    """Optional visual mesh: a missing/null drone.mesh block (or null file) disables it.

    Without a mesh the vehicle still renders through its primitive geoms
    (body box, wheel cylinders, propeller discs), so params files can ship
    without any mesh asset.
    """
    mesh = params["drone"].get("mesh")
    if not isinstance(mesh, dict) or not mesh.get("file"):
        return None
    return mesh


def _build_mesh_asset_block(mesh: dict[str, Any] | None, file_reference: str | None) -> str:
    if mesh is None or file_reference is None:
        return ""
    return f'    <mesh name="drone_cad" file="{file_reference}" scale="{format_vector(mesh["scale"])}"/>'


def _build_drone_body_block(params: dict[str, Any], rotor_specs: list[RotorSpec], uav_specs: list[UAVModelSpec], initial_poses: list[InitialPoseSpec]) -> str:
    drone = params["drone"]
    mesh = _mesh_config(params)
    drone_mass = _get_drone_mass(drone)
    drone_diaginertia = _get_drone_diaginertia(drone)
    propeller_radius = float(drone["propeller"]["radius"])
    propeller_thickness = float(drone["propeller"]["thickness"])
    wheel_offset_y = float(drone["wheels"]["offset_y"])

    body_blocks: list[str] = []
    for uav_index, (uav_spec, initial_pose) in enumerate(zip(uav_specs, initial_poses, strict=True)):
        prefix = uav_spec.contact_prefix
        body_box_rgba = _build_uav_color(uav_index, float(drone["body_box"]["rgba"][3]))
        prop_rgba = _build_uav_color(uav_index, 0.55)
        rotor_site_lines = _build_rotor_site_lines(rotor_specs, prefix, propeller_radius, propeller_thickness, prop_rgba)
        body_attributes = [f'name="{uav_spec.body_name}"', f'pos="{format_vector(initial_pose.position)}"']
        if initial_pose.quaternion is not None:
            body_attributes.append(f'quat="{format_vector(initial_pose.quaternion)}"')
        mesh_geom_lines: list[str] = []
        if mesh is not None:
            mesh_geom_lines.append(
                f'      <geom type="mesh" mesh="drone_cad" contype="{format_scalar(mesh["contact"]["contype"])}" conaffinity="{format_scalar(mesh["contact"]["conaffinity"])}" mass="0" group="{format_scalar(mesh["contact"]["group"])}"/>'
            )
        body_blocks.append(
            "\n".join(
                [
                    f'    <body {" ".join(body_attributes)}>',
                    '      <joint type="free"/>',
                    f'      <inertial pos="0 0 0" mass="{format_scalar(drone_mass)}" diaginertia="{format_vector(list(drone_diaginertia))}"/>',
                    *rotor_site_lines,
                    *mesh_geom_lines,
                    f'      <geom name="{prefix}drone_body_box" type="box" size="{format_vector(drone["body_box"]["size"])}" euler="{format_vector(drone["body_box"]["euler"])}" mass="0" group="{format_scalar(drone["body_box"]["contact"]["group"])}" rgba="{format_vector(body_box_rgba)}" contype="{format_scalar(drone["body_box"]["contact"]["contype"])}" conaffinity="{format_scalar(drone["body_box"]["contact"]["conaffinity"])}"/>',
                    *_build_wheel_body_block(drone, prefix, "right", wheel_offset_y),
                    *_build_wheel_body_block(drone, prefix, "left", wheel_offset_y),
                    '    </body>',
                ]
            )
        )
    return "\n\n".join(body_blocks)


def _build_actuator_block(rotor_specs: list[RotorSpec], uav_specs: list[UAVModelSpec]) -> str:
    actuator_lines: list[str] = []
    for uav_spec in uav_specs:
        prefix = uav_spec.contact_prefix
        for actuator_name, rotor_spec in zip(uav_spec.actuator_names, rotor_specs, strict=True):
            # Site-transmission gear is expressed in the SITE frame. The prop site is
            # already oriented with zaxis=thrust_axis, so the gear must stay [0,0,1,...];
            # using thrust_axis here would rotate the thrust direction twice.
            gear_vector = [0.0, 0.0, 1.0, 0.0, 0.0, rotor_spec.spin_sign * rotor_spec.yaw_moment_ratio]
            actuator_lines.append(
                f'    <motor name="{actuator_name}" site="{prefix}prop_{rotor_spec.suffix}" gear="{format_vector(gear_vector)}"/>'
            )
    return "\n".join(actuator_lines)


def _resolve_mesh_file_reference(mesh: dict[str, Any], template_path: Path, output_path: Path, params_dir: Path | None) -> str:
    raw_mesh_file = Path(str(mesh["file"]))
    if raw_mesh_file.is_absolute():
        return format_scalar(raw_mesh_file.as_posix())

    # Relative paths prefer the directory of the params file that referenced the
    # mesh (the same rule actuation.calibration_file uses), so overlay repos can
    # ship their own mesh next to their params. The template directory stays as
    # the fallback, which keeps pre-existing layouts resolving unchanged.
    mesh_file_path: Path | None = None
    if params_dir is not None:
        candidate = (params_dir / raw_mesh_file).resolve()
        if candidate.is_file():
            mesh_file_path = candidate
    if mesh_file_path is None:
        mesh_file_path = (template_path.parent / raw_mesh_file).resolve()
    relative_mesh_path = Path(os.path.relpath(mesh_file_path, start=output_path.parent))
    return format_scalar(relative_mesh_path.as_posix())


_SUPPORTED_INTEGRATORS = ("Euler", "RK4", "implicit", "implicitfast")
_SUPPORTED_FRICTION_CONES = ("pyramidal", "elliptic")


def _build_option_extra_attrs(simulation: dict[str, Any]) -> str:
    # Optional solver settings appended to <option>. Kept out of the template's
    # required placeholders so legacy parameter files stay valid.
    extra_attrs = ""
    integrator = simulation.get("integrator")
    if integrator is not None:
        integrator = str(integrator).strip()
        matched = next((name for name in _SUPPORTED_INTEGRATORS if name.lower() == integrator.lower()), None)
        if matched is None:
            raise ValueError(f"simulation.integrator must be one of {_SUPPORTED_INTEGRATORS}")
        extra_attrs += f' integrator="{matched}"'
    cone = simulation.get("cone")
    if cone is not None:
        cone = str(cone).strip().lower()
        if cone not in _SUPPORTED_FRICTION_CONES:
            raise ValueError(f"simulation.cone must be one of {_SUPPORTED_FRICTION_CONES}")
        extra_attrs += f' cone="{cone}"'
    return extra_attrs


def _build_wall_extra_attrs(environment: dict[str, Any]) -> str:
    extra_attrs = ""
    wall_solimp = environment.get("wall_solimp")
    if wall_solimp is not None:
        extra_attrs += f' solimp="{format_vector(wall_solimp)}"'
    return extra_attrs


def _wall_face_x_world(environment: dict[str, Any]) -> float:
    wall_position = environment["wall_position"]
    wall_size = environment["wall_size"]
    return float(wall_position[0]) - float(wall_size[0])


def _sample_wall_guide_curve_yz(guide: dict[str, Any]) -> list[tuple[float, float]]:
    curve = guide.get("curve")
    if not isinstance(curve, dict):
        return []

    curve_type = str(curve.get("type", "")).strip().lower()
    if curve_type != "sinusoidal_ramp":
        raise ValueError('environment.wall_guide.curve.type must be "sinusoidal_ramp"')

    # y(t) = y_amplitude * sin(omega * t), z(t) = z_rate * t  (simulation-time parametrization)
    y_amplitude = float(curve.get("y_amplitude_m", curve.get("y_amplitude", 0.5)))
    angular_frequency = float(curve.get("y_angular_frequency_rad_s", 0.5))
    z_rate = float(curve.get("z_rate_m_s", 0.5))
    t_start = float(curve.get("t_start_s", 0.0))
    t_end = float(curve.get("t_end_s", 10.0))
    sample_count = int(curve.get("sample_count", 120))
    if t_end <= t_start:
        raise ValueError("environment.wall_guide.curve.t_end_s must exceed t_start_s")
    if sample_count < 2:
        raise ValueError("environment.wall_guide.curve.sample_count must be >= 2")

    samples: list[tuple[float, float]] = []
    for index in range(sample_count):
        fraction = float(index) / float(sample_count - 1)
        t_s = t_start + fraction * (t_end - t_start)
        y_m = y_amplitude * math.sin(angular_frequency * t_s)
        z_m = z_rate * t_s
        samples.append((y_m, z_m))
    return samples


def _resolve_wall_guide_waypoints(environment: dict[str, Any], guide: dict[str, Any]) -> list[list[float]]:
    surface_offset = float(guide.get("surface_offset", 0.025))
    auto_x = bool(guide.get("auto_x", True))
    guide_x = _wall_face_x_world(environment) - surface_offset

    curve_samples = _sample_wall_guide_curve_yz(guide)
    if curve_samples:
        return [[guide_x, y_m, z_m] for y_m, z_m in curve_samples]

    waypoints_raw = guide.get("waypoints", [])
    if not isinstance(waypoints_raw, list) or len(waypoints_raw) < 2:
        return []

    resolved: list[list[float]] = []
    for waypoint in waypoints_raw:
        if not isinstance(waypoint, (list, tuple)) or len(waypoint) < 2:
            raise ValueError("environment.wall_guide.waypoints entries must be [y, z] or [x, y, z]")
        values = [float(value) for value in waypoint]
        if len(values) == 2:
            resolved.append([guide_x, values[0], values[1]])
        else:
            x_coord = guide_x if auto_x else values[0]
            resolved.append([x_coord, values[1], values[2]])
    return resolved


def _build_wall_guide_block(environment: dict[str, Any]) -> str:
    # Visual-only path on the wall face for manual path-following practice.
    guide = environment.get("wall_guide")
    if not isinstance(guide, dict) or not bool(guide.get("enabled", False)):
        return ""

    waypoints = _resolve_wall_guide_waypoints(environment, guide)
    if len(waypoints) < 2:
        return ""

    line_radius = float(guide.get("line_radius", 0.012))
    marker_radius = float(guide.get("marker_radius", 0.03))
    rgba = format_vector(guide.get("rgba", [1.0, 0.85, 0.0, 0.95]))
    visual_attrs = ' contype="0" conaffinity="0" group="1"'
    uses_parametric_curve = isinstance(guide.get("curve"), dict)
    show_markers = bool(guide.get("show_markers", not uses_parametric_curve))
    marker_stride = max(1, int(guide.get("marker_stride", 1)))

    lines = ['    <body name="wall_guide" pos="0 0 0">']
    if show_markers:
        for index, point in enumerate(waypoints):
            if index % marker_stride != 0 and index not in (0, len(waypoints) - 1):
                continue
            lines.append(
                f'      <geom name="wall_guide_marker_{index}" type="sphere" pos="{format_vector(point)}" '
                f'size="{format_scalar(marker_radius)}" rgba="{rgba}"{visual_attrs}/>'
            )
    for index in range(len(waypoints) - 1):
        start = waypoints[index]
        end = waypoints[index + 1]
        fromto = format_vector([*start, *end])
        lines.append(
            f'      <geom name="wall_guide_seg_{index}" type="capsule" fromto="{fromto}" '
            f'size="{format_scalar(line_radius)}" rgba="{rgba}"{visual_attrs}/>'
        )
    lines.append("    </body>")
    return "\n".join(lines)


# Distinct colours handed out to successive references (i.e. successive UAVs) when
# an entry does not name its own rgba.
_OVERLAY_PALETTE = (
    (1.00, 0.45, 0.05, 1.0),
    (0.10, 0.55, 0.95, 1.0),
    (0.10, 0.70, 0.25, 1.0),
    (0.80, 0.20, 0.75, 1.0),
    (0.95, 0.80, 0.05, 1.0),
    (0.10, 0.75, 0.75, 1.0),
)


def _overlay_rgba(entry: dict[str, Any], defaults: dict[str, Any], key: str, index: int) -> str:
    rgba = entry.get("rgba", defaults.get(key))
    if rgba is None:
        rgba = _OVERLAY_PALETTE[index % len(_OVERLAY_PALETTE)]
    return format_vector(list(rgba))


def _overlay_ring_lines(name: str, x: float, centre_yz: Any, radius: float, rgba: str,
                        line_radius: float, segments: int, attrs: str) -> list[str]:
    cy, cz = float(centre_yz[0]), float(centre_yz[1])
    lines: list[str] = []
    for k in range(segments):
        a0 = 2.0 * math.pi * k / segments
        a1 = 2.0 * math.pi * (k + 1) / segments
        fromto = format_vector([
            x, cy + radius * math.cos(a0), cz + radius * math.sin(a0),
            x, cy + radius * math.cos(a1), cz + radius * math.sin(a1),
        ])
        lines.append(
            f'      <geom name="{name}_seg_{k}" type="capsule" fromto="{fromto}" '
            f'size="{format_scalar(line_radius)}" rgba="{rgba}"{attrs}/>'
        )
    return lines


def _build_wall_overlay_block(environment: dict[str, Any]) -> str:
    """Visual-only overlay on the wall face: obstacle outlines plus, per reference,
    a static path and a mocap sphere the simulator drives to the reference's current
    position (see WallOverlayAnimator). Everything here has contype/conaffinity=0 and
    therefore no effect on the physics.

    environment.wall_overlay:
      enabled          bool
      surface_offset   float   how far to float the overlay off the wall face [m]
      obstacles        list of {center_yz:[y,z], radius:float, rgba?, line_radius?}
      references       list of {start_yz:[y,z], target_yz:[y,z], speed:float, rgba?}
                       -- one entry per UAV; omitted rgba is taken from a palette so
                       several vehicles stay distinguishable.
      obstacle/reference: dicts of shared defaults (rgba, line_radius, segments,
                       path_radius, marker_radius).
    """
    overlay = environment.get("wall_overlay")
    if not isinstance(overlay, dict) or not bool(overlay.get("enabled", False)):
        return ""

    x = _wall_face_x_world(environment) - float(overlay.get("surface_offset", 0.02))
    attrs = ' contype="0" conaffinity="0" group="1"'
    obstacle_defaults = overlay.get("obstacle", {}) or {}
    reference_defaults = overlay.get("reference", {}) or {}

    static: list[str] = []
    for index, obstacle in enumerate(overlay.get("obstacles", []) or []):
        if "center_yz" not in obstacle or "radius" not in obstacle:
            raise ValueError("environment.wall_overlay.obstacles entries need center_yz and radius")
        static.extend(_overlay_ring_lines(
            f"overlay_obstacle_{index}", x, obstacle["center_yz"], float(obstacle["radius"]),
            _overlay_rgba(obstacle, obstacle_defaults, "rgba", index),
            float(obstacle.get("line_radius", obstacle_defaults.get("line_radius", 0.03))),
            max(8, int(obstacle.get("segments", obstacle_defaults.get("segments", 48)))),
            attrs,
        ))

    mocap: list[str] = []
    for index, reference in enumerate(overlay.get("references", []) or []):
        if "start_yz" not in reference or "target_yz" not in reference:
            raise ValueError("environment.wall_overlay.references entries need start_yz and target_yz")
        rgba = _overlay_rgba(reference, reference_defaults, "rgba", index)
        start = [float(v) for v in reference["start_yz"]]
        target = [float(v) for v in reference["target_yz"]]
        static.append(
            f'      <geom name="overlay_reference_{index}_path" type="capsule" '
            f'fromto="{format_vector([x, start[0], start[1], x, target[0], target[1]])}" '
            f'size="{format_scalar(float(reference.get("path_radius", reference_defaults.get("path_radius", 0.012))))}" '
            f'rgba="{rgba}"{attrs}/>'
        )
        # Mocap bodies must sit directly under <worldbody> and carry no joints.
        marker_radius = float(reference.get("marker_radius", reference_defaults.get("marker_radius", 0.07)))
        mocap.append(f'    <body name="overlay_reference_marker_{index}" mocap="true" '
                     f'pos="{format_vector([x, start[0], start[1]])}">')
        mocap.append(f'      <geom name="overlay_reference_marker_{index}_geom" type="sphere" '
                     f'size="{format_scalar(marker_radius)}" rgba="{rgba}"{attrs}/>')
        mocap.append("    </body>")

    if not static and not mocap:
        return ""
    lines: list[str] = []
    if static:
        lines.append('    <body name="wall_overlay" pos="0 0 0">')
        lines.extend(static)
        lines.append("    </body>")
    lines.extend(mocap)
    return "\n".join(lines)


def build_xml_replacements(params: dict[str, Any], num_uavs: int = 1, spawn_radius: float = 1.5) -> tuple[dict[str, str], SurfaceEvaluator | None, list[UAVModelSpec]]:
    simulation = params["simulation"]
    actuation = params["actuation"]
    environment = params["environment"]
    surface_spec = build_surface_model_spec(environment)
    rotor_specs = build_rotor_specs(params)
    uav_specs = build_uav_model_specs(params, num_uavs)
    initial_poses = build_initial_poses(params["drone"], surface_spec, num_uavs, spawn_radius, environment=params.get("environment"))

    replacements = {
        "__STATISTIC_EXTENT__": format_scalar(environment["statistic_extent"]),
        "__STATISTIC_CENTER__": format_vector(environment["statistic_center"]),
        "__GRAVITY__": format_vector(simulation["gravity"]),
        "__TIMESTEP__": format_scalar(simulation["timestep"]),
        "__OPTION_ITERATIONS__": format_scalar(int(simulation.get("iterations", 100))),
        "__OPTION_NOSLIP_ITERATIONS__": format_scalar(int(simulation.get("noslip_iterations", 0))),
        "__OPTION_EXTRA_ATTRS__": _build_option_extra_attrs(simulation),
        "__MAX_ROTOR_THRUST__": format_scalar(actuation["max_rotor_thrust"]),
        "__SURFACE_ASSET_BLOCK__": surface_spec.asset_block,
        "__SURFACE_GEOM_BLOCK__": surface_spec.geom_block,
        "__WALL_POS__": format_vector(environment["wall_position"]),
        "__WALL_SIZE__": format_vector(environment["wall_size"]),
        "__WALL_EULER__": format_vector(environment.get("wall_euler", [0.0, 0.0, 0.0])),
        "__WALL_CONDIM__": format_scalar(int(environment.get("wall_condim", 4))),
        "__WALL_RGBA__": format_vector(environment["wall_rgba"]),
        "__WALL_SOLREF__": format_vector(environment["wall_solref"]),
        "__WALL_EXTRA_ATTRS__": _build_wall_extra_attrs(environment),
        "__WALL_FRICTION__": format_vector(environment["wall_friction"]),
        "__WALL_CONTYPE__": format_scalar(environment["wall_contact"]["contype"]),
        "__WALL_CONAFFINITY__": format_scalar(environment["wall_contact"]["conaffinity"]),
        "__WALL_GUIDE_BLOCK__": _build_wall_guide_block(environment),
        "__WALL_OVERLAY_BLOCK__": _build_wall_overlay_block(environment),
        "__DRONE_BODY_BLOCK__": _build_drone_body_block(params, rotor_specs, uav_specs, initial_poses),
        "__ACTUATOR_BLOCK__": _build_actuator_block(rotor_specs, uav_specs),
        "__SENSOR_BLOCK__": build_sensor_block(params, uav_specs),
    }
    mesh = _mesh_config(params)
    if mesh is not None:
        # Raw (unresolved) reference: render_model_xml overwrites these with the
        # path-resolved form. __MESH_FILE__/__MESH_SCALE__ stay for templates
        # that still carry their own <mesh> asset line.
        replacements["__MESH_FILE__"] = format_scalar(mesh["file"])
        replacements["__MESH_SCALE__"] = format_vector(mesh["scale"])
    replacements["__MESH_ASSET_BLOCK__"] = _build_mesh_asset_block(mesh, replacements.get("__MESH_FILE__"))
    return replacements, surface_spec.evaluator, uav_specs


def render_model_xml(
    params: dict[str, Any],
    instance_id: int = 0,
    output_path: Path | None = None,
    num_uavs: int = 1,
    spawn_radius: float = 1.5,
    template_path: str | Path | None = None,
    generated_xml_dir: str | Path | None = None,
    path_resolver: PathResolver | None = None,
    params_dir: str | Path | None = None,
) -> tuple[Path, SurfaceEvaluator | None, list[UAVModelSpec]]:
    resolver = path_resolver or DEFAULT_PATH_RESOLVER
    resolved_template_path = resolver.get_xml_template_path(template_path)
    resolved_output_path = output_path or resolver.get_generated_xml_path(instance_id, output_directory=generated_xml_dir)
    template_text = resolved_template_path.read_text(encoding="utf-8")
    rendered_text = template_text
    replacements, surface_evaluator, uav_specs = build_xml_replacements(params, num_uavs=num_uavs, spawn_radius=spawn_radius)
    mesh = _mesh_config(params)
    if mesh is not None:
        resolved_reference = _resolve_mesh_file_reference(
            mesh,
            resolved_template_path,
            resolved_output_path,
            Path(params_dir) if params_dir is not None else None,
        )
        replacements["__MESH_FILE__"] = resolved_reference
        replacements["__MESH_ASSET_BLOCK__"] = _build_mesh_asset_block(mesh, resolved_reference)
    for placeholder, value in replacements.items():
        rendered_text = rendered_text.replace(placeholder, value)
    if "__MESH_FILE__" in rendered_text or "__MESH_SCALE__" in rendered_text:
        raise ValueError(
            "XML template still contains __MESH_FILE__/__MESH_SCALE__ but the params file has "
            "no drone.mesh block. Replace the template's <mesh> asset line with the "
            "__MESH_ASSET_BLOCK__ placeholder (emitted only when a mesh is configured), or add "
            "a drone.mesh block to the params file."
        )
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_output_path.write_text(rendered_text, encoding="utf-8")
    return resolved_output_path, surface_evaluator, uav_specs
