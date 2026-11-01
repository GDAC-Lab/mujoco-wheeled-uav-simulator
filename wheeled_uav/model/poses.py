"""Initial spawn states: surface-following, wall-contact, and explicit layouts.

Besides the pose, a spawn can carry a rolling initial velocity
(``drone.initial_spawn.forward_speed`` / ``turn_rate``): the vehicle then
starts moving along its wheels' rolling direction, tangent to the surface it
spawns on, with the wheels spinning so that they roll without slipping.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

import numpy as np

from ..config import get_wheels_config
from ..types import InitialPoseSpec, InitialVelocitySpec, SurfaceEvaluator, SurfaceModelSpec
from .surface import (
    build_surface_blocks,
    evaluate_height_function,
    evaluate_height_gradient,
    evaluate_surface_normal,
    get_surface_config,
)

__all__ = [
    "build_initial_poses",
    "build_rolling_velocity",
    "build_surface_model_spec",
    "build_surface_spawn_positions",
    "get_drone_initial_pose",
    "get_drone_initial_position",
    "get_wall_contact_initial_pose",
    "rotation_from_euler_xyz_degrees",
    "rotation_matrix_to_quaternion_wxyz",
]


def build_surface_model_spec(environment: dict[str, Any]) -> SurfaceModelSpec:
    surface_config = get_surface_config(environment)
    surface_asset_block, surface_geom_block, surface_evaluator = build_surface_blocks(surface_config)
    return SurfaceModelSpec(
        config=surface_config,
        asset_block=surface_asset_block,
        geom_block=surface_geom_block,
        evaluator=surface_evaluator,
    )


def build_surface_spawn_positions(initial_position: list[float], num_uavs: int, spawn_radius: float) -> list[list[float]]:
    if num_uavs == 1:
        return [[float(initial_position[0]), float(initial_position[1]), float(initial_position[2])]]
    if spawn_radius <= 0.0:
        raise ValueError(f"spawn_radius must be positive to place {num_uavs} UAVs on the spawn ring")

    positions: list[list[float]] = []
    for uav_index in range(num_uavs):
        angle = 2.0 * np.pi * uav_index / num_uavs
        positions.append(
            [
                float(initial_position[0] + spawn_radius * np.cos(angle)),
                float(initial_position[1] + spawn_radius * np.sin(angle)),
                float(initial_position[2]),
            ]
        )
    return positions


# ---------------------------------------------------------------------------
# Surface-following spawn (wheels resting on the terrain)
# ---------------------------------------------------------------------------

# Unit vectors in the world xy-plane: the heading (body x before roll and
# pitch) and the wheel axle (body y). The default spawn faces +x.
_DEFAULT_HEADING = (1.0, 0.0)
_DEFAULT_AXLE = (0.0, 1.0)


def _wheel_contact_height(
    surface_evaluator: SurfaceEvaluator,
    x_coord: float,
    y_coord: float,
    wheel_offset_y: float,
    wheel_radius: float,
    roll_angle: float,
    side_sign: float,
    axle: tuple[float, float] = _DEFAULT_AXLE,
) -> float:
    side_offset = side_sign * wheel_offset_y * math.cos(roll_angle)
    roll_offset = wheel_radius * math.sin(roll_angle)
    # Added term by term so that the default axle reproduces earlier releases
    # bit for bit (spawn poses end up in the generated XML).
    contact_x = x_coord + side_offset * axle[0] + roll_offset * axle[0]
    contact_y = y_coord + side_offset * axle[1] + roll_offset * axle[1]
    return evaluate_height_function(surface_evaluator, contact_x, contact_y)


def _solve_initial_roll_angle(
    surface_evaluator: SurfaceEvaluator,
    x_coord: float,
    y_coord: float,
    wheel_offset_y: float,
    wheel_radius: float,
    axle: tuple[float, float] = _DEFAULT_AXLE,
) -> float:
    if wheel_offset_y <= 0.0:
        return 0.0

    max_roll = math.radians(80.0)

    def residual(roll_angle: float) -> float:
        left_height = _wheel_contact_height(surface_evaluator, x_coord, y_coord, wheel_offset_y, wheel_radius, roll_angle, 1.0, axle)
        right_height = _wheel_contact_height(surface_evaluator, x_coord, y_coord, wheel_offset_y, wheel_radius, roll_angle, -1.0, axle)
        return left_height - right_height - 2.0 * wheel_offset_y * math.sin(roll_angle)

    sample_angles = np.linspace(-max_roll, max_roll, 257)
    sample_residuals = [residual(float(angle)) for angle in sample_angles]
    best_index = min(range(len(sample_residuals)), key=lambda index: abs(sample_residuals[index]))
    best_angle = float(sample_angles[best_index])

    for lower_index in range(len(sample_angles) - 1):
        lower_residual = sample_residuals[lower_index]
        upper_residual = sample_residuals[lower_index + 1]
        if lower_residual == 0.0:
            return float(sample_angles[lower_index])
        if lower_residual * upper_residual > 0.0:
            continue

        lower_angle = float(sample_angles[lower_index])
        upper_angle = float(sample_angles[lower_index + 1])
        for _ in range(60):
            midpoint_angle = 0.5 * (lower_angle + upper_angle)
            midpoint_residual = residual(midpoint_angle)
            if abs(midpoint_residual) <= 1.0e-10:
                return midpoint_angle
            if lower_residual * midpoint_residual <= 0.0:
                upper_angle = midpoint_angle
                upper_residual = midpoint_residual
            else:
                lower_angle = midpoint_angle
                lower_residual = midpoint_residual
        return 0.5 * (lower_angle + upper_angle)

    return best_angle


def _get_initial_wheel_contact_clearance(surface_config: dict[str, Any], drone: dict[str, Any]) -> float:
    configured_clearance = surface_config.get("initial_wheel_contact_clearance")
    if configured_clearance is not None:
        return float(configured_clearance)

    base_initial_position = [float(value) for value in drone["initial_position"]]
    wheel_radius = float(drone["wheels"]["radius"])
    return base_initial_position[2] - wheel_radius


def _wheel_offset_z(drone: dict[str, Any]) -> float:
    wheels = get_wheels_config(drone)
    return 0.0 if wheels is None else float(wheels.get("offset_z", 0.0))


def _get_initial_pitch_angle(
    surface_evaluator: SurfaceEvaluator,
    x_coord: float,
    y_coord: float,
    heading: tuple[float, float] = _DEFAULT_HEADING,
) -> float:
    dh_dx, dh_dy = evaluate_height_gradient(surface_evaluator, x_coord, y_coord)
    if heading == _DEFAULT_HEADING:  # bit-identical to earlier releases
        return -math.atan(dh_dx)
    return -math.atan(dh_dx * heading[0] + dh_dy * heading[1])


def _compose_roll_pitch_quaternion(roll_angle: float, pitch_angle: float) -> tuple[float, float, float, float]:
    half_roll = 0.5 * roll_angle
    half_pitch = 0.5 * pitch_angle
    cos_roll = math.cos(half_roll)
    sin_roll = math.sin(half_roll)
    cos_pitch = math.cos(half_pitch)
    sin_pitch = math.sin(half_pitch)
    return (
        cos_roll * cos_pitch,
        sin_roll * cos_pitch,
        cos_roll * sin_pitch,
        sin_roll * sin_pitch,
    )


def _yaw_quaternion(yaw: float) -> tuple[float, float, float, float]:
    return (math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw))


def _multiply_quaternions(
    first: tuple[float, float, float, float], second: tuple[float, float, float, float]
) -> tuple[float, float, float, float]:
    w1, x1, y1, z1 = first
    w2, x2, y2, z2 = second
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


def _quaternion_to_rotation_matrix(quaternion: tuple[float, float, float, float] | None) -> np.ndarray:
    if quaternion is None:
        return np.eye(3)
    w, x, y, z = quaternion
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
            [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
            [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
        ]
    )


def rotation_from_euler_xyz_degrees(euler_degrees: list[float]) -> np.ndarray:
    """Rotation of MuJoCo's default ``euler`` convention (intrinsic x, y, z)."""
    roll, pitch, yaw = (math.radians(float(angle)) for angle in euler_degrees)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rotation_x = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    rotation_y = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rotation_z = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return rotation_x @ rotation_y @ rotation_z


def _normalize_vector3(vector: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1.0e-9:
        return fallback
    return vector / norm


def rotation_matrix_to_quaternion_wxyz(rotation_matrix: np.ndarray) -> tuple[float, float, float, float]:
    rotation = np.asarray(rotation_matrix, dtype=float).reshape(3, 3)
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (rotation[2, 1] - rotation[1, 2]) / scale
        y = (rotation[0, 2] - rotation[2, 0]) / scale
        z = (rotation[1, 0] - rotation[0, 1]) / scale
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        scale = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
        w = (rotation[2, 1] - rotation[1, 2]) / scale
        x = 0.25 * scale
        y = (rotation[0, 1] + rotation[1, 0]) / scale
        z = (rotation[0, 2] + rotation[2, 0]) / scale
    elif rotation[1, 1] > rotation[2, 2]:
        scale = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
        w = (rotation[0, 2] - rotation[2, 0]) / scale
        x = (rotation[0, 1] + rotation[1, 0]) / scale
        y = 0.25 * scale
        z = (rotation[1, 2] + rotation[2, 1]) / scale
    else:
        scale = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
        w = (rotation[1, 0] - rotation[0, 1]) / scale
        x = (rotation[0, 2] + rotation[2, 0]) / scale
        y = (rotation[1, 2] + rotation[2, 1]) / scale
        z = 0.25 * scale
    quaternion = np.array([w, x, y, z], dtype=float)
    quaternion /= max(np.linalg.norm(quaternion), 1.0e-12)
    return (float(quaternion[0]), float(quaternion[1]), float(quaternion[2]), float(quaternion[3]))


# ---------------------------------------------------------------------------
# Wall-contact spawn (wheels resting on the environment wall)
# ---------------------------------------------------------------------------

# Spawning with the wheels EXACTLY touching the wall gives the contact solver a
# zero-margin contact at initialization, which is the most fragile possible
# initial state (tiny numerical penetration -> large corrective impulses).
# A few millimetres of standoff lets the vehicle settle gently instead; the
# controller's pressing force closes the gap at negligible speed.
_DEFAULT_WALL_SPAWN_CLEARANCE_M = 0.005


def _resolve_wall_contact_center_x(environment: dict[str, Any], drone: dict[str, Any]) -> float:
    spawn_cfg = drone.get("initial_spawn", {})
    if isinstance(spawn_cfg, dict) and "center_x" in spawn_cfg:
        return float(spawn_cfg["center_x"])
    wall_clearance = _DEFAULT_WALL_SPAWN_CLEARANCE_M
    if isinstance(spawn_cfg, dict) and "wall_clearance_m" in spawn_cfg:
        wall_clearance = float(spawn_cfg["wall_clearance_m"])
    wall_position = environment["wall_position"]
    wall_size = environment["wall_size"]
    wall_face_x = float(wall_position[0]) - float(wall_size[0])
    wheels = get_wheels_config(drone)
    if wheels is None:
        raise ValueError('drone.initial_spawn.mode="wall_contact" needs wheels or an explicit initial_spawn.center_x')
    return wall_face_x - float(wheels["radius"]) - wall_clearance


def get_wall_contact_initial_pose(
    drone: dict[str, Any],
    environment: dict[str, Any],
    *,
    x_coord: float | None = None,
    y_coord: float | None = None,
) -> InitialPoseSpec:
    # Spawn near the wall with a wall-leaning startup attitude. By default the
    # wheels sit initial_spawn.wall_clearance_m (5 mm) short of the wall face
    # so the run starts from a settled, contact-free state rather than an
    # exact-touch condition (see _resolve_wall_contact_center_x).
    base_initial_position = [float(value) for value in drone["initial_position"]]
    spawn_cfg = drone.get("initial_spawn", {})
    if not isinstance(spawn_cfg, dict):
        spawn_cfg = {}

    b2_axis = _normalize_vector3(
        np.asarray(spawn_cfg.get("b2_body", [0.0, 1.0, 0.0]), dtype=float).reshape(3),
        np.array([0.0, 1.0, 0.0], dtype=float),
    )
    b3_axis = _normalize_vector3(
        np.asarray(spawn_cfg.get("b3_body", [0.640, 0.0, 0.768]), dtype=float).reshape(3),
        np.array([0.0, 0.0, 1.0], dtype=float),
    )
    b1_axis = _normalize_vector3(np.cross(b2_axis, b3_axis), np.array([1.0, 0.0, 0.0], dtype=float))
    rotation_matrix = np.column_stack((b1_axis, b2_axis, b3_axis))

    center_x = _resolve_wall_contact_center_x(environment, drone)
    resolved_x = center_x if x_coord is None else float(x_coord)
    resolved_y = base_initial_position[1] if y_coord is None else float(y_coord)
    # center_x and initial_position[2] place the wheel axle; the body origin sits offset_z below it along b3.
    axle_offset = rotation_matrix @ np.array([0.0, 0.0, _wheel_offset_z(drone)])
    position = [resolved_x - float(axle_offset[0]), resolved_y - float(axle_offset[1]), base_initial_position[2] - float(axle_offset[2])]
    return InitialPoseSpec(position=position, quaternion=rotation_matrix_to_quaternion_wxyz(rotation_matrix))


def get_drone_initial_pose(
    drone: dict[str, Any],
    surface_evaluator: SurfaceEvaluator | None,
    surface_config: dict[str, Any],
    x_coord: float | None = None,
    y_coord: float | None = None,
    *,
    yaw: float = 0.0,
) -> InitialPoseSpec:
    """Spawn pose at (x, y), heading ``yaw`` [rad] counterclockwise from +x.

    On a height-function surface both wheels are set on the terrain: the roll
    makes the two wheel contacts consistent across the axle and the pitch
    follows the slope along the heading.
    """
    base_initial_position = [float(value) for value in drone["initial_position"]]
    resolved_x_coord = base_initial_position[0] if x_coord is None else float(x_coord)
    resolved_y_coord = base_initial_position[1] if y_coord is None else float(y_coord)
    initial_position = [resolved_x_coord, resolved_y_coord, base_initial_position[2]]
    follow_surface_for_initial_position = bool(surface_config.get("follow_surface_for_initial_position", True))
    if not follow_surface_for_initial_position or surface_evaluator is None:
        return InitialPoseSpec(position=initial_position, quaternion=None if yaw == 0.0 else _yaw_quaternion(yaw))

    heading = _DEFAULT_HEADING if yaw == 0.0 else (math.cos(yaw), math.sin(yaw))
    axle = _DEFAULT_AXLE if yaw == 0.0 else (-math.sin(yaw), math.cos(yaw))
    wheel_config = get_wheels_config(drone)
    if wheel_config is None:
        # Without wheels the body origin rests initial_position.z above the
        # terrain, with the body tilted to the terrain slopes.
        dh_dx, dh_dy = evaluate_height_gradient(surface_evaluator, resolved_x_coord, resolved_y_coord)
        roll_angle = math.atan(dh_dx * axle[0] + dh_dy * axle[1])
        pitch_angle = _get_initial_pitch_angle(surface_evaluator, resolved_x_coord, resolved_y_coord, heading)
        initial_position[2] += evaluate_height_function(surface_evaluator, resolved_x_coord, resolved_y_coord)
        quaternion = _compose_roll_pitch_quaternion(roll_angle, pitch_angle)
        if yaw != 0.0:
            quaternion = _multiply_quaternions(_yaw_quaternion(yaw), quaternion)
        return InitialPoseSpec(position=initial_position, quaternion=quaternion)
    wheel_offset_y = float(wheel_config["offset_y"])
    wheel_radius = float(wheel_config["radius"])
    roll_angle = _solve_initial_roll_angle(surface_evaluator, resolved_x_coord, resolved_y_coord, wheel_offset_y, wheel_radius, axle)
    pitch_angle = _get_initial_pitch_angle(surface_evaluator, resolved_x_coord, resolved_y_coord, heading)
    left_height = _wheel_contact_height(surface_evaluator, resolved_x_coord, resolved_y_coord, wheel_offset_y, wheel_radius, roll_angle, 1.0, axle)
    right_height = _wheel_contact_height(surface_evaluator, resolved_x_coord, resolved_y_coord, wheel_offset_y, wheel_radius, roll_angle, -1.0, axle)
    base_clearance = _get_initial_wheel_contact_clearance(surface_config, drone)
    initial_position[2] = base_clearance + 0.5 * (left_height + right_height) + wheel_radius * math.cos(roll_angle)
    quaternion = _compose_roll_pitch_quaternion(roll_angle, pitch_angle)
    if yaw != 0.0:
        quaternion = _multiply_quaternions(_yaw_quaternion(yaw), quaternion)
    # The height above places the wheel axle; the body origin sits offset_z below it along b3.
    offset_z = _wheel_offset_z(drone)
    if offset_z != 0.0:
        w, x, y, z = quaternion
        b3 = np.array([2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y)])
        initial_position = [float(value) for value in np.asarray(initial_position) - offset_z * b3]
    return InitialPoseSpec(position=initial_position, quaternion=quaternion)


def get_drone_initial_position(
    drone: dict[str, Any],
    surface_evaluator: SurfaceEvaluator | None,
    surface_config: dict[str, Any],
) -> list[float]:
    return get_drone_initial_pose(drone, surface_evaluator, surface_config).position


def _per_uav_values(spawn_cfg: dict[str, Any], key: str, num_uavs: int) -> list[float]:
    # A scalar applies to every UAV; a list gives one value per UAV.
    raw_value = spawn_cfg.get(key, 0.0)
    if isinstance(raw_value, (list, tuple)):
        if len(raw_value) < num_uavs:
            raise ValueError(f"drone.initial_spawn.{key} needs one value per UAV ({num_uavs}), got {len(raw_value)}")
        values = [float(value) for value in raw_value[:num_uavs]]
    else:
        values = [float(raw_value)] * num_uavs
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"drone.initial_spawn.{key} must be finite")
    return values


def _surface_spawn_normal(surface_spec: SurfaceModelSpec, x_coord: float, y_coord: float) -> np.ndarray:
    follow_surface = bool(surface_spec.config.get("follow_surface_for_initial_position", True))
    if not follow_surface or surface_spec.evaluator is None:
        return np.array([0.0, 0.0, 1.0])
    return evaluate_surface_normal(surface_spec.evaluator, x_coord, y_coord)


def _wall_spawn_normal(environment: dict[str, Any]) -> np.ndarray:
    # Outward normal of the working (-x) face of the wall box, toward the vehicle.
    rotation = rotation_from_euler_xyz_degrees(environment.get("wall_euler", [0.0, 0.0, 0.0]))
    return rotation @ np.array([-1.0, 0.0, 0.0])


def build_rolling_velocity(
    drone: dict[str, Any],
    pose: InitialPoseSpec,
    contact_normal: np.ndarray,
    forward_speed: float,
    turn_rate: float,
) -> InitialVelocitySpec | None:
    """Initial velocity of a vehicle rolling without slip on a contact plane.

    ``forward_speed`` [m/s] is the speed of the axle midpoint along the rolling
    direction, the axle (body y) crossed with ``contact_normal``. That
    direction is tangent to the contact plane and normal to the axle, so the
    vehicle keeps its clearance and does not slip sideways. ``turn_rate``
    [rad/s] is the yaw rate about ``contact_normal``. Each wheel spins at the
    rate that makes it roll without slipping on the plane. Returns None for a
    vehicle at rest.
    """
    if forward_speed == 0.0 and turn_rate == 0.0:
        return None
    rotation = _quaternion_to_rotation_matrix(pose.quaternion)
    axle_axis, up_axis = rotation[:, 1], rotation[:, 2]
    normal = np.asarray(contact_normal, dtype=float) / np.linalg.norm(contact_normal)
    rolling = np.cross(axle_axis, normal)
    rolling_norm = float(np.linalg.norm(rolling))
    if rolling_norm < 1.0e-6:
        raise ValueError("cannot spawn rolling: the wheel axle is parallel to the contact normal")
    rolling /= rolling_norm

    angular_velocity = turn_rate * normal
    # The axle midpoint sits offset_z above the body origin along body z.
    axle_midpoint_offset = _wheel_offset_z(drone) * up_axis
    origin_velocity = forward_speed * rolling - np.cross(angular_velocity, axle_midpoint_offset)
    linear = tuple(float(value) for value in origin_velocity)
    angular = tuple(float(value) for value in angular_velocity)
    wheels = get_wheels_config(drone)
    if wheels is None:
        return InitialVelocitySpec(linear=linear, angular=angular)
    # The wheel centres at +-offset_y along the axle move at
    # forward_speed -+ turn_rate * offset_y * |axle x normal| along `rolling`.
    # A wheel touches the plane at the lowest point of its rim, which leaves
    # the normal through the centre when the axle is tilted to the plane;
    # keeping that point at rest takes the spin
    # (centre speed) / radius - turn_rate * (axle . normal).
    wheel_radius = float(wheels["radius"])
    lateral = turn_rate * float(wheels["offset_y"]) * rolling_norm
    tilt_spin = turn_rate * float(axle_axis @ normal)
    return InitialVelocitySpec(
        linear=linear,
        angular=angular,
        left_wheel_rate=(forward_speed - lateral) / wheel_radius - tilt_spin,
        right_wheel_rate=(forward_speed + lateral) / wheel_radius - tilt_spin,
    )


def build_initial_poses(
    drone: dict[str, Any],
    surface_spec: SurfaceModelSpec,
    num_uavs: int,
    spawn_radius: float,
    *,
    environment: dict[str, Any] | None = None,
) -> list[InitialPoseSpec]:
    spawn_cfg = drone.get("initial_spawn", {})
    if not isinstance(spawn_cfg, dict):
        spawn_cfg = {}
    spawn_mode = str(spawn_cfg.get("mode", "surface")).strip().lower()
    yaws = [math.radians(value) for value in _per_uav_values(spawn_cfg, "yaw_deg", num_uavs)]
    forward_speeds = _per_uav_values(spawn_cfg, "forward_speed", num_uavs)
    turn_rates = _per_uav_values(spawn_cfg, "turn_rate", num_uavs)

    if spawn_mode == "wall_contact":
        if environment is None:
            raise ValueError('drone.initial_spawn.mode="wall_contact" requires environment wall metadata')
        if "yaw_deg" in spawn_cfg:
            raise ValueError(
                'drone.initial_spawn.yaw_deg applies to the "surface" and "explicit" modes; '
                '"wall_contact" takes its attitude from b2_body/b3_body'
            )
        base_pose = get_wall_contact_initial_pose(drone, environment)
        initial_positions = build_surface_spawn_positions(base_pose.position, num_uavs, spawn_radius)
        poses = [
            get_wall_contact_initial_pose(drone, environment, x_coord=position[0], y_coord=position[1])
            for position in initial_positions
        ]
        normals = [_wall_spawn_normal(environment)] * num_uavs
    elif spawn_mode == "explicit":
        raw_positions = spawn_cfg.get("positions_xy")
        if not isinstance(raw_positions, list) or len(raw_positions) < num_uavs:
            raise ValueError(
                'drone.initial_spawn.mode="explicit" requires positions_xy with at least num_uavs [x, y] entries'
            )
        poses = []
        normals = []
        for entry, yaw in zip(raw_positions[:num_uavs], yaws):
            if not isinstance(entry, (list, tuple)) or len(entry) < 2:
                raise ValueError("initial_spawn.positions_xy entries must be [x, y] pairs")
            x_coord, y_coord = float(entry[0]), float(entry[1])
            poses.append(
                get_drone_initial_pose(
                    drone,
                    surface_spec.evaluator,
                    surface_spec.config,
                    x_coord=x_coord,
                    y_coord=y_coord,
                    yaw=yaw,
                )
            )
            normals.append(_surface_spawn_normal(surface_spec, x_coord, y_coord))
    else:
        initial_position = get_drone_initial_position(drone, surface_spec.evaluator, surface_spec.config)
        initial_positions = build_surface_spawn_positions(initial_position, num_uavs, spawn_radius)
        poses = [
            get_drone_initial_pose(
                drone, surface_spec.evaluator, surface_spec.config, x_coord=position[0], y_coord=position[1], yaw=yaw
            )
            for position, yaw in zip(initial_positions, yaws)
        ]
        normals = [_surface_spawn_normal(surface_spec, position[0], position[1]) for position in initial_positions]

    return [
        replace(pose, velocity=build_rolling_velocity(drone, pose, normal, forward_speed, turn_rate))
        for pose, normal, forward_speed, turn_rate in zip(poses, normals, forward_speeds, turn_rates)
    ]
