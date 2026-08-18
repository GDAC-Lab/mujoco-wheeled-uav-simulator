"""Initial spawn poses: surface-following, wall-contact, and explicit layouts."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..types import InitialPoseSpec, SurfaceEvaluator, SurfaceModelSpec
from .surface import (
    build_surface_blocks,
    evaluate_height_function,
    evaluate_height_gradient,
    get_surface_config,
)

__all__ = [
    "build_initial_poses",
    "build_surface_model_spec",
    "build_surface_spawn_positions",
    "get_drone_initial_pose",
    "get_drone_initial_position",
    "get_wall_contact_initial_pose",
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

def _wheel_contact_height(
    surface_evaluator: SurfaceEvaluator,
    x_coord: float,
    y_coord: float,
    wheel_offset_y: float,
    wheel_radius: float,
    roll_angle: float,
    side_sign: float,
) -> float:
    contact_y = y_coord + side_sign * wheel_offset_y * math.cos(roll_angle) + wheel_radius * math.sin(roll_angle)
    return evaluate_height_function(surface_evaluator, x_coord, contact_y)


def _solve_initial_roll_angle(
    surface_evaluator: SurfaceEvaluator,
    x_coord: float,
    y_coord: float,
    wheel_offset_y: float,
    wheel_radius: float,
) -> float:
    if wheel_offset_y <= 0.0:
        return 0.0

    max_roll = math.radians(80.0)

    def residual(roll_angle: float) -> float:
        left_height = _wheel_contact_height(surface_evaluator, x_coord, y_coord, wheel_offset_y, wheel_radius, roll_angle, 1.0)
        right_height = _wheel_contact_height(surface_evaluator, x_coord, y_coord, wheel_offset_y, wheel_radius, roll_angle, -1.0)
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


def _get_initial_pitch_angle(surface_evaluator: SurfaceEvaluator, x_coord: float, y_coord: float) -> float:
    dh_dx, _ = evaluate_height_gradient(surface_evaluator, x_coord, y_coord)
    return -math.atan(dh_dx)


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
    wheel_radius = float(drone["wheels"]["radius"])
    return wall_face_x - wheel_radius - wall_clearance


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
    position = [resolved_x, resolved_y, base_initial_position[2]]
    return InitialPoseSpec(position=position, quaternion=rotation_matrix_to_quaternion_wxyz(rotation_matrix))


def get_drone_initial_pose(
    drone: dict[str, Any],
    surface_evaluator: SurfaceEvaluator | None,
    surface_config: dict[str, Any],
    x_coord: float | None = None,
    y_coord: float | None = None,
) -> InitialPoseSpec:
    base_initial_position = [float(value) for value in drone["initial_position"]]
    resolved_x_coord = base_initial_position[0] if x_coord is None else float(x_coord)
    resolved_y_coord = base_initial_position[1] if y_coord is None else float(y_coord)
    initial_position = [resolved_x_coord, resolved_y_coord, base_initial_position[2]]
    follow_surface_for_initial_position = bool(surface_config.get("follow_surface_for_initial_position", True))
    if not follow_surface_for_initial_position or surface_evaluator is None:
        return InitialPoseSpec(position=initial_position)

    wheel_config = drone["wheels"]
    wheel_offset_y = float(wheel_config["offset_y"])
    wheel_radius = float(wheel_config["radius"])
    roll_angle = _solve_initial_roll_angle(surface_evaluator, resolved_x_coord, resolved_y_coord, wheel_offset_y, wheel_radius)
    pitch_angle = _get_initial_pitch_angle(surface_evaluator, resolved_x_coord, resolved_y_coord)
    left_height = _wheel_contact_height(surface_evaluator, resolved_x_coord, resolved_y_coord, wheel_offset_y, wheel_radius, roll_angle, 1.0)
    right_height = _wheel_contact_height(surface_evaluator, resolved_x_coord, resolved_y_coord, wheel_offset_y, wheel_radius, roll_angle, -1.0)
    base_clearance = _get_initial_wheel_contact_clearance(surface_config, drone)
    initial_position[2] = base_clearance + 0.5 * (left_height + right_height) + wheel_radius * math.cos(roll_angle)
    quaternion = _compose_roll_pitch_quaternion(roll_angle, pitch_angle)
    return InitialPoseSpec(position=initial_position, quaternion=quaternion)


def get_drone_initial_position(
    drone: dict[str, Any],
    surface_evaluator: SurfaceEvaluator | None,
    surface_config: dict[str, Any],
) -> list[float]:
    return get_drone_initial_pose(drone, surface_evaluator, surface_config).position


def build_initial_poses(
    drone: dict[str, Any],
    surface_spec: SurfaceModelSpec,
    num_uavs: int,
    spawn_radius: float,
    *,
    environment: dict[str, Any] | None = None,
) -> list[InitialPoseSpec]:
    spawn_cfg = drone.get("initial_spawn", {})
    spawn_mode = "surface"
    if isinstance(spawn_cfg, dict):
        spawn_mode = str(spawn_cfg.get("mode", "surface")).strip().lower()

    if spawn_mode == "wall_contact":
        if environment is None:
            raise ValueError('drone.initial_spawn.mode="wall_contact" requires environment wall metadata')
        base_pose = get_wall_contact_initial_pose(drone, environment)
        initial_positions = build_surface_spawn_positions(base_pose.position, num_uavs, spawn_radius)
        return [
            get_wall_contact_initial_pose(drone, environment, x_coord=position[0], y_coord=position[1])
            for position in initial_positions
        ]

    if spawn_mode == "explicit":
        raw_positions = spawn_cfg.get("positions_xy")
        if not isinstance(raw_positions, list) or len(raw_positions) < num_uavs:
            raise ValueError(
                'drone.initial_spawn.mode="explicit" requires positions_xy with at least num_uavs [x, y] entries'
            )
        poses = []
        for entry in raw_positions[:num_uavs]:
            if not isinstance(entry, (list, tuple)) or len(entry) < 2:
                raise ValueError("initial_spawn.positions_xy entries must be [x, y] pairs")
            poses.append(
                get_drone_initial_pose(
                    drone,
                    surface_spec.evaluator,
                    surface_spec.config,
                    x_coord=float(entry[0]),
                    y_coord=float(entry[1]),
                )
            )
        return poses

    initial_position = get_drone_initial_position(drone, surface_spec.evaluator, surface_spec.config)
    initial_positions = build_surface_spawn_positions(initial_position, num_uavs, spawn_radius)
    return [
        get_drone_initial_pose(drone, surface_spec.evaluator, surface_spec.config, x_coord=position[0], y_coord=position[1])
        for position in initial_positions
    ]
