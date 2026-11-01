from __future__ import annotations

import copy
import json
import math

import mujoco
import numpy as np
import pytest

from wheeled_uav.config import load_vehicle_params
from wheeled_uav.model.poses import build_initial_poses, build_surface_model_spec
from wheeled_uav.runtime.loop import run_headless_loop
from wheeled_uav.runtime.scene import SimulationRequest, load_simulation_scene


def _hill_surface(slope_x: float, slope_y: float, centers: list[tuple[float, float]]) -> dict:
    parameters = {"slope_x": slope_x, "slope_y": slope_y}
    for index, (center_x, center_y) in enumerate(centers, start=1):
        parameters.update(
            {
                f"gauss{index}_amplitude": 0.4 if index == 1 else -0.3,
                f"gauss{index}_center_x": center_x,
                f"gauss{index}_center_y": center_y,
                f"gauss{index}_denominator": 0.5,
            }
        )
    return {
        "surface": {
            "mode": "composite",
            "solref": [0.01, 1.0],
            "contact": {"contype": 1, "conaffinity": 1},
            "initial_wheel_contact_clearance": 0.0,
            "height_function": {
                "name": "composite",
                "x_range": [-2.0, 2.0],
                "y_range": [-2.0, 2.0],
                "grid_resolution": [21, 21],
                "parameters": parameters,
            },
        }
    }


def _rotate_xy(point: tuple[float, float], angle: float) -> tuple[float, float]:
    cos_angle, sin_angle = math.cos(angle), math.sin(angle)
    return (cos_angle * point[0] - sin_angle * point[1], sin_angle * point[0] + cos_angle * point[1])


def _rotation(quaternion) -> np.ndarray:
    matrix = np.zeros(9)
    mujoco.mju_quat2Mat(matrix, np.asarray(quaternion, dtype=float))
    return matrix.reshape(3, 3)


def _rim_contact_velocity(velocity, axle: np.ndarray, normal: np.ndarray, side: float) -> np.ndarray:
    """Velocity of the rim point that touches the plane, for the wheel at side * 0.2 along the axle."""
    linear, angular = np.asarray(velocity.linear), np.asarray(velocity.angular)
    wheel_rate = velocity.left_wheel_rate if side > 0 else velocity.right_wheel_rate
    centre_velocity = linear + np.cross(angular, side * 0.2 * axle)
    toward_plane = -(normal - (axle @ normal) * axle)
    toward_plane /= np.linalg.norm(toward_plane)
    return centre_velocity + np.cross(angular + wheel_rate * axle, 0.15 * toward_plane)


def _drone(**initial_spawn) -> dict:
    return {
        "initial_position": [0.0, 0.0, 0.3],
        "initial_spawn": {"mode": "explicit", "positions_xy": [[0.4, -0.3]], **initial_spawn},
        "wheels": {"offset_y": 0.2, "radius": 0.15},
    }


def test_yawed_spawn_matches_the_unyawed_spawn_on_the_rotated_terrain():
    # Turning the vehicle by yaw on a terrain is the same as turning the terrain
    # by -yaw under an unturned vehicle (the gaussian bumps are isotropic).
    yaw = math.radians(35.0)
    centers = [(-0.5, 0.2), (0.6, 0.5)]
    terrain = build_surface_model_spec(_hill_surface(0.2, -0.1, centers))
    slope = _rotate_xy((0.2, -0.1), -yaw)
    turned_terrain = build_surface_model_spec(_hill_surface(slope[0], slope[1], [_rotate_xy(c, -yaw) for c in centers]))

    (yawed,) = build_initial_poses(_drone(yaw_deg=35.0), terrain, num_uavs=1, spawn_radius=0.0)
    position_in_turned_frame = _rotate_xy((0.4, -0.3), -yaw)
    (reference,) = build_initial_poses(
        _drone(positions_xy=[list(position_in_turned_frame)]), turned_terrain, num_uavs=1, spawn_radius=0.0
    )

    assert yawed.position[:2] == [0.4, -0.3]
    assert math.isclose(yawed.position[2], reference.position[2], abs_tol=1e-9)
    turn = np.array([[math.cos(yaw), -math.sin(yaw), 0.0], [math.sin(yaw), math.cos(yaw), 0.0], [0.0, 0.0, 1.0]])
    np.testing.assert_allclose(_rotation(yawed.quaternion), turn @ _rotation(reference.quaternion), atol=1e-9)


def test_rolling_velocity_is_tangent_free_of_side_slip_and_rolls_the_wheels():
    terrain = build_surface_model_spec(_hill_surface(0.2, -0.1, [(-0.5, 0.2)]))
    (pose,) = build_initial_poses(
        _drone(yaw_deg=20.0, forward_speed=0.4, turn_rate=0.3), terrain, num_uavs=1, spawn_radius=0.0
    )
    velocity = pose.velocity
    assert velocity is not None

    rotation = _rotation(pose.quaternion)
    axle = rotation[:, 1]
    from wheeled_uav.model.surface import evaluate_surface_normal

    normal = evaluate_surface_normal(terrain.evaluator, 0.4, -0.3)
    linear = np.asarray(velocity.linear)
    angular = np.asarray(velocity.angular)
    assert abs(linear @ normal) < 1e-12  # keeps its clearance
    assert abs(linear @ axle) < 1e-12  # no side slip
    assert math.isclose(np.linalg.norm(linear), 0.4, rel_tol=1e-12)
    np.testing.assert_allclose(angular, 0.3 * normal, atol=1e-15)
    # Each wheel's contact point is at rest: rolling without slip.
    for side in (1.0, -1.0):
        np.testing.assert_allclose(_rim_contact_velocity(velocity, axle, normal, side), 0.0, atol=1e-12)
    assert velocity.left_wheel_rate < velocity.right_wheel_rate  # turning left


def test_wheels_roll_without_slip_with_the_axle_tilted_to_the_contact_plane():
    from wheeled_uav.model.poses import build_rolling_velocity
    from wheeled_uav.types import InitialPoseSpec

    # Body rolled by 20 deg on level ground: the axle is tilted 20 deg to the plane.
    roll = math.radians(20.0)
    pose = InitialPoseSpec(position=[0.0, 0.0, 0.2], quaternion=(math.cos(roll / 2), math.sin(roll / 2), 0.0, 0.0))
    normal = np.array([0.0, 0.0, 1.0])
    velocity = build_rolling_velocity(_drone(), pose, normal, 0.4, 0.3)

    axle = _rotation(pose.quaternion)[:, 1]
    assert abs(axle @ normal) > 0.3
    assert abs(np.asarray(velocity.linear) @ normal) < 1e-12
    assert abs(np.asarray(velocity.linear) @ axle) < 1e-12
    for side in (1.0, -1.0):
        np.testing.assert_allclose(_rim_contact_velocity(velocity, axle, normal, side), 0.0, atol=1e-12)


def test_vehicles_at_rest_carry_no_velocity():
    terrain = build_surface_model_spec(_hill_surface(0.2, -0.1, [(-0.5, 0.2)]))
    (pose,) = build_initial_poses(_drone(), terrain, num_uavs=1, spawn_radius=0.0)
    assert pose.velocity is None


def test_wall_contact_spawn_rolls_up_the_wall():
    drone = {
        "initial_position": [0.0, 0.0, 0.5],
        "initial_spawn": {"mode": "wall_contact", "forward_speed": [0.3, -0.2]},
        "wheels": {"offset_y": 0.2, "radius": 0.15},
    }
    environment = {"wall_position": [3.0, 0.0, 3.5], "wall_size": [0.2, 2.5, 3.5]}
    plane = build_surface_model_spec(
        {"surface": {"mode": "plane", "solref": [0.01, 1.0], "contact": {"contype": 1, "conaffinity": 1}, "plane": {"size": [3.0, 3.0, 0.1]}}}
    )

    climbing, descending = build_initial_poses(drone, plane, num_uavs=2, spawn_radius=1.0, environment=environment)

    np.testing.assert_allclose(climbing.velocity.linear, [0.0, 0.0, 0.3], atol=1e-12)
    np.testing.assert_allclose(descending.velocity.linear, [0.0, 0.0, -0.2], atol=1e-12)
    assert math.isclose(climbing.velocity.left_wheel_rate, 0.3 / 0.15)
    assert math.isclose(climbing.velocity.right_wheel_rate, 0.3 / 0.15)


def test_spawn_velocity_options_are_validated():
    terrain = build_surface_model_spec(_hill_surface(0.2, -0.1, [(-0.5, 0.2)]))
    with pytest.raises(ValueError, match="one value per UAV"):
        build_initial_poses(_drone(forward_speed=[0.1]), terrain, num_uavs=2, spawn_radius=0.0)
    with pytest.raises(ValueError, match="finite"):
        build_initial_poses(_drone(turn_rate=float("nan")), terrain, num_uavs=1, spawn_radius=0.0)
    wall_drone = {
        "initial_position": [0.0, 0.0, 0.5],
        "initial_spawn": {"mode": "wall_contact", "yaw_deg": 10.0},
        "wheels": {"offset_y": 0.2, "radius": 0.15},
    }
    with pytest.raises(ValueError, match="yaw_deg"):
        build_initial_poses(wall_drone, terrain, num_uavs=1, spawn_radius=0.0, environment={"wall_position": [3.0, 0.0, 3.5], "wall_size": [0.2, 2.5, 3.5]})


def _write_params(tmp_path, initial_spawn: dict, **simulation) -> str:
    params = copy.deepcopy(load_vehicle_params())
    params["drone"]["initial_spawn"] = initial_spawn
    params["simulation"].update(simulation)
    params_path = tmp_path / "vehicle_params.json"
    params_path.write_text(json.dumps(params), encoding="utf-8")
    return str(params_path)


class _SilentSocket:
    def recvfrom(self, _buffer_size: int):
        raise BlockingIOError()

    def sendto(self, _payload: bytes, _endpoint) -> None:
        pass


def test_scene_starts_rolling_without_slip(tmp_path):
    params_path = _write_params(tmp_path, {"mode": "explicit", "positions_xy": [[0.0, 0.0]], "forward_speed": 0.5, "turn_rate": 0.4})
    scene = load_simulation_scene(SimulationRequest(params_path=params_path, generated_xml_dir=tmp_path))
    model, data = scene.model, scene.data
    velocity = scene.uav_specs[0].initial_velocity

    layout = scene.sensor_layouts[0]
    np.testing.assert_allclose(data.sensordata[layout.linear_velocity], velocity.linear, atol=1e-12)
    np.testing.assert_allclose(data.sensordata[layout.angular_velocity], velocity.angular, atol=1e-12)
    # The default floor is the plane z = 0: each wheel's lowest point is at rest.
    for wheel_body in ("left_wheel", "right_wheel"):
        spatial_velocity = np.zeros(6)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, model.body(wheel_body).id, spatial_velocity, 0)
        contact_velocity = spatial_velocity[3:] + np.cross(spatial_velocity[:3], [0.0, 0.0, -0.15])
        np.testing.assert_allclose(contact_velocity, 0.0, atol=1e-12)


def test_hold_keeps_the_spawn_velocity_until_the_first_command(tmp_path):
    params_path = _write_params(
        tmp_path, {"mode": "explicit", "positions_xy": [[0.0, 0.0]], "forward_speed": 0.5}, hold_until_first_command=True
    )
    scene = load_simulation_scene(
        SimulationRequest(params_path=params_path, generated_xml_dir=tmp_path, pacing_mode="accelerated", duration_seconds=0.02)
    )
    spawn_qpos = scene.data.qpos.copy()
    spawn_qvel = scene.data.qvel.copy()
    assert np.any(spawn_qvel != 0.0)

    run_headless_loop(scene, _SilentSocket())

    assert scene.data.time >= 0.02
    np.testing.assert_array_equal(scene.data.qpos, spawn_qpos)
    np.testing.assert_array_equal(scene.data.qvel, spawn_qvel)
