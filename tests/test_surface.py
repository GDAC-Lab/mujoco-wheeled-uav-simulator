from __future__ import annotations

import math
import unittest

import numpy as np

from wheeled_uav.model.builder import build_xml_replacements, render_model_xml
from wheeled_uav.model.poses import (
    build_initial_poses,
    build_surface_model_spec,
    build_surface_spawn_positions,
)
from wheeled_uav.model.surface import (
    build_surface_blocks,
    build_surface_evaluator,
    evaluate_height_function,
    evaluate_height_gradient,
    evaluate_surface_normal,
    get_surface_config,
    surface_can_use_plane_geom,
)


class SurfaceBehaviorTests(unittest.TestCase):
    def test_get_surface_config_maps_slope_mode_to_height_function(self) -> None:
        config = get_surface_config(
            {
                "surface": {
                    "mode": "slope",
                    "solref": [0.01, 1.0],
                    "contact": {"contype": 1, "conaffinity": 1},
                    "height_function": {"parameters": {"slope_x": 0.2, "slope_y": -0.1}},
                }
            }
        )

        self.assertEqual(config["type"], "height_function")
        self.assertEqual(config["height_function"]["name"], "slope")

    def test_slope_surface_returns_expected_height_gradient_and_normal(self) -> None:
        evaluator = build_surface_evaluator(
            {
                "type": "height_function",
                "height_function": {
                    "name": "slope",
                    "parameters": {"z_offset": 0.5, "slope_x": 0.2, "slope_y": -0.1},
                },
            }
        )

        self.assertIsNotNone(evaluator)
        assert evaluator is not None
        self.assertAlmostEqual(evaluate_height_function(evaluator, 2.0, 3.0), 0.6)
        self.assertEqual(evaluate_height_gradient(evaluator, 2.0, 3.0), (0.2, -0.1))
        normal = evaluate_surface_normal(evaluator, 2.0, 3.0)
        expected = np.array([-0.2, 0.1, 1.0], dtype=float)
        expected /= np.linalg.norm(expected)
        self.assertTrue(np.allclose(normal, expected))

    def test_gaussian_surface_uses_hfield_when_not_planar(self) -> None:
        asset_block, geom_block, evaluator = build_surface_blocks(
            {
                "type": "height_function",
                "material": "floor_mat",
                "solref": [0.01, 1.0],
                "contact": {"contype": 1, "conaffinity": 1},
                "height_function": {
                    "name": "gaussian",
                    "x_range": [-1.0, 1.0],
                    "y_range": [-1.0, 1.0],
                    "grid_resolution": [5, 5],
                    "parameters": {"amplitude": 0.4, "sigma_x": 0.5, "sigma_y": 0.5},
                },
            }
        )

        self.assertIn("<hfield", asset_block)
        self.assertIn('type="hfield"', geom_block)
        self.assertIsNotNone(evaluator)

    def test_flat_surface_can_use_plane_geom(self) -> None:
        evaluator = build_surface_evaluator(
            {
                "type": "height_function",
                "height_function": {
                    "name": "flat",
                    "parameters": {"z_offset": math.pi},
                },
            }
        )

        assert evaluator is not None
        self.assertTrue(surface_can_use_plane_geom(evaluator))

    def test_surface_builder_returns_surface_spec(self) -> None:
        surface_spec = build_surface_model_spec(
            {
                "surface": {
                    "mode": "plane",
                    "solref": [0.01, 1.0],
                    "contact": {"contype": 1, "conaffinity": 1},
                    "plane": {"size": [3.0, 3.0, 0.1]},
                }
            }
        )

        self.assertEqual(surface_spec.config["type"], "plane")
        self.assertIn('type="plane"', surface_spec.geom_block)
        self.assertIsNone(surface_spec.evaluator)

    def test_surface_spawn_positions_are_distributed_on_circle(self) -> None:
        positions = build_surface_spawn_positions([0.0, 0.0, 1.5], num_uavs=4, spawn_radius=2.0)

        self.assertEqual(len(positions), 4)
        self.assertEqual(positions[0], [2.0, 0.0, 1.5])
        self.assertTrue(np.allclose(positions[1], [0.0, 2.0, 1.5], atol=1.0e-9))

    def test_build_initial_poses_uses_surface_spec(self) -> None:
        drone = {
            "initial_position": [0.0, 0.0, 0.3],
            "wheels": {"offset_y": 0.2, "radius": 0.1},
        }
        surface_spec = build_surface_model_spec(
            {
                "surface": {
                    "mode": "flat",
                    "solref": [0.01, 1.0],
                    "contact": {"contype": 1, "conaffinity": 1},
                    "follow_surface_for_initial_position": False,
                    "height_function": {
                        "x_range": [-1.0, 1.0],
                        "y_range": [-1.0, 1.0],
                        "grid_resolution": [5, 5],
                        "parameters": {"z_offset": 0.0},
                    },
                }
            }
        )

        poses = build_initial_poses(drone, surface_spec, num_uavs=2, spawn_radius=1.0)

        self.assertEqual(len(poses), 2)
        self.assertEqual(poses[0].position, [1.0, 0.0, 0.3])
        self.assertTrue(np.allclose(poses[1].position, [-1.0, 0.0, 0.3], atol=1.0e-9))

    def test_build_initial_poses_supports_wall_contact_spawn(self) -> None:
        drone = {
            "initial_position": [0.0, 0.0, 0.5],
            "initial_spawn": {
                "mode": "wall_contact",
                "b2_body": [0.0, 1.0, 0.0],
                "b3_body": [0.640, 0.0, 0.768],
            },
            "wheels": {"offset_y": 0.2, "radius": 0.15},
        }
        environment = {
            "wall_position": [3.0, 0.0, 3.5],
            "wall_size": [0.2, 2.5, 3.5],
        }
        surface_spec = build_surface_model_spec(
            {
                "surface": {
                    "mode": "plane",
                    "solref": [0.01, 1.0],
                    "contact": {"contype": 1, "conaffinity": 1},
                    "follow_surface_for_initial_position": False,
                    "plane": {"size": [3.0, 3.0, 0.1]},
                }
            }
        )

        poses = build_initial_poses(drone, surface_spec, num_uavs=1, spawn_radius=1.0, environment=environment)

        self.assertEqual(len(poses), 1)
        # wall face (2.8) - wheel radius (0.15) - default 5 mm spawn clearance
        self.assertAlmostEqual(poses[0].position[0], 2.645, places=3)
        self.assertAlmostEqual(poses[0].position[1], 0.0)
        self.assertAlmostEqual(poses[0].position[2], 0.5)
        self.assertIsNotNone(poses[0].quaternion)
        assert poses[0].quaternion is not None
        self.assertAlmostEqual(sum(component * component for component in poses[0].quaternion), 1.0, places=6)

        drone["initial_spawn"]["wall_clearance_m"] = 0.0
        touching_poses = build_initial_poses(drone, surface_spec, num_uavs=1, spawn_radius=1.0, environment=environment)
        self.assertAlmostEqual(touching_poses[0].position[0], 2.650, places=3)

    def test_composite_surface_matches_analytic_height_and_gradient(self) -> None:
        # Composite surface: h = 0.2x - 0.2y + 0.5 exp(-((x+1)^2+(y-0.2)^2)/0.3)
        #                                     - 0.5 exp(-(x^2+y^2)/0.6)
        evaluator = build_surface_evaluator(
            {
                "type": "height_function",
                "height_function": {
                    "name": "composite",
                    "parameters": {
                        "z_offset": 0.0,
                        "slope_x": 0.2,
                        "slope_y": -0.2,
                        "gauss1_amplitude": 0.5,
                        "gauss1_center_x": -1.0,
                        "gauss1_center_y": 0.2,
                        "gauss1_denominator": 0.3,
                        "gauss2_amplitude": -0.5,
                        "gauss2_center_x": 0.0,
                        "gauss2_center_y": 0.0,
                        "gauss2_denominator": 0.6,
                    },
                },
            }
        )
        self.assertIsNotNone(evaluator)
        assert evaluator is not None

        def analytic_height(x: float, y: float) -> float:
            return (
                0.2 * x
                - 0.2 * y
                + 0.5 * math.exp(-((x + 1.0) ** 2 + (y - 0.2) ** 2) / 0.3)
                - 0.5 * math.exp(-(x**2 + y**2) / 0.6)
            )

        finite_difference_step = 1.0e-6
        for x_coord, y_coord in [(0.0, 0.0), (0.5, -0.3), (-1.0, 0.2), (1.2, 1.1)]:
            self.assertAlmostEqual(
                evaluate_height_function(evaluator, x_coord, y_coord),
                analytic_height(x_coord, y_coord),
                places=12,
            )
            dh_dx, dh_dy = evaluate_height_gradient(evaluator, x_coord, y_coord)
            fd_dx = (
                analytic_height(x_coord + finite_difference_step, y_coord)
                - analytic_height(x_coord - finite_difference_step, y_coord)
            ) / (2.0 * finite_difference_step)
            fd_dy = (
                analytic_height(x_coord, y_coord + finite_difference_step)
                - analytic_height(x_coord, y_coord - finite_difference_step)
            ) / (2.0 * finite_difference_step)
            self.assertAlmostEqual(dh_dx, fd_dx, places=6)
            self.assertAlmostEqual(dh_dy, fd_dy, places=6)

    def test_composite_surface_uses_hfield(self) -> None:
        asset_block, geom_block, evaluator = build_surface_blocks(
            {
                "type": "height_function",
                "material": "floor_mat",
                "solref": [0.01, 1.0],
                "contact": {"contype": 1, "conaffinity": 1},
                "height_function": {
                    "name": "composite",
                    "x_range": [-2.0, 2.0],
                    "y_range": [-2.0, 2.0],
                    "grid_resolution": [21, 21],
                    "parameters": {
                        "slope_x": 0.2,
                        "slope_y": -0.2,
                        "gauss1_amplitude": 0.5,
                        "gauss1_center_x": -1.0,
                        "gauss1_center_y": 0.2,
                        "gauss1_denominator": 0.3,
                    },
                },
            }
        )
        self.assertIsNotNone(evaluator)
        assert evaluator is not None
        self.assertIn("hfield", asset_block)
        self.assertIn('type="hfield"', geom_block)
        self.assertFalse(surface_can_use_plane_geom(evaluator))

    def test_composite_surface_rejects_nonpositive_denominator(self) -> None:
        evaluator = build_surface_evaluator(
            {
                "type": "height_function",
                "height_function": {
                    "name": "composite",
                    "parameters": {"gauss1_amplitude": 0.5, "gauss1_denominator": 0.0},
                },
            }
        )
        assert evaluator is not None
        with self.assertRaises(ValueError):
            evaluate_height_function(evaluator, 0.0, 0.0)

    def test_build_initial_poses_supports_explicit_spawn(self) -> None:
        drone = {
            "initial_position": [0.0, 0.0, 0.3],
            "initial_spawn": {
                "mode": "explicit",
                "positions_xy": [[0.0, -0.6], [0.0, 0.0], [0.0, 0.6]],
            },
            "wheels": {"offset_y": 0.2, "radius": 0.15},
        }
        surface_spec = build_surface_model_spec(
            {
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
                        "parameters": {
                            "slope_x": 0.2,
                            "slope_y": -0.2,
                            "gauss1_amplitude": 0.5,
                            "gauss1_center_x": -1.0,
                            "gauss1_center_y": 0.2,
                            "gauss1_denominator": 0.3,
                            "gauss2_amplitude": -0.5,
                            "gauss2_center_x": 0.0,
                            "gauss2_center_y": 0.0,
                            "gauss2_denominator": 0.6,
                        },
                    },
                }
            }
        )

        poses = build_initial_poses(drone, surface_spec, num_uavs=3, spawn_radius=1.5)

        self.assertEqual(len(poses), 3)
        for pose, expected_y in zip(poses, [-0.6, 0.0, 0.6], strict=True):
            self.assertAlmostEqual(pose.position[0], 0.0)
            self.assertAlmostEqual(pose.position[1], expected_y)
            self.assertIsNotNone(pose.quaternion)

        with self.assertRaises(ValueError):
            build_initial_poses(drone, surface_spec, num_uavs=4, spawn_radius=1.5)

    def test_build_wall_guide_block_emits_visual_capsules(self) -> None:
        from pathlib import Path

        import mujoco

        from wheeled_uav.config import load_vehicle_params

        project_params = (
            Path(__file__).resolve().parents[3] / "configs" / "vehicle" / "vehicle_params.project.json"
        )
        if not project_params.is_file():
            self.skipTest("project vehicle params not available")

        params = load_vehicle_params(params_path=project_params)
        replacements, _, _ = build_xml_replacements(params)
        guide_block = replacements["__WALL_GUIDE_BLOCK__"]
        self.assertIn('name="wall_guide"', guide_block)
        self.assertIn('type="capsule"', guide_block)
        self.assertIn("wall_guide_seg_", guide_block)
        self.assertNotIn("wall_guide_marker_", guide_block)

        generated_xml_dir = Path(__file__).resolve().parents[3] / "build" / "generated_xml"
        generated_xml_dir.mkdir(parents=True, exist_ok=True)
        model_xml_path, _, _ = render_model_xml(
            params,
            output_path=generated_xml_dir / "wall_guide_check.generated.xml",
            generated_xml_dir=generated_xml_dir,
        )
        model = mujoco.MjModel.from_xml_path(str(model_xml_path))
        self.assertGreater(model.ngeom, 0)

    def test_parametric_wall_guide_matches_sinusoidal_ramp(self) -> None:
        from wheeled_uav.model.builder import _resolve_wall_guide_waypoints

        environment = {
            "wall_position": [3.0, 0.0, 5.0],
            "wall_size": [0.2, 5.0, 8.0],
        }
        guide = {
            "surface_offset": 0.025,
            "curve": {
                "type": "sinusoidal_ramp",
                "y_amplitude_m": 0.5,
                "y_angular_frequency_rad_s": 0.5,
                "z_rate_m_s": 0.5,
                "t_start_s": 0.0,
                "t_end_s": 2.0,
                "sample_count": 5,
            },
        }
        waypoints = _resolve_wall_guide_waypoints(environment, guide)
        self.assertEqual(len(waypoints), 5)
        self.assertAlmostEqual(waypoints[0][1], 0.0)
        self.assertAlmostEqual(waypoints[0][2], 0.0)
        self.assertAlmostEqual(waypoints[-1][2], 1.0)
        self.assertAlmostEqual(waypoints[2][1], 0.5 * math.sin(0.5 * 1.0), places=6)


if __name__ == "__main__":
    unittest.main()