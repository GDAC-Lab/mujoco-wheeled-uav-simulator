from __future__ import annotations

import json
import unittest
from unittest.mock import patch

import numpy as np

from wheeled_uav.controllers.hover import (
    ControllerRuntimeStats,
    _apply_force_safety_clamps,
    _build_state_packet_metrics,
    _clamp_position_error,
    _read_latest_state,
    _update_runtime_stats_for_state,
    build_hover_command_payload,
    build_hover_command_payload_with_metadata,
    build_hover_controller_config,
    compute_hover_control,
)
from wheeled_uav.timing import StateSampleTracker, build_state_sample_key


class _FakeSocket:
    def __init__(self, *payloads: dict[str, object]):
        self._queue = [json.dumps(payload).encode("utf-8") for payload in payloads]

    def recvfrom(self, _buffer_size: int) -> tuple[bytes, tuple[str, int]]:
        if not self._queue:
            raise BlockingIOError()
        return self._queue.pop(0), ("127.0.0.1", 5001)


class PythonHoverControllerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.params = {
            "simulation": {"gravity": [0.0, 0.0, -9.81]},
            "actuation": {
                "command_mode": "omega",
                "max_rotor_thrust": 20.0,
                "yaw_moment_ratio": 0.02,
                "thrust_coefficient": 2.0e-5,
            },
            "drone": {
                "arm": {"x": 0.08, "y": 0.1},
                "body_box": {"mass": 0.9},
                "wheels": {"mass": 0.1},
            },
            "controller": {
                "desired_heading": [1.0, 0.0, 0.0],
                "position_gain": [3.0, 3.0, 6.0],
                "velocity_gain": [2.2, 2.2, 4.0],
                "attitude_gain": [0.8, 0.8, 0.25],
                "angular_velocity_gain": [0.12, 0.12, 0.08],
            },
        }
        self.config = build_hover_controller_config(self.params)
        self.nominal_state = {
            "position": [0.0, 0.0, 1.5],
            "velocity": [0.0, 0.0, 0.0],
            "angular_velocity_body": [0.0, 0.0, 0.0],
            "rotation_matrix": [
                1.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                1.0,
            ],
        }

    def test_compute_hover_control_balances_nominal_hover(self) -> None:
        rotor_thrusts = compute_hover_control(self.nominal_state, np.array([0.0, 0.0, 1.5]), self.config)

        expected_rotor_thrust = self.config.mass * self.config.gravity / 4.0
        np.testing.assert_allclose(rotor_thrusts, np.full(4, expected_rotor_thrust), rtol=1.0e-6, atol=1.0e-6)

    def test_compute_hover_control_clips_to_max_rotor_thrust(self) -> None:
        # Disable the position-error clamp so the far target actually saturates the rotors.
        unclamped_config = build_hover_controller_config(
            {
                **self.params,
                "controller": {
                    **self.params["controller"],
                    "position_error_limit_m": 0.0,
                },
            }
        )
        aggressive_target = np.array([0.0, 0.0, 20.0])

        rotor_thrusts = compute_hover_control(self.nominal_state, aggressive_target, unclamped_config)

        self.assertTrue(np.all(rotor_thrusts <= unclamped_config.max_rotor_thrust))
        self.assertTrue(np.any(np.isclose(rotor_thrusts, unclamped_config.max_rotor_thrust)))

    def test_config_parses_safety_clamp_defaults_and_overrides(self) -> None:
        self.assertAlmostEqual(self.config.position_error_limit_m, 1.5)
        self.assertAlmostEqual(self.config.max_tilt_deg, 35.0)

        overridden_config = build_hover_controller_config(
            {
                **self.params,
                "controller": {
                    **self.params["controller"],
                    "position_error_limit_m": 2.5,
                    "max_tilt_deg": 20.0,
                },
            }
        )
        self.assertAlmostEqual(overridden_config.position_error_limit_m, 2.5)
        self.assertAlmostEqual(overridden_config.max_tilt_deg, 20.0)

    def test_clamp_position_error_saturates_norm(self) -> None:
        clamped_error = _clamp_position_error(np.array([3.0, 4.0, 0.0]), 1.0)
        np.testing.assert_allclose(clamped_error, [0.6, 0.8, 0.0], rtol=1.0e-12)

        small_error = np.array([0.2, 0.0, -0.1])
        np.testing.assert_allclose(_clamp_position_error(small_error, 1.0), small_error)
        # <= 0 disables the clamp entirely.
        big_error = np.array([30.0, 0.0, 0.0])
        np.testing.assert_allclose(_clamp_position_error(big_error, 0.0), big_error)

    def test_apply_force_safety_clamps_floors_vertical_and_caps_tilt(self) -> None:
        clamped_force = _apply_force_safety_clamps(np.array([10.0, 0.0, -5.0]), 2.0, 35.0)

        self.assertAlmostEqual(clamped_force[2], 2.0)
        tilt_deg = np.degrees(np.arctan2(np.linalg.norm(clamped_force[:2]), clamped_force[2]))
        self.assertAlmostEqual(tilt_deg, 35.0, places=9)

        # Disabled clamp passes any force through, even a downward one.
        raw_force = np.array([10.0, 0.0, -5.0])
        np.testing.assert_allclose(_apply_force_safety_clamps(raw_force, 2.0, 0.0), raw_force)

    def test_apply_force_safety_clamps_treats_tilt_at_or_over_90_as_uncapped(self) -> None:
        # Regression: tan(>90 deg) is negative and used to FLIP and amplify the
        # lateral force. >= 90 must keep only the vertical floor.
        clamped_force = _apply_force_safety_clamps(np.array([10.0, 0.0, -5.0]), 2.0, 120.0)
        np.testing.assert_allclose(clamped_force, [10.0, 0.0, 2.0])

        clamped_force = _apply_force_safety_clamps(np.array([10.0, 0.0, -5.0]), 2.0, 90.0)
        np.testing.assert_allclose(clamped_force, [10.0, 0.0, 2.0])

    def test_far_target_commands_same_thrust_as_moderately_far_target(self) -> None:
        # Both offsets exceed position_error_limit_m, so the commanded pull is identical:
        # a vehicle dragged 300 m away recovers exactly as gently as one 3 m away.
        moderate_thrusts = compute_hover_control(self.nominal_state, np.array([3.0, 0.0, 1.5]), self.config)
        far_thrusts = compute_hover_control(self.nominal_state, np.array([300.0, 0.0, 1.5]), self.config)

        np.testing.assert_allclose(far_thrusts, moderate_thrusts, rtol=1.0e-12, atol=1.0e-12)

    def test_recovers_upright_when_far_above_target_and_climbing(self) -> None:
        # Without the clamps the raw PD demands a downward force here and the
        # desired attitude flips; with them the command is a gentle upright descent.
        runaway_state = {
            **self.nominal_state,
            "position": [0.0, 0.0, 10.0],
            "velocity": [0.0, 0.0, 3.0],
        }

        rotor_thrusts = compute_hover_control(runaway_state, np.array([0.0, 0.0, 1.5]), self.config)

        expected_rotor_thrust = 0.25 * self.config.mass * self.config.gravity / 4.0
        np.testing.assert_allclose(rotor_thrusts, np.full(4, expected_rotor_thrust), rtol=1.0e-9, atol=1.0e-9)

    def test_build_hover_command_payload_emits_rotor_omega_for_omega_mode(self) -> None:
        rotor_thrusts = np.array([2.0, 8.0, 18.0, 0.0])

        payload = build_hover_command_payload(rotor_thrusts, self.config)

        decoded_payload = json.loads(payload.decode("utf-8"))
        self.assertNotIn("rotor_thrusts", decoded_payload)
        np.testing.assert_allclose(
            decoded_payload["rotor_omega"],
            np.sqrt(rotor_thrusts / self.config.thrust_coefficient),
            rtol=1.0e-9,
            atol=1.0e-9,
        )

    def test_build_hover_command_payload_emits_rotor_thrusts_for_thrust_mode(self) -> None:
        thrust_mode_config = build_hover_controller_config(
            {
                **self.params,
                "actuation": {
                    **self.params["actuation"],
                    "command_mode": "thrust",
                },
            }
        )
        rotor_thrusts = np.array([1.0, 2.0, 3.0, 4.0])

        payload = build_hover_command_payload(rotor_thrusts, thrust_mode_config)

        decoded_payload = json.loads(payload.decode("utf-8"))
        self.assertEqual(decoded_payload, {"rotor_thrusts": [1.0, 2.0, 3.0, 4.0]})

    def test_build_hover_command_payload_can_include_packet_metadata(self) -> None:
        payload = build_hover_command_payload_with_metadata(
            np.array([1.0, 2.0, 3.0, 4.0]),
            self.config,
            sequence=7,
            source_state_sequence=6,
            wall_time_send_ns=123456789,
            fidelity_mode="hil",
        )

        decoded_payload = json.loads(payload.decode("utf-8"))
        self.assertEqual(decoded_payload["protocol_version"], 2)
        self.assertEqual(decoded_payload["sequence"], 7)
        self.assertEqual(decoded_payload["source_state_sequence"], 6)
        self.assertEqual(decoded_payload["wall_time_send_ns"], 123456789)
        self.assertEqual(decoded_payload["fidelity_mode"], "hil")
        self.assertIn("rotor_omega", decoded_payload)

    def test_build_state_packet_metrics_extracts_age_and_sequence(self) -> None:
        metrics = _build_state_packet_metrics(
            {
                "protocol_version": 2,
                "sequence": 11,
                "wall_time_send_ns": 1_500_000,
                "fidelity_mode": "hil",
            },
            receive_time_ns=4_000_000,
        )

        self.assertEqual(metrics.protocol_version, 2)
        self.assertEqual(metrics.sequence, 11)
        self.assertEqual(metrics.fidelity_mode, "hil")
        self.assertAlmostEqual(metrics.age_ms or 0.0, 2.5)

    def test_update_runtime_stats_tracks_state_sequence_gap(self) -> None:
        stats = ControllerRuntimeStats(last_state_sequence=3)

        _update_runtime_stats_for_state(
            stats,
            _build_state_packet_metrics({"sequence": 6, "wall_time_send_ns": 1_000_000}, receive_time_ns=2_000_000),
            compute_time_ms=0.125,
        )

        self.assertEqual(stats.last_state_sequence, 6)
        self.assertEqual(stats.last_state_sequence_gap, 2)
        self.assertEqual(stats.state_sequence_gap_count, 2)
        self.assertAlmostEqual(stats.last_state_age_ms or 0.0, 1.0)
        self.assertAlmostEqual(stats.last_controller_compute_ms or 0.0, 0.125)

    def test_read_latest_state_returns_latest_packet_and_metrics(self) -> None:
        sock = _FakeSocket(
            {"sequence": 1, "time": 0.1, **self.nominal_state},
            {"protocol_version": 2, "sequence": 4, "wall_time_send_ns": 10, "time": 0.2, **self.nominal_state},
        )

        with patch("wheeled_uav.controllers.hover.time.time_ns", return_value=2_000_010):
            state_packet = _read_latest_state(sock)

        self.assertIsNotNone(state_packet)
        assert state_packet is not None
        state, metrics = state_packet
        self.assertEqual(state["time"], 0.2)
        self.assertEqual(metrics.sequence, 4)
        self.assertAlmostEqual(metrics.age_ms or 0.0, 2.0)

    def test_state_sample_tracker_matches_matlab_key_format(self) -> None:
        key = build_state_sample_key({"sequence": 7, "time": 0.12345})
        tracker = StateSampleTracker()
        self.assertTrue(tracker.is_new({"sequence": 7, "time": 0.12345}))
        self.assertFalse(tracker.is_new({"sequence": 7, "time": 0.12345}))
        self.assertNotEqual(key, "")

    def test_build_hover_controller_config_uses_matlab_aligned_defaults(self) -> None:
        config = build_hover_controller_config(
            {
                "simulation": self.params["simulation"],
                "actuation": self.params["actuation"],
                "drone": self.params["drone"],
            }
        )

        np.testing.assert_allclose(config.desired_heading, np.array([1.0, 0.0, 0.0]))
        np.testing.assert_allclose(config.position_gain, np.array([3.0, 3.0, 6.0]))
        np.testing.assert_allclose(config.velocity_gain, np.array([2.2, 2.2, 4.0]))
        np.testing.assert_allclose(config.attitude_gain, np.array([0.8, 0.8, 0.25]))
        np.testing.assert_allclose(config.angular_velocity_gain, np.array([0.12, 0.12, 0.08]))

    def test_mixer_follows_rotor_geometry_regardless_of_json_order(self) -> None:
        # Rotors listed in a non-standard order (fr, bl, fl, br): the mixer columns
        # must follow the JSON/actuator ordering, not an assumed fr, fl, br, bl layout.
        reordered_params = {
            **self.params,
            "actuation": {
                **self.params["actuation"],
                "rotors": [
                    {
                        "name": "fr",
                        "position_body": [0.08, -0.1, 0.0125],
                        "thrust_axis_body": [0.0, 0.0, 1.0],
                        "spin_sign": 1,
                    },
                    {
                        "name": "bl",
                        "position_body": [-0.08, 0.1, 0.0125],
                        "thrust_axis_body": [0.0, 0.0, 1.0],
                        "spin_sign": 1,
                    },
                    {
                        "name": "fl",
                        "position_body": [0.08, 0.1, 0.0125],
                        "thrust_axis_body": [0.0, 0.0, 1.0],
                        "spin_sign": -1,
                    },
                    {
                        "name": "br",
                        "position_body": [-0.08, -0.1, 0.0125],
                        "thrust_axis_body": [0.0, 0.0, 1.0],
                        "spin_sign": -1,
                    },
                ],
            },
        }

        config = build_hover_controller_config(reordered_params)

        arm_x = 0.08
        arm_y = 0.1
        kappa = 0.02
        expected_allocation = np.array(
            [
                [1.0, 1.0, 1.0, 1.0],
                [-arm_y, arm_y, arm_y, -arm_y],
                [-arm_x, arm_x, -arm_x, arm_x],
                [kappa, kappa, -kappa, -kappa],
            ]
        )
        np.testing.assert_allclose(config.mixer, np.linalg.pinv(expected_allocation), rtol=1.0e-9, atol=1.0e-12)

    def test_mixer_supports_tilted_rotors(self) -> None:
        tilted_axis = [-0.14834, 0.197905, 0.968912]
        tilted_params = {
            **self.params,
            "actuation": {
                **self.params["actuation"],
                "rotors": [
                    {
                        "name": "fr",
                        "position_body": [0.08, -0.1, 0.0125],
                        "thrust_axis_body": tilted_axis,
                        "spin_sign": 1,
                    },
                    {
                        "name": "fl",
                        "position_body": [0.08, 0.1, 0.0125],
                        "thrust_axis_body": [0.0, 0.0, 1.0],
                        "spin_sign": -1,
                    },
                    {
                        "name": "br",
                        "position_body": [-0.08, -0.1, 0.0125],
                        "thrust_axis_body": [0.0, 0.0, 1.0],
                        "spin_sign": -1,
                    },
                    {
                        "name": "bl",
                        "position_body": [-0.08, 0.1, 0.0125],
                        "thrust_axis_body": [0.0, 0.0, 1.0],
                        "spin_sign": 1,
                    },
                ],
            },
        }

        config = build_hover_controller_config(tilted_params)

        axis = np.asarray(tilted_axis) / np.linalg.norm(tilted_axis)
        position = np.asarray([0.08, -0.1, 0.0125])
        expected_first_column = np.concatenate(([axis[2]], np.cross(position, axis) + 0.02 * axis))
        allocation = np.linalg.pinv(config.mixer)
        np.testing.assert_allclose(allocation[:, 0], expected_first_column, rtol=1.0e-6, atol=1.0e-9)


if __name__ == "__main__":
    unittest.main()