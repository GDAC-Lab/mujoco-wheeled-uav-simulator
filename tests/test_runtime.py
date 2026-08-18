from __future__ import annotations

import json
import unittest
from unittest.mock import patch

import numpy as np

from wheeled_uav.protocol import build_packet_metadata
from wheeled_uav.runtime.command_dispatcher import ControlCommandDispatcher
from wheeled_uav.runtime.fidelity import (
    ActuatorSnapshot,
    apply_actuator_dynamics_step,
    apply_sensor_fidelity,
)
from wheeled_uav.runtime.state_publisher import StatePayloadPublisher, build_state_payload
from wheeled_uav.types import (
    ActuatorDynamicsConfig,
    FidelityConfig,
    LoggingConfig,
    NetworkFidelityConfig,
    SensorFidelityConfig,
    SensorLayout,
    SensorNames,
    UAVModelSpec,
)


class _FakeSocket:
    def __init__(self, *payloads: dict[str, object]):
        self._queue = [json.dumps(payload).encode("utf-8") for payload in payloads]
        self.sent_packets: list[tuple[bytes, tuple[str, int]]] = []

    def recvfrom(self, _buffer_size: int) -> tuple[bytes, tuple[str, int]]:
        if not self._queue:
            raise BlockingIOError()
        return self._queue.pop(0), ("127.0.0.1", 5000)

    def sendto(self, payload: bytes, endpoint: tuple[str, int]) -> None:
        self.sent_packets.append((payload, endpoint))


class _FakeData:
    def __init__(self, nu: int, nbody: int = 3):
        self.ctrl = np.zeros(nu, dtype=float)
        self.xfrc_applied = np.zeros((nbody, 6), dtype=float)


class _FakeSensorData:
    def __init__(self) -> None:
        self.time = 1.25
        self.sensordata = np.array(
            [
                1.0, 2.0, 3.0,
                0.1, 0.2, 0.3,
                0.01, 0.02, 0.03,
                1.0, 0.0, 0.0,
                0.0, 1.0, 0.0,
                0.0, 0.0, 1.0,
            ],
            dtype=float,
        )


class _DummyRequest:
    def __init__(self, fidelity_mode: str):
        self.instance_id = 0
        self.fidelity_mode = fidelity_mode


class _DummyScene:
    """Duck-typed stand-in for runtime.scene.SimulationScene."""

    def __init__(
        self,
        fidelity: FidelityConfig,
        *,
        sensor_layouts: list[SensorLayout] | None = None,
        uav_specs: list[UAVModelSpec] | None = None,
    ):
        self.fidelity = fidelity
        self.request = _DummyRequest(fidelity.mode)
        self.model = object()
        self.data = _FakeSensorData()
        self.sensor_layouts = sensor_layouts or []
        self.surface_evaluator = None
        self.uav_specs = uav_specs or []
        self.geom_names = ()


_EMPTY_CONTACT_REPORT = {
    "count": 0,
    "total_force_magnitude": 0.0,
    "max_force_magnitude": 0.0,
    "total_normal_force": 0.0,
    "max_normal_force": 0.0,
    "left_wheel": {},
    "right_wheel": {},
    "surface": {},
    "wall": {},
    "contacts": [],
}


def _single_uav_layout() -> tuple[SensorLayout, UAVModelSpec]:
    sensor_layout = SensorLayout(
        position=slice(0, 3),
        linear_velocity=slice(3, 6),
        angular_velocity=slice(6, 9),
        x_axis=slice(9, 12),
        y_axis=slice(12, 15),
        z_axis=slice(15, 18),
    )
    uav_spec = UAVModelSpec(
        name="uav",
        body_name="uav_body",
        actuator_names=("m1", "m2", "m3", "m4"),
        sensor_names=SensorNames(
            position="pos",
            linear_velocity="vel",
            angular_velocity="gyro",
            x_axis="x",
            y_axis="y",
            z_axis="z",
        ),
        contact_prefix="wheel",
    )
    return sensor_layout, uav_spec


_DISPATCHER_PARAMS = {
    "actuation": {"thrust_coefficient": 2.0e-5},
    "drone": {"body_box": {"mass": 0.9}, "wheels": {"mass": 0.1}},
    "simulation": {"gravity": [0.0, 0.0, -9.81]},
}


class StatePayloadTests(unittest.TestCase):
    def test_build_state_payload_includes_logged_actuator_and_sensor_truth(self) -> None:
        sensor_layout, uav_spec = _single_uav_layout()
        scene = _DummyScene(
            FidelityConfig(
                mode="hil",
                logging=LoggingConfig(log_actuator_stats=True, log_sensor_truth=True),
                sensor_fidelity=SensorFidelityConfig(position_noise_std_m=0.01),
            ),
            sensor_layouts=[sensor_layout],
            uav_specs=[uav_spec],
        )
        actuator_snapshot = ActuatorSnapshot(
            requested_ctrl=np.array([4.0, 4.1, 4.2, 4.3]),
            applied_ctrl=np.array([3.5, 3.6, 3.7, 3.8]),
        )

        with patch("wheeled_uav.runtime.state_publisher.build_contact_report", return_value=_EMPTY_CONTACT_REPORT):
            payload = build_state_payload(
                scene,
                1.0,
                sequence=7,
                wall_time_send_ns=123,
                sensor_rng=np.random.default_rng(0),
                actuator_snapshot=actuator_snapshot,
            )

        decoded = json.loads(payload.decode("utf-8"))
        self.assertEqual(decoded["sequence"], 7)
        self.assertEqual(decoded["fidelity_mode"], "hil")
        self.assertIn("actuator", decoded)
        self.assertIn("sensor_truth", decoded)
        self.assertEqual(decoded["actuator"]["requested_rotor_thrusts"], [4.0, 4.1, 4.2, 4.3])
        self.assertEqual(decoded["actuator"]["applied_rotor_thrusts"], [3.5, 3.6, 3.7, 3.8])
        self.assertEqual(decoded["sensor_truth"]["position"], [1.0, 2.0, 3.0])
        self.assertNotEqual(decoded["position"], decoded["sensor_truth"]["position"])
        self.assertIn("wall", decoded["contact_summary"])

    def test_build_state_payload_omits_optional_fidelity_logs_by_default(self) -> None:
        sensor_layout, uav_spec = _single_uav_layout()
        scene = _DummyScene(
            FidelityConfig(),
            sensor_layouts=[sensor_layout],
            uav_specs=[uav_spec],
        )

        with patch("wheeled_uav.runtime.state_publisher.build_contact_report", return_value=_EMPTY_CONTACT_REPORT):
            payload = build_state_payload(
                scene,
                1.0,
                sequence=8,
                wall_time_send_ns=456,
                sensor_rng=np.random.default_rng(0),
                actuator_snapshot=ActuatorSnapshot(
                    requested_ctrl=np.array([4.0, 4.0, 4.0, 4.0]),
                    applied_ctrl=np.array([4.0, 4.0, 4.0, 4.0]),
                ),
            )

        decoded = json.loads(payload.decode("utf-8"))
        self.assertNotIn("actuator", decoded)
        self.assertNotIn("sensor_truth", decoded)

    def test_build_packet_metadata_uses_v2_defaults(self) -> None:
        metadata = build_packet_metadata(sequence=5, wall_time_send_ns=42, fidelity_mode="baseline")

        self.assertEqual(
            metadata,
            {
                "protocol_version": 2,
                "sequence": 5,
                "wall_time_send_ns": 42,
                "fidelity_mode": "baseline",
            },
        )

    def test_state_payload_publisher_delays_hil_packets_until_due(self) -> None:
        publisher = StatePayloadPublisher(
            _DummyScene(
                FidelityConfig(
                    mode="hil",
                    network=NetworkFidelityConfig(enabled=True, state_tx_latency_ms=5.0),
                )
            )
        )

        def build_payload(realtime_factor, sequence, wall_time_send_ns, actuator_snapshot, aero_snapshot=None, timing=None):
            self.assertIsNone(actuator_snapshot)
            self.assertIsNone(aero_snapshot)
            self.assertIsNone(timing)
            return f"seq={sequence}".encode("utf-8")

        publisher.build_payload = build_payload
        sock = _FakeSocket()

        with patch("wheeled_uav.runtime.state_publisher.time.time_ns", side_effect=[1_000_000, 7_000_000]):
            publisher.send_state(sock, "127.0.0.1", 5001, 1.0, None)
            publisher.send_state(sock, "127.0.0.1", 5001, 1.0, None)

        self.assertEqual(len(sock.sent_packets), 1)
        self.assertEqual(sock.sent_packets[0], (b"seq=1", ("127.0.0.1", 5001)))


class CommandDispatcherTests(unittest.TestCase):
    def test_tracks_sequence_gap_for_fresh_commands(self) -> None:
        dispatcher = ControlCommandDispatcher(_DISPATCHER_PARAMS, FidelityConfig(), num_uavs=1)
        data = _FakeData(4)

        dispatcher.apply_next_command(_FakeSocket({"sequence": 1, "rotor_thrusts": [1.0, 1.0, 1.0, 1.0]}), data)
        dispatcher.apply_next_command(_FakeSocket({"sequence": 4, "rotor_thrusts": [2.0, 2.0, 2.0, 2.0]}), data)

        np.testing.assert_allclose(data.ctrl, np.array([2.0, 2.0, 2.0, 2.0]))
        self.assertEqual(dispatcher.stats.last_sequence_gap, 2)
        self.assertEqual(dispatcher.stats.missed_command_updates, 2)
        self.assertEqual(dispatcher.stats.last_applied_policy, "fresh")

    def test_zeroes_stale_packet(self) -> None:
        dispatcher = ControlCommandDispatcher(
            _DISPATCHER_PARAMS,
            FidelityConfig(
                network=NetworkFidelityConfig(
                    stale_command_threshold_ms=5.0,
                    stale_command_policy="zero-thrust",
                )
            ),
            num_uavs=1,
        )
        data = _FakeData(4)
        data.ctrl[:] = [9.0, 9.0, 9.0, 9.0]

        dispatcher.apply_next_command(
            _FakeSocket(
                {
                    "sequence": 2,
                    "wall_time_send_ns": 0,
                    "rotor_thrusts": [1.0, 2.0, 3.0, 4.0],
                }
            ),
            data,
        )

        np.testing.assert_allclose(data.ctrl, np.zeros(4))
        self.assertEqual(dispatcher.stats.stale_command_count, 1)
        self.assertEqual(dispatcher.stats.stale_command_apply_count, 1)
        self.assertEqual(dispatcher.stats.last_applied_policy, "zero-thrust")

    def test_holds_last_command_after_timeout(self) -> None:
        dispatcher = ControlCommandDispatcher(
            _DISPATCHER_PARAMS,
            FidelityConfig(network=NetworkFidelityConfig(stale_command_threshold_ms=10.0)),
            num_uavs=1,
        )
        data = _FakeData(4)

        with patch("wheeled_uav.runtime.command_dispatcher.time.time_ns", return_value=1_000_000):
            dispatcher.apply_next_command(_FakeSocket({"sequence": 1, "rotor_thrusts": [1.0, 2.0, 3.0, 4.0]}), data)

        data.ctrl[:] = [0.0, 0.0, 0.0, 0.0]
        with patch("wheeled_uav.runtime.command_dispatcher.time.time_ns", return_value=20_000_000):
            dispatcher.apply_next_command(_FakeSocket(), data)

        np.testing.assert_allclose(data.ctrl, np.array([1.0, 2.0, 3.0, 4.0]))
        self.assertEqual(dispatcher.stats.command_timeout_count, 1)
        self.assertEqual(dispatcher.stats.stale_command_apply_count, 1)
        self.assertEqual(dispatcher.stats.last_applied_policy, "hold-last-command")

    def test_hold_last_command_keeps_last_fresh_command_on_stale_packet(self) -> None:
        # Regression: _last_command must not be overwritten by a stale packet,
        # or hold-last-command replays the stale command instead of holding.
        dispatcher = ControlCommandDispatcher(
            _DISPATCHER_PARAMS,
            FidelityConfig(network=NetworkFidelityConfig(stale_command_threshold_ms=5.0)),
            num_uavs=1,
        )
        data = _FakeData(4)

        with patch("wheeled_uav.runtime.command_dispatcher.time.time_ns", return_value=1_000_000):
            dispatcher.apply_next_command(
                _FakeSocket({"sequence": 1, "wall_time_send_ns": 1_000_000, "rotor_thrusts": [1.0, 1.0, 1.0, 1.0]}), data
            )
        np.testing.assert_allclose(data.ctrl, np.ones(4))

        # A packet 20 ms old (> 5 ms threshold) must be ignored in favor of
        # the last fresh command.
        with patch("wheeled_uav.runtime.command_dispatcher.time.time_ns", return_value=21_000_000):
            dispatcher.apply_next_command(
                _FakeSocket({"sequence": 2, "wall_time_send_ns": 1_000_000, "rotor_thrusts": [9.0, 9.0, 9.0, 9.0]}), data
            )

        np.testing.assert_allclose(data.ctrl, np.ones(4))
        self.assertEqual(dispatcher.stats.last_applied_policy, "hold-last-command")

    def test_body_wrench_persists_across_steps_and_survives_aero_addition(self) -> None:
        dispatcher = ControlCommandDispatcher(_DISPATCHER_PARAMS, FidelityConfig(), num_uavs=1)
        dispatcher._uav_body_ids = [1]  # what bind_bodies would resolve
        data = _FakeData(4)
        wrench = [8.0, 0.0, 0.0, 0.0, 0.5, 0.0]

        dispatcher.apply_next_command(
            _FakeSocket({"rotor_thrusts": [1.0, 1.0, 1.0, 1.0], "body_wrench": wrench}), data
        )
        dispatcher.refresh_body_wrenches(data)
        np.testing.assert_allclose(data.xfrc_applied[1], wrench)

        # An aerodynamics-style addition on top must be rebased away on the
        # next step's refresh, not accumulate and not clobber the command.
        data.xfrc_applied[1, 0:3] += [0.0, 0.0, 2.5]
        dispatcher.apply_next_command(_FakeSocket(), data)  # no new packet
        dispatcher.refresh_body_wrenches(data)
        np.testing.assert_allclose(data.xfrc_applied[1], wrench)

        # Other bodies' rows (e.g. viewer perturbations) are left alone.
        data.xfrc_applied[2, 1] = 4.0
        dispatcher.refresh_body_wrenches(data)
        self.assertEqual(data.xfrc_applied[2, 1], 4.0)

        # A fresh command without a wrench clears the command-owned row (once).
        dispatcher.apply_next_command(_FakeSocket({"rotor_thrusts": [1.0, 1.0, 1.0, 1.0]}), data)
        dispatcher.refresh_body_wrenches(data)
        np.testing.assert_allclose(data.xfrc_applied[1], np.zeros(6))

        # With no wrench command active, the rows are left untouched so viewer
        # drag perturbations on the vehicle reach the physics.
        data.xfrc_applied[1, 4] = 7.7
        dispatcher.refresh_body_wrenches(data)
        self.assertEqual(data.xfrc_applied[1, 4], 7.7)

        # ...unless a per-step rebuild is forced (active aerodynamics), which
        # reclaims the row for the command/aero composition.
        dispatcher.refresh_body_wrenches(data, force_rebuild=True)
        np.testing.assert_allclose(data.xfrc_applied[1], np.zeros(6))

    def test_delays_hil_command_until_due(self) -> None:
        dispatcher = ControlCommandDispatcher(
            _DISPATCHER_PARAMS,
            FidelityConfig(
                mode="hil",
                network=NetworkFidelityConfig(enabled=True, command_rx_latency_ms=5.0),
            ),
            num_uavs=1,
        )
        data = _FakeData(4)

        with patch("wheeled_uav.runtime.command_dispatcher.time.time_ns", side_effect=[1_000_000, 1_000_000, 7_000_000]):
            dispatcher.apply_next_command(_FakeSocket({"sequence": 1, "rotor_thrusts": [1.0, 2.0, 3.0, 4.0]}), data)
            np.testing.assert_allclose(data.ctrl, np.zeros(4))
            dispatcher.apply_next_command(_FakeSocket(), data)

        np.testing.assert_allclose(data.ctrl, np.array([1.0, 2.0, 3.0, 4.0]))
        self.assertEqual(dispatcher.stats.last_packet_sequence, 1)


class PlantFidelityTests(unittest.TestCase):
    def test_apply_actuator_dynamics_step_applies_first_order_lag(self) -> None:
        applied = apply_actuator_dynamics_step(
            np.array([4.0, 4.0, 4.0, 4.0]),
            np.zeros(4),
            0.001,
            FidelityConfig(mode="hil", actuator_dynamics=ActuatorDynamicsConfig(motor_tau_ms=10.0)),
            2.0e-5,
        )

        np.testing.assert_allclose(applied, np.full(4, 4.0 / 11.0), rtol=1.0e-6, atol=1.0e-6)

    def test_apply_actuator_dynamics_step_limits_thrust_rate(self) -> None:
        applied = apply_actuator_dynamics_step(
            np.array([4.0, 4.0, 4.0, 4.0]),
            np.zeros(4),
            0.001,
            FidelityConfig(mode="hil", actuator_dynamics=ActuatorDynamicsConfig(thrust_rate_limit_n_per_s=500.0)),
            2.0e-5,
        )

        np.testing.assert_allclose(applied, np.full(4, 0.5), rtol=1.0e-9, atol=1.0e-9)

    def test_apply_sensor_fidelity_preserves_truth_in_baseline(self) -> None:
        position = np.array([1.0, 2.0, 3.0])
        velocity = np.array([0.1, 0.2, 0.3])
        angular_velocity_world = np.array([0.01, 0.02, 0.03])
        rotation_matrix = np.eye(3)

        measured = apply_sensor_fidelity(
            position,
            velocity,
            angular_velocity_world,
            rotation_matrix,
            FidelityConfig(),
            np.random.default_rng(0),
        )

        for actual, expected in zip(measured, (position, velocity, angular_velocity_world, rotation_matrix), strict=True):
            np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)

    def test_apply_sensor_fidelity_adds_noise_in_hil(self) -> None:
        position = np.array([1.0, 2.0, 3.0])
        velocity = np.array([0.1, 0.2, 0.3])
        angular_velocity_world = np.array([0.01, 0.02, 0.03])
        rotation_matrix = np.eye(3)

        measured_position, measured_velocity, measured_angular_velocity, measured_rotation = apply_sensor_fidelity(
            position,
            velocity,
            angular_velocity_world,
            rotation_matrix,
            FidelityConfig(
                mode="hil",
                sensor_fidelity=SensorFidelityConfig(
                    position_noise_std_m=0.01,
                    velocity_noise_std_m_per_s=0.02,
                    angular_velocity_noise_std_rad_per_s=0.03,
                    attitude_noise_std_rad=0.01,
                ),
            ),
            np.random.default_rng(0),
        )

        self.assertFalse(np.allclose(measured_position, position))
        self.assertFalse(np.allclose(measured_velocity, velocity))
        self.assertFalse(np.allclose(measured_angular_velocity, angular_velocity_world))
        self.assertFalse(np.allclose(measured_rotation, rotation_matrix))


if __name__ == "__main__":
    unittest.main()
