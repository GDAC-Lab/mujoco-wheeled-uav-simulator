from __future__ import annotations

import json
import unittest

from wheeled_uav.protocol import (
    get_instance_ports,
    parse_command_packet,
    parse_control_input,
    parse_multi_uav_command_packet,
    parse_multi_uav_control_input,
    receive_control_command,
)


class _FakeSocket:
    def __init__(self, *payloads: dict[str, object]):
        self._queue = [json.dumps(payload).encode("utf-8") for payload in payloads]

    def recvfrom(self, _buffer_size: int) -> tuple[bytes, tuple[str, int]]:
        if not self._queue:
            raise BlockingIOError()
        return self._queue.pop(0), ("127.0.0.1", 5000)


class NetworkParsingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.params = {"actuation": {"thrust_coefficient": 2.0e-5}}

    def test_get_instance_ports_offsets_send_and_receive(self) -> None:
        self.assertEqual(get_instance_ports(0), (5000, 5001))
        self.assertEqual(get_instance_ports(3), (5006, 5007))

    def test_parse_control_input_accepts_scalar_thrust(self) -> None:
        command = parse_control_input({"thrust": 1.5}, self.params)

        self.assertEqual(command, [1.5, 1.5, 1.5, 1.5])

    def test_parse_control_input_converts_rotor_omega_to_thrust(self) -> None:
        command = parse_control_input({"rotor_omega": [10.0, 20.0, 30.0, 40.0]}, self.params)

        self.assertEqual(command, [0.002, 0.008, 0.018000000000000002, 0.032])

    def test_parse_control_input_maps_wrench_through_rotor_geometry(self) -> None:
        # Regression: the deferred import behind the documented `wrench` command
        # form must resolve (it silently broke when model_builder.py moved to
        # model/builder.py). A pure-collective wrench on a symmetric quad splits
        # evenly across the four rotors.
        params = {
            "actuation": {
                "thrust_coefficient": 2.0e-5,
                "max_rotor_thrust": 20.0,
                "yaw_moment_ratio": 0.02,
            },
            "drone": {
                "arm": {"x": 0.08, "y": 0.1},
                "propeller": {"z": 0.0125},
            },
        }

        command = parse_control_input({"wrench": [4.0, 0.0, 0.0, 0.0]}, params)

        self.assertEqual(len(command), 4)
        for rotor_thrust in command:
            self.assertAlmostEqual(rotor_thrust, 1.0, places=9)

    def test_parse_command_packet_extracts_metadata_and_stale_flag(self) -> None:
        packet = parse_command_packet(
            {
                "protocol_version": 2,
                "sequence": 9,
                "source_state_sequence": 8,
                "wall_time_send_ns": 1_000_000,
                "fidelity_mode": "hil",
                "rotor_thrusts": [1.0, 2.0, 3.0, 4.0],
            },
            self.params,
            receive_time_ns=6_500_000,
            stale_command_threshold_ms=4.0,
        )

        self.assertIsNotNone(packet)
        assert packet is not None
        self.assertEqual(packet.rotor_thrusts, [1.0, 2.0, 3.0, 4.0])
        self.assertEqual(packet.metrics.protocol_version, 2)
        self.assertEqual(packet.metrics.sequence, 9)
        self.assertEqual(packet.metrics.source_state_sequence, 8)
        self.assertEqual(packet.metrics.fidelity_mode, "hil")
        self.assertAlmostEqual(packet.metrics.age_ms or 0.0, 5.5)
        self.assertTrue(packet.metrics.is_stale)

    def test_parse_multi_uav_command_packet_preserves_nested_rotor_thrusts(self) -> None:
        packet = parse_multi_uav_command_packet(
            {
                "sequence": 4,
                "rotor_thrusts": [[1.0, 2.0, 3.0, 4.0], [0.5, 0.5, 0.5, 0.5]],
            },
            self.params,
            num_uavs=2,
            receive_time_ns=20,
        )

        self.assertIsNotNone(packet)
        assert packet is not None
        self.assertEqual(packet.rotor_thrusts, [[1.0, 2.0, 3.0, 4.0], [0.5, 0.5, 0.5, 0.5]])
        self.assertEqual(packet.metrics.sequence, 4)
        self.assertIsNone(packet.metrics.age_ms)

    def test_receive_control_command_returns_latest_packet(self) -> None:
        sock = _FakeSocket(
            {"sequence": 1, "rotor_thrusts": [1.0, 1.0, 1.0, 1.0]},
            {"sequence": 3, "rotor_thrusts": [2.0, 2.0, 2.0, 2.0]},
        )

        packet = receive_control_command(sock, self.params)

        self.assertIsNotNone(packet)
        assert packet is not None
        self.assertEqual(packet.rotor_thrusts, [2.0, 2.0, 2.0, 2.0])
        self.assertEqual(packet.metrics.sequence, 3)

    def test_parse_multi_uav_control_input_accepts_embedded_uav_commands(self) -> None:
        command = parse_multi_uav_control_input(
            {
                "uavs": [
                    {"rotor_thrusts": [1.0, 2.0, 3.0, 4.0]},
                    {"thrust": 0.5},
                ]
            },
            self.params,
            num_uavs=2,
        )

        self.assertEqual(command, [[1.0, 2.0, 3.0, 4.0], [0.5, 0.5, 0.5, 0.5]])

    def test_parse_multi_uav_control_input_accepts_both_omega_spellings(self) -> None:
        # A controller using the single-UAV field name must not go silent at
        # num_uavs=2.
        omegas = [[10.0, 10.0, 10.0, 10.0], [20.0, 20.0, 20.0, 20.0]]

        for field_name in ("rotor_omegas", "rotor_omega"):
            command = parse_multi_uav_control_input({field_name: omegas}, self.params, num_uavs=2)
            self.assertIsNotNone(command, field_name)
            self.assertEqual(command[0], [0.002] * 4)
            self.assertEqual(command[1], [0.008] * 4)

    def test_multi_uav_packet_collects_per_entry_body_wrenches(self) -> None:
        packet = parse_multi_uav_command_packet(
            {
                "uavs": [
                    {"rotor_thrusts": [1.0, 1.0, 1.0, 1.0], "body_wrench": [5.0, 0.0, 0.0, 0.0, 0.0, 0.0]},
                    {"rotor_thrusts": [2.0, 2.0, 2.0, 2.0]},
                ]
            },
            self.params,
            num_uavs=2,
        )

        self.assertIsNotNone(packet)
        self.assertEqual(packet.body_wrenches, [[5.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0] * 6])

    def test_receive_control_command_survives_malformed_datagrams(self) -> None:
        class _RawSocket:
            def __init__(self, *payloads: bytes):
                self._queue = list(payloads)

            def recvfrom(self, _buffer_size: int) -> tuple[bytes, tuple[str, int]]:
                if not self._queue:
                    raise BlockingIOError()
                return self._queue.pop(0), ("127.0.0.1", 5000)

        # Latest datagram is truncated JSON -> dropped, no exception.
        command = receive_control_command(_RawSocket(b'{"rotor_thrusts": [1.0, 1.0, 1.0'), self.params)
        self.assertIsNone(command)
        # Latest datagram is not UTF-8 -> dropped, no exception.
        command = receive_control_command(_RawSocket(b"\xff\xfe\x00"), self.params)
        self.assertIsNone(command)

    def test_parse_multi_uav_control_input_rejects_wrong_lengths(self) -> None:
        command = parse_multi_uav_control_input({"rotor_thrusts": [[1.0, 2.0, 3.0]]}, self.params, num_uavs=1)

        self.assertIsNone(command)


if __name__ == "__main__":
    unittest.main()