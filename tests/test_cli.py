from __future__ import annotations

import unittest
from unittest.mock import patch

from wheeled_uav.cli import build_cli_parser, main, resolve_params_file
from wheeled_uav.paths import REPO_ROOT


class PresetResolutionTests(unittest.TestCase):
    def test_explicit_params_file_wins_over_preset(self) -> None:
        self.assertEqual(resolve_params_file("custom.json", "wall_demo"), "custom.json")

    def test_none_when_neither_given(self) -> None:
        self.assertIsNone(resolve_params_file(None, None))

    def test_preset_resolves_to_bundled_config(self) -> None:
        resolved = resolve_params_file(None, "wall_demo")
        self.assertEqual(resolved, str(REPO_ROOT / "configs" / "vehicle_params.wall_demo.json"))

    def test_unknown_preset_exits_with_helpful_message(self) -> None:
        with self.assertRaises(SystemExit) as context:
            resolve_params_file(None, "no_such_preset")
        # The listing must name what the user can type after --preset, not filenames.
        self.assertIn("available presets", str(context.exception))
        self.assertIn("wall_demo", str(context.exception))
        self.assertNotIn("vehicle_params.wall_demo.json", str(context.exception))

    def test_simulate_accepts_preset_flag(self) -> None:
        arguments = build_cli_parser().parse_args(["simulate", "--preset", "wall_demo"])
        self.assertEqual(arguments.preset, "wall_demo")


class CliParserTests(unittest.TestCase):
    def test_simulate_headless_duration_options_are_parsed(self) -> None:
        parser = build_cli_parser()

        arguments = parser.parse_args([
            "simulate",
            "--headless",
            "--duration-seconds",
            "4.5",
            "--num-uavs",
            "3",
        ])

        self.assertTrue(arguments.headless)
        self.assertEqual(arguments.duration_seconds, 4.5)
        self.assertEqual(arguments.num_uavs, 3)

    def test_bare_invocation_provides_every_simulate_default(self) -> None:
        # A bare `mujoco-wheeled-uav-simulator` run falls through to simulate;
        # every option the handler reads must have a top-level default or the
        # run dies with AttributeError (regression: `record` was missing).
        arguments = build_cli_parser().parse_args([])

        for option_name in (
            "instance_id", "num_uavs", "spawn_radius", "recv_port", "send_port",
            "bind_ip", "state_target_ip", "params_file", "preset", "xml_template_file",
            "generated_xml_dir", "fidelity_mode", "pacing_mode", "headless",
            "duration_seconds", "record", "hold_until_first_command",
        ):
            self.assertTrue(hasattr(arguments, option_name), f"missing default for {option_name}")

    def test_main_wires_simulate_options_into_request(self) -> None:
        captured_requests = []

        with patch("wheeled_uav.runtime.run_simulation", side_effect=lambda request: captured_requests.append(request)):
            exit_code = main(["simulate", "--no-hold-until-first-command", "--headless", "--record", "out.mp4"])

        self.assertEqual(exit_code, 0)
        self.assertEqual(len(captured_requests), 1)
        request = captured_requests[0]
        self.assertFalse(request.hold_until_first_command)
        self.assertTrue(request.headless)
        self.assertEqual(request.record_path, "out.mp4")
        # None -> params file decides (tri-state like pacing_mode).
        self.assertIsNone(request.fidelity_mode)
        self.assertIsNone(request.pacing_mode)

    def test_simulate_hold_until_first_command_flag_is_tristate(self) -> None:
        parser = build_cli_parser()

        # Default None keeps the params-file value in charge.
        self.assertIsNone(parser.parse_args(["simulate"]).hold_until_first_command)
        self.assertTrue(parser.parse_args(["simulate", "--hold-until-first-command"]).hold_until_first_command)
        self.assertFalse(parser.parse_args(["simulate", "--no-hold-until-first-command"]).hold_until_first_command)

    def test_simulate_fidelity_mode_is_parsed(self) -> None:
        parser = build_cli_parser()

        arguments = parser.parse_args([
            "simulate",
            "--fidelity-mode",
            "hil",
        ])

        self.assertEqual(arguments.fidelity_mode, "hil")

    def test_simulate_ip_endpoints_can_be_split(self) -> None:
        parser = build_cli_parser()

        arguments = parser.parse_args([
            "simulate",
            "--bind-ip",
            "0.0.0.0",
            "--state-target-ip",
            "192.168.0.42",
        ])

        self.assertEqual(arguments.bind_ip, "0.0.0.0")
        self.assertEqual(arguments.state_target_ip, "192.168.0.42")

    def test_simulate_ip_endpoints_default_to_localhost(self) -> None:
        parser = build_cli_parser()

        arguments = parser.parse_args(["simulate"])

        self.assertIsNone(arguments.bind_ip)
        self.assertIsNone(arguments.state_target_ip)

    def test_hover_controller_options_are_parsed(self) -> None:
        parser = build_cli_parser()

        arguments = parser.parse_args([
            "hover-controller",
            "--bind-ip",
            "0.0.0.0",
            "--target-ip",
            "192.168.0.42",
            "--duration-seconds",
            "5",
            "--target-position",
            "0",
            "0",
            "1.8",
        ])

        self.assertEqual(arguments.bind_ip, "0.0.0.0")
        self.assertEqual(arguments.target_ip, "192.168.0.42")
        self.assertEqual(arguments.duration_seconds, 5.0)
        self.assertEqual(arguments.target_position, [0.0, 0.0, 1.8])

    def test_hover_controller_fidelity_mode_is_parsed(self) -> None:
        parser = build_cli_parser()

        arguments = parser.parse_args([
            "hover-controller",
            "--fidelity-mode",
            "hil",
        ])

        self.assertEqual(arguments.fidelity_mode, "hil")


if __name__ == "__main__":
    unittest.main()
