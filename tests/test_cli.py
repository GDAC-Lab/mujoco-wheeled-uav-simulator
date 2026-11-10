from __future__ import annotations

import io
import unittest
from contextlib import redirect_stderr
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
            "physics_threads", "state_format", "seed",
        ):
            self.assertTrue(hasattr(arguments, option_name), f"missing default for {option_name}")

    def test_simulate_large_team_options_reach_the_request(self) -> None:
        with patch("wheeled_uav.runtime.run_simulation") as run_simulation:
            main(["simulate", "--headless", "--num-uavs", "8", "--physics-threads", "4", "--state-format", "binary"])
            main(["simulate", "--headless"])

        first, second = (call.args[0] for call in run_simulation.call_args_list)
        self.assertEqual((first.physics_threads, first.state_format), (4, "binary"))
        # None -> simulation.physics_threads / simulation.state_format in the params file decide.
        self.assertEqual((second.physics_threads, second.state_format), (None, None))
        with patch("wheeled_uav.runtime.run_simulation") as run_simulation:
            main(["simulate", "--headless", "--seed", "12"])
        self.assertEqual(run_simulation.call_args.args[0].seed, 12)
        for invalid in (["--state-format", "msgpack"], ["--physics-threads", "-1"], ["--seed", "-3"]):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(["simulate", "--headless", *invalid])

    def test_spawn_radius_zero_is_accepted_and_negative_rejected(self) -> None:
        # Explicit spawns and single-UAV runs never place a ring, so 0 must pass the CLI.
        with patch("wheeled_uav.runtime.run_simulation") as run_simulation:
            self.assertEqual(main(["simulate", "--headless", "--spawn-radius", "0"]), 0)
        self.assertEqual(run_simulation.call_args.args[0].spawn_radius, 0.0)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(["simulate", "--headless", "--spawn-radius", "-1"])

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


class IdentificationCommandTests(unittest.TestCase):
    def test_excite_identify_and_predict(self) -> None:
        import json
        import tempfile
        from contextlib import redirect_stdout
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "true_plant.json").write_text(
                json.dumps({"actuation": {"delay_s": 0.004, "motor_time_constant_up_s": 0.03, "thrust_scale": 0.97}}), encoding="utf-8"
            )
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(main(["excite", "--plant", str(folder / "true_plant.json"), "--duration", "4", "--out", str(folder / "flight.npz")]), 0)
                self.assertEqual(main(["identify", str(folder / "flight.npz"), "--correction", "linear", "--out", str(folder / "result.json")]), 0)
                self.assertEqual(main(["predict", str(folder / "flight.npz"), "--plant", str(folder / "result.json"), "--horizons", "0.1", "--stride", "1", "--out", str(folder / "prediction.json")]), 0)
            result = json.loads((folder / "result.json").read_text(encoding="utf-8"))
            self.assertAlmostEqual(result["plant"]["actuation"]["delay_s"], 0.004)
            self.assertAlmostEqual(result["plant"]["actuation"]["thrust_scale"], 0.97, places=2)
            self.assertEqual(result["plant"]["correction"]["file"], "result.correction.json")  # next to the result
            prediction = json.loads((folder / "prediction.json").read_text(encoding="utf-8"))
            self.assertLess(prediction["horizons_s"]["0.1"]["position_m"]["median"], 1e-3)

    def test_correction_in_another_folder_and_log_names_without_a_suffix(self) -> None:
        import json
        import tempfile
        from contextlib import redirect_stdout
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "true_plant.json").write_text(json.dumps({"actuation": {"delay_s": 0.004, "motor_time_constant_up_s": 0.03}}), encoding="utf-8")
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(main(["excite", "--plant", str(folder / "true_plant.json"), "--duration", "3", "--out", str(folder / "flight")]), 0)
                flight = folder / "flight.npz"
                self.assertTrue(flight.is_file())  # the name the command reports
                self.assertIn(f"wrote {flight}", output.getvalue())
                arguments = ["identify", str(flight), "--correction", "linear", "--out", str(folder / "out" / "result.json")]
                self.assertEqual(main(arguments + ["--correction-out", str(folder / "corr" / "c.json")]), 0)
                # The result names the correction from its own folder, which is where --plant looks for it.
                result = json.loads((folder / "out" / "result.json").read_text(encoding="utf-8"))
                self.assertEqual(result["plant"]["correction"]["file"], "../corr/c.json")
                self.assertEqual(main(["predict", str(flight), "--plant", str(folder / "out" / "result.json"), "--horizons", "0.1", "--stride", "1"]), 0)

    def test_plant_argument_replaces_the_params_files_section(self) -> None:
        import json
        import tempfile
        from contextlib import redirect_stdout
        from pathlib import Path

        from wheeled_uav.config import load_vehicle_params

        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            params = load_vehicle_params()
            params["plant"] = {"actuation": {"delay_s": 0.02}, "aerodynamics": {"rotor_drag_coefficients_n_s_per_m": [0.5, 0.5, 0.5]}}
            (folder / "vehicle_params.json").write_text(json.dumps(params), encoding="utf-8")
            (folder / "plant.json").write_text(json.dumps({"actuation": {"thrust_scale": 0.9}}), encoding="utf-8")
            with redirect_stdout(io.StringIO()):
                arguments = ["excite", "--params-file", str(folder / "vehicle_params.json"), "--plant", str(folder / "plant.json")]
                self.assertEqual(main(arguments + ["--duration", "1", "--out", str(folder / "flight.npz")]), 0)
            from wheeled_uav.flightlog import FlightLog

            self.assertEqual(FlightLog.load(folder / "flight.npz").metadata["plant"], {"actuation": {"thrust_scale": 0.9}})
