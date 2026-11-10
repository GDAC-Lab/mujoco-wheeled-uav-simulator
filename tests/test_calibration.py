from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from wheeled_uav.calibration import apply_calibration_file
from wheeled_uav.config import clear_vehicle_params_cache, load_vehicle_params


def _write_params(directory: Path, actuation_extra: dict | None = None) -> Path:
    actuation = {
        "command_mode": "omega",
        "thrust_coefficient": 2.0e-5,
        "yaw_moment_ratio": 0.02,
        "rotors": [
            {"name": "fr", "yaw_moment_ratio": 0.02, "spin_sign": 1},
            {"name": "fl", "yaw_moment_ratio": 0.02, "spin_sign": -1},
        ],
    }
    if actuation_extra:
        actuation.update(actuation_extra)
    params_path = directory / "vehicle_params.json"
    params_path.write_text(
        json.dumps(
            {
                "actuation": actuation,
                "actuator_dynamics": {"motor_tau_ms": 0.0},
            }
        ),
        encoding="utf-8",
    )
    return params_path


def _write_calibration(path: Path, sim_params: dict, schema: str = "uav-propulsion-calibration/1") -> None:
    path.write_text(
        json.dumps(
            {
                "schema": schema,
                "source": {"mat_file": "thrust_20260727_test.mat", "motor": "M2206"},
                "sim_params": sim_params,
            }
        ),
        encoding="utf-8",
    )


class ApplyCalibrationTests(unittest.TestCase):
    def tearDown(self) -> None:
        clear_vehicle_params_cache()

    def test_calibration_overrides_actuation_and_rotor_values(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            params_path = _write_params(directory, {"calibration_file": "calib.json"})
            _write_calibration(
                directory / "calib.json",
                {"thrust_coefficient": 3.5e-5, "yaw_moment_ratio": 0.025, "motor_tau_ms": 40.0},
            )

            params = load_vehicle_params(params_path=params_path)

            self.assertEqual(params["actuation"]["thrust_coefficient"], 3.5e-5)
            self.assertEqual(params["actuation"]["yaw_moment_ratio"], 0.025)
            for rotor in params["actuation"]["rotors"]:
                self.assertEqual(rotor["yaw_moment_ratio"], 0.025)
            self.assertEqual(params["actuator_dynamics"]["motor_tau_ms"], 40.0)
            self.assertEqual(params["calibration_applied"]["source"]["mat_file"], "thrust_20260727_test.mat")
            self.assertEqual(
                set(params["calibration_applied"]["applied"]),
                {"thrust_coefficient", "yaw_moment_ratio", "motor_tau_ms"},
            )

    def test_motor_time_constant_goes_to_the_plant_model_when_it_has_rotors(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            params_path = _write_params(directory, {"calibration_file": "calib.json"})
            document = json.loads(params_path.read_text(encoding="utf-8"))
            document["plant"] = {"actuation": {"delay_s": 0.004}}
            params_path.write_text(json.dumps(document), encoding="utf-8")
            _write_calibration(directory / "calib.json", {"motor_tau_ms": 35.0})

            params = load_vehicle_params(params_path=params_path)

            self.assertEqual(params["plant"]["actuation"]["motor_time_constant_up_s"], 0.035)
            self.assertEqual(params["actuator_dynamics"]["motor_tau_ms"], 0.0)  # the HIL-only lag stays off
            clear_vehicle_params_cache()
            # A time constant set in the plant section (an identified value) wins over the bench value.
            document["plant"]["actuation"]["motor_time_constant_up_s"] = 0.028
            params_path.write_text(json.dumps(document), encoding="utf-8")
            params = load_vehicle_params(params_path=params_path)
            self.assertEqual(params["plant"]["actuation"]["motor_time_constant_up_s"], 0.028)

    def test_null_entries_are_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            params_path = _write_params(directory, {"calibration_file": "calib.json"})
            _write_calibration(
                directory / "calib.json",
                {"thrust_coefficient": None, "yaw_moment_ratio": 0.03, "motor_tau_ms": None},
            )

            params = load_vehicle_params(params_path=params_path)

            self.assertEqual(params["actuation"]["thrust_coefficient"], 2.0e-5)
            self.assertEqual(params["actuation"]["yaw_moment_ratio"], 0.03)
            self.assertEqual(params["actuator_dynamics"]["motor_tau_ms"], 0.0)
            self.assertEqual(set(params["calibration_applied"]["applied"]), {"yaw_moment_ratio"})

    def test_params_without_calibration_file_are_untouched(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            params_path = _write_params(Path(temp_dir))

            params = load_vehicle_params(params_path=params_path)

            self.assertNotIn("calibration_applied", params)
            self.assertEqual(params["actuation"]["thrust_coefficient"], 2.0e-5)

    def test_missing_calibration_file_raises(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            params_path = _write_params(Path(temp_dir), {"calibration_file": "missing.json"})

            with self.assertRaises(FileNotFoundError):
                load_vehicle_params(params_path=params_path)

    def test_unsupported_schema_raises(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            params_path = _write_params(directory, {"calibration_file": "calib.json"})
            _write_calibration(directory / "calib.json", {"yaw_moment_ratio": 0.03}, schema="something-else/9")

            with self.assertRaises(ValueError):
                load_vehicle_params(params_path=params_path)

    def test_nonpositive_thrust_coefficient_raises(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            params_path = _write_params(directory, {"calibration_file": "calib.json"})
            _write_calibration(directory / "calib.json", {"thrust_coefficient": 0.0})

            with self.assertRaises(ValueError):
                load_vehicle_params(params_path=params_path)

    def test_a_plant_merged_in_after_loading_takes_the_bench_time_constant(self) -> None:
        # Simulator(overrides=...) and --plant add the plant section to params that already went
        # through the calibration: applying it again must move the value, not leave it in both places.
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            params_path = _write_params(directory, {"calibration_file": "calib.json"})
            _write_calibration(directory / "calib.json", {"motor_tau_ms": 35.0})
            params = load_vehicle_params(params_path=params_path)
            self.assertEqual(params["actuator_dynamics"]["motor_tau_ms"], 35.0)
            params["plant"] = {"actuation": {"delay_s": 0.004}}

            again = apply_calibration_file(params, params_path)

            self.assertEqual(again["plant"]["actuation"]["motor_time_constant_up_s"], 0.035)
            self.assertNotIn("motor_tau_ms", again["actuator_dynamics"])


if __name__ == "__main__":
    unittest.main()
