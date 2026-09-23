from __future__ import annotations

import json
import tempfile
import unittest
import warnings
from pathlib import Path

import pytest

from wheeled_uav.config import (
    InertialReferenceWarning,
    build_fidelity_config,
    clear_vehicle_params_cache,
    load_vehicle_params,
    parse_total_mass,
)
from wheeled_uav.paths import PathResolver

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SHIPPED_VEHICLE_PARAMS = (
    REPOSITORY_ROOT / "vehicle_params.json",
    REPOSITORY_ROOT / "vehicle_params.tilted_rotor_example.json",
    REPOSITORY_ROOT / "configs" / "vehicle_params.wall_demo.json",
)


class LoadVehicleParamsTests(unittest.TestCase):
    def tearDown(self) -> None:
        clear_vehicle_params_cache()

    def test_load_vehicle_params_uses_resolver(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir)
            params_path = repo_root / "vehicle_params.json"
            params_path.write_text(json.dumps({"simulation": {"timestep": 0.01}}), encoding="utf-8")

            params = load_vehicle_params(path_resolver=PathResolver(repo_root=repo_root))

            self.assertEqual(params["simulation"]["timestep"], 0.01)

    def test_cache_can_be_cleared_after_file_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            params_path = Path(temp_dir) / "vehicle_params.json"
            params_path.write_text(json.dumps({"value": 1}), encoding="utf-8")

            first_params = load_vehicle_params(params_path=params_path)
            params_path.write_text(json.dumps({"value": 2}), encoding="utf-8")
            cached_params = load_vehicle_params(params_path=params_path)
            clear_vehicle_params_cache()
            refreshed_params = load_vehicle_params(params_path=params_path)

            self.assertEqual(first_params["value"], 1)
            self.assertEqual(cached_params["value"], 1)
            self.assertEqual(refreshed_params["value"], 2)

    def test_load_vehicle_params_returns_an_isolated_copy(self) -> None:
        # Regression: the cache used to hand out the SAME dict to every caller,
        # so one caller's mutation poisoned all later loads in the process.
        with tempfile.TemporaryDirectory() as temp_dir:
            params_path = Path(temp_dir) / "vehicle_params.json"
            params_path.write_text(json.dumps({"controller": {"max_tilt_deg": 35.0}}), encoding="utf-8")

            first_params = load_vehicle_params(params_path=params_path)
            first_params["controller"]["max_tilt_deg"] = 0.0
            second_params = load_vehicle_params(params_path=params_path)

            self.assertEqual(second_params["controller"]["max_tilt_deg"], 35.0)

    def test_build_fidelity_config_uses_backward_compatible_defaults(self) -> None:
        fidelity = build_fidelity_config({})

        self.assertEqual(fidelity.mode, "baseline")
        self.assertFalse(fidelity.network.enabled)
        self.assertEqual(fidelity.network.state_tx_latency_ms, 0.0)
        self.assertEqual(fidelity.network.stale_command_policy, "hold-last-command")
        self.assertEqual(fidelity.actuator_dynamics.motor_tau_ms, 0.0)

    def test_build_fidelity_config_reads_hil_overrides(self) -> None:
        fidelity = build_fidelity_config(
            {
                "fidelity_mode": "hil",
                "network_fidelity": {
                    "enabled": True,
                    "state_tx_latency_ms": 12.5,
                    "command_rx_latency_ms": 8.0,
                    "packet_loss_percent": 1.5,
                    "jitter_std_dev_ms": 2.0,
                    "stale_command_threshold_ms": 40.0,
                    "stale_command_policy": "zero-thrust",
                },
                "actuator_dynamics": {
                    "motor_tau_ms": 15.0,
                },
                "sensor_fidelity": {
                    "position_noise_std_m": 0.01,
                },
                "logging_config": {
                    "log_network_stats": True,
                },
            }
        )

        self.assertEqual(fidelity.mode, "hil")
        self.assertTrue(fidelity.network.enabled)
        self.assertEqual(fidelity.network.state_tx_latency_ms, 12.5)
        self.assertEqual(fidelity.network.stale_command_threshold_ms, 40.0)
        self.assertEqual(fidelity.network.stale_command_policy, "zero-thrust")
        self.assertEqual(fidelity.actuator_dynamics.motor_tau_ms, 15.0)
        self.assertEqual(fidelity.sensor_fidelity.position_noise_std_m, 0.01)
        self.assertTrue(fidelity.logging.log_network_stats)

    def test_repository_default_vehicle_params_include_baseline_fidelity_schema(self) -> None:
        params = load_vehicle_params()

        fidelity = build_fidelity_config(params)

        self.assertEqual(fidelity.mode, "baseline")
        self.assertFalse(fidelity.network.enabled)
        self.assertEqual(fidelity.network.state_tx_latency_ms, 0.0)
        self.assertEqual(fidelity.network.command_rx_latency_ms, 0.0)
        self.assertEqual(fidelity.network.packet_loss_percent, 0.0)
        self.assertEqual(fidelity.network.jitter_std_dev_ms, 0.0)
        self.assertEqual(fidelity.network.stale_command_policy, "hold-last-command")
        self.assertEqual(fidelity.actuator_dynamics.motor_tau_ms, 0.0)
        self.assertEqual(fidelity.sensor_fidelity.position_noise_std_m, 0.0)
        self.assertFalse(fidelity.logging.log_network_stats)


def test_missing_inertial_reference_warns_and_keeps_body_only():
    params = load_vehicle_params()
    del params["drone"]["inertial_reference"]

    with pytest.warns(InertialReferenceWarning, match=r"0\.8 kg plus two wheels of 0\.1 kg = 1 kg"):
        total_mass = parse_total_mass(params)

    assert total_mass == pytest.approx(0.8 + 2.0 * 0.1)


@pytest.mark.parametrize(
    ("reference", "expected_total_mass"),
    [("body_only", 1.0), ("total_vehicle", 0.8), (" Total_Vehicle ", 0.8)],
)
def test_explicit_inertial_reference_does_not_warn(reference, expected_total_mass):
    params = load_vehicle_params()
    params["drone"]["inertial_reference"] = reference

    with warnings.catch_warnings():
        warnings.simplefilter("error", InertialReferenceWarning)
        assert parse_total_mass(params) == pytest.approx(expected_total_mass)


def test_unknown_inertial_reference_is_rejected():
    params = load_vehicle_params()
    params["drone"]["inertial_reference"] = "total"

    with pytest.raises(ValueError, match="inertial_reference"):
        parse_total_mass(params)


@pytest.mark.parametrize("params_path", SHIPPED_VEHICLE_PARAMS, ids=lambda path: path.name)
def test_shipped_vehicle_params_set_inertial_reference(params_path):
    # The bundled configs must not trip the warning; they keep the historical meaning.
    drone = json.loads(params_path.read_text(encoding="utf-8"))["drone"]
    assert drone["inertial_reference"] == "body_only"


if __name__ == "__main__":
    unittest.main()