"""System identification: the estimators recover the plant that made a simulated flight."""

from __future__ import annotations

import json

import numpy as np
import pytest

from wheeled_uav.config import load_vehicle_params, parse_total_mass
from wheeled_uav.flightlog import FlightLog, quaternion_to_rotation, rotation_to_quaternion
from wheeled_uav.plant import CorrectionModel
from wheeled_uav.sim import Simulator
from wheeled_uav.sysid import (
    estimate_delay,
    excitation_flight,
    fit_actuation,
    fit_battery,
    fit_drag,
    identify,
    prediction_error,
    simulate_rotor_thrusts,
)

TRUE_PLANT = {
    "plant": {
        "actuation": {"delay_s": 0.006, "motor_time_constant_up_s": 0.028, "motor_time_constant_down_s": 0.042, "thrust_scale": 0.94},
        "aerodynamics": {"rotor_drag_coefficients_n_s_per_m": [0.22, 0.30, 0.08]},
        "battery": {"internal_resistance_ohm": 0.05, "capacity_mah": 400.0, "auxiliary_power_w": 4.0},
    }
}
MASS = parse_total_mass(load_vehicle_params())


@pytest.fixture(scope="module")
def flights():
    with Simulator(overrides=TRUE_PLANT) as sim:
        identification = excitation_flight(sim, duration_s=10.0, seed=1)
        validation = excitation_flight(sim, duration_s=4.0, seed=2)
    return identification, validation


def test_flight_log_round_trip_and_resampling(tmp_path, flights):
    identification, _ = flights
    identification.save(tmp_path / "flight.npz")
    loaded = FlightLog.load(tmp_path / "flight.npz")
    assert loaded.metadata == json.loads(json.dumps(identification.metadata))
    for name in identification.channels:
        assert np.array_equal(loaded[name], identification[name])
    middle = 0.5 * (identification.time[10] + identification.time[11])
    # Commands are held, measurements interpolated.
    assert np.array_equal(identification.sample("rotor_thrust_command", np.array([middle]))[0], identification["rotor_thrust_command"][10])
    assert identification.sample("position", np.array([middle]))[0] == pytest.approx(0.5 * (identification["position"][10] + identification["position"][11]))
    rotation = quaternion_to_rotation(identification["quaternion"])
    assert rotation_to_quaternion(rotation) == pytest.approx(identification["quaternion"] * np.sign(identification["quaternion"][:, :1]), abs=1e-12)
    with pytest.raises(ValueError, match="unknown channel"):
        FlightLog(identification.time, {**identification.channels, "altitude": identification["position"][:, 2]})


def test_actuation_is_recovered(flights):
    identification, _ = flights
    fit = fit_actuation(identification, mass=MASS)
    assert fit.delay_s == pytest.approx(0.006, abs=1e-9)
    assert fit.motor_time_constant_up_s == pytest.approx(0.028, rel=0.03)
    assert fit.motor_time_constant_down_s == pytest.approx(0.042, rel=0.03)
    assert fit.thrust_scale == pytest.approx(0.94, rel=0.003)
    assert fit.vertical_drag_n_s_per_m == pytest.approx(0.08, abs=0.01)
    assert fit.r_squared > 0.999


def test_drag_and_battery_are_recovered(flights):
    identification, _ = flights
    drag = fit_drag(identification, mass=MASS)
    assert drag.rotor_drag_coefficients_n_s_per_m == pytest.approx((0.22, 0.30), rel=0.03)
    battery = fit_battery(identification, rotor_thrusts=simulate_rotor_thrusts(identification, fit_actuation(identification, mass=MASS)))
    assert battery.internal_resistance_ohm == pytest.approx(0.05, rel=0.01)
    assert battery.open_circuit_slope_v_per_ah == pytest.approx((16.8 - 13.6) / 0.4, rel=0.01)
    assert battery.power_coefficient_w_per_n1p5 == pytest.approx(12.0, rel=0.03)


def test_delay_by_cross_correlation():
    time = np.arange(0.0, 4.0, 0.002)
    rng = np.random.default_rng(0)
    cause = np.cumsum(rng.normal(size=time.size))
    effect = np.interp(time - 0.013, time, cause)
    assert estimate_delay(time, cause, effect, max_delay_s=0.05) == pytest.approx(0.013, abs=0.001)


def test_identified_plant_predicts_validation_flights(tmp_path, flights):
    identification, validation = flights
    result = identify(identification, load_vehicle_params(), correction="knn", correction_path=tmp_path / "correction.json")
    plant = result["plant"]
    assert set(plant) == {"actuation", "aerodynamics", "battery", "correction"}
    assert CorrectionModel.from_file(tmp_path / "correction.json").kind == "knn"
    errors = {}
    for name, overrides in (("identified", {"plant": plant}), ("rigid body only", {})):
        with Simulator(overrides=overrides) as sim:
            report = prediction_error(validation, sim, horizons_s=(0.1, 0.5), stride_s=0.5)
        errors[name] = report["horizons_s"]["0.5"]["position_m"]["median"]
    assert errors["identified"] < 0.005
    assert errors["identified"] < 0.1 * errors["rigid body only"]


def test_prediction_error_of_the_true_plant_is_small(flights):
    _, validation = flights
    with Simulator(overrides=TRUE_PLANT) as sim:
        report = prediction_error(validation, sim, horizons_s=(0.1,), stride_s=0.5)
    assert report["starts"] >= 7
    # Only the wheels' spin, which logs do not record, differs from the flight.
    assert report["horizons_s"]["0.1"]["position_m"]["max"] < 2e-4
