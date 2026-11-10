"""System identification: the estimators recover the plant that made a simulated flight."""

from __future__ import annotations

import json

import numpy as np
import pytest

from wheeled_uav.config import load_vehicle_params, parse_total_mass
from wheeled_uav.controllers.hover import build_hover_controller_config, compute_hover_control
from wheeled_uav.flightlog import FlightLog, quaternion_to_rotation, rotation_to_quaternion
from wheeled_uav.flightlog_ulog import convert_topics
from wheeled_uav.paths import DEFAULT_PATH_RESOLVER
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


# -- what a real log does to the fits -------------------------------------------------------------------


def _without(log: FlightLog, *names: str) -> FlightLog:
    return FlightLog(log.time, {name: values for name, values in log.channels.items() if name not in names}, dict(log.metadata))


def _with_dropouts(log: FlightLog, every_s: float = 2.0, length_s: float = 0.06) -> FlightLog:
    """Pose and velocity held for ``length_s`` every ``every_s`` (a motion-capture dropout), flagged invalid."""
    channels = {name: values.copy() for name, values in log.channels.items()}
    valid = np.ones(len(log))
    period = log.sample_period
    span = int(round(length_s / period))
    for start in range(int(round(1.0 / period)), len(log) - span - 1, int(round(every_s / period))):
        for name in ("position", "velocity", "quaternion"):
            channels[name][start : start + span] = channels[name][start - 1]
        valid[start : start + span] = 0.0
    channels["valid"] = valid
    return FlightLog(log.time, channels, dict(log.metadata))


def test_samples_next_to_a_dropout_are_left_out(flights):
    # The derivative at the samples around a dropout uses the held values: they must go with it.
    log = _with_dropouts(_without(flights[0], "specific_force_body"))
    assert np.count_nonzero(~log.valid_mask()) > 20
    fit = fit_actuation(log, mass=MASS)
    assert fit.delay_s == pytest.approx(0.006, abs=1e-9)
    assert fit.thrust_scale == pytest.approx(0.94, rel=0.003)
    assert fit.motor_time_constant_up_s == pytest.approx(0.028, rel=0.03)
    assert fit.r_squared > 0.999
    assert fit_drag(log, mass=MASS).rotor_drag_coefficients_n_s_per_m == pytest.approx((0.22, 0.30), rel=0.03)


def test_position_only_log_keeps_the_timing(flights):
    # Two central differences average the force twice; the model is averaged the same way.
    fit = fit_actuation(_without(flights[0], "velocity", "specific_force_body"), mass=MASS)
    assert fit.delay_s == pytest.approx(0.006, abs=1e-9)
    # The attitude turns within the wider window, which costs a little of the scale.
    assert fit.thrust_scale == pytest.approx(0.94, rel=0.01)


def test_rotor_tilt_stays_out_of_the_thrust_scale():
    params_path = DEFAULT_PATH_RESOLVER.repo_root / "vehicle_params.tilted_rotor_example.json"
    plant = {"plant": {"actuation": {"delay_s": 0.006, "motor_time_constant_up_s": 0.028, "thrust_scale": 1.0}}}
    with Simulator(params_path, overrides=plant) as sim:
        log = excitation_flight(sim, duration_s=6.0, seed=3)
    result = identify(log, load_vehicle_params(params_path))
    # The rotors lean (axis z about 0.97): that is geometry the simulator applies itself, not a thrust error.
    assert result["plant"]["actuation"]["thrust_scale"] == pytest.approx(1.0, rel=0.005)
    assert result["plant"]["actuation"]["delay_s"] == pytest.approx(0.006, abs=1e-9)


def test_battery_fit_refuses_what_it_cannot_identify(flights):
    identification, _ = flights
    channels = dict(identification.channels)
    channels["battery_current"] = np.full((len(identification), 1), -1.0)  # a log without a current sensor
    log = FlightLog(identification.time, channels)
    with pytest.raises(ValueError, match="measured current"):
        fit_battery(log)
    channels["battery_current"] = np.full((len(identification), 1), 10.0)
    with pytest.raises(ValueError, match="varies by only"):
        fit_battery(FlightLog(identification.time, channels))
    result = identify(log, load_vehicle_params())
    assert "battery" not in result["plant"] and "battery" not in result["fits"]
    assert any(note.startswith("battery not identified") for note in result["notes"])
    assert set(result["plant"]) == {"actuation", "aerodynamics"}


def _truth_flight(plant: dict, duration_s: float = 8.0, seed: int = 4, steps_per_update: int | None = None) -> FlightLog:
    """A flight logged at every physics step (the estimators' logs are at the control period).

    ``steps_per_update``: physics steps per controller update (default: the simulator's control period).
    """
    with Simulator(overrides={"plant": plant}) as sim:
        state = sim.reset()
        config = build_hover_controller_config(sim.scene.params)
        rng = np.random.default_rng(seed)
        start = np.asarray(state["position"], dtype=float)
        rows = {name: [] for name in ("position", "velocity", "quaternion", "rotor_thrust_command")}
        times = []
        perturbation, until = 0.0, 0.0
        steps_per_update = steps_per_update or sim.steps_per_control_period
        for _ in range(int(round(duration_s / (steps_per_update * sim.control_period / sim.steps_per_control_period)))):
            time = float(state["time"])
            thrusts = compute_hover_control(state, start + [0.0, 0.0, 1.0], config)
            if time >= until:
                perturbation, until = rng.choice([-0.2, 0.2]), time + rng.uniform(0.04, 0.16)
            thrusts = np.clip(thrusts + (perturbation if time > 2.0 else 0.0), 0.0, config.max_rotor_thrust)
            for _ in range(steps_per_update):
                truth = sim.truth()
                times.append(sim.time)
                for name in ("position", "velocity", "quaternion"):
                    rows[name].append(truth[name])
                rows["rotor_thrust_command"].append(thrusts)
                state = sim.step(thrusts, steps=1)
    return FlightLog(np.asarray(times), {name: np.asarray(values, dtype=float) for name, values in rows.items()}).window(start=2.0)


def test_converted_log_with_the_default_mapping_keeps_the_delay():
    """Position at 100 Hz off the controller's ticks, motor commands at the controller's 200 Hz: the
    default time base is the command topic, so a command change is not recorded late."""
    truth = _truth_flight({"actuation": {"delay_s": 0.012, "motor_time_constant_up_s": 0.03, "thrust_scale": 0.95}})
    milliseconds = np.round(truth.time * 1000.0).astype(int)
    flip = np.diag([1.0, -1.0, -1.0])
    position, velocity = truth["position"] @ flip, truth["velocity"] @ flip
    attitude = rotation_to_quaternion(flip @ quaternion_to_rotation(truth["quaternion"]) @ flip)

    def topic(index: np.ndarray, fields: dict) -> dict:
        return {"timestamp": (milliseconds[index] * 1000).astype(np.uint64), **fields}

    slow = np.nonzero(milliseconds % 10 == 3)[0]
    fast = np.nonzero(milliseconds % 4 == 1)[0]
    control = np.nonzero(milliseconds % 5 == 0)[0]
    table = {
        "vehicle_local_position": topic(slow, {name: values[slow, k] for name, values, k in (("x", position, 0), ("y", position, 1), ("z", position, 2), ("vx", velocity, 0), ("vy", velocity, 1), ("vz", velocity, 2))}),
        "vehicle_attitude": topic(fast, {f"q[{k}]": attitude[fast, k] for k in range(4)}),
        "actuator_motors": topic(control, {f"control[{k}]": truth["rotor_thrust_command"][control, k] / 20.0 for k in range(4)}),
    }
    curve = {"coefficients": [[0.0], [20.0]], "voltage": 1.0}
    log = convert_topics(table, {"thrust_curve": curve})
    assert log.metadata["time_base"] == {"topic": "actuator_motors"}
    fit = fit_actuation(log, mass=MASS)
    assert fit.delay_s == pytest.approx(0.012, abs=1e-9)
    assert fit.thrust_scale == pytest.approx(0.95, rel=0.01)
    # On the position topic's time base each command change is recorded up to one sample late: the delay comes out short.
    late = fit_actuation(convert_topics(table, {"thrust_curve": curve, "time_base": {"topic": "vehicle_local_position"}}), mass=MASS)
    assert late.delay_s < 0.010


def test_log_slower_than_the_controller_needs_sampled_commands():
    """The controller updates every millisecond, the log keeps every fifth sample. Held, each logged
    command stands for the updates up to the next sample, half a sample late on average; marked
    "sampled", the commands are interpolated and the delay is right."""
    truth = _truth_flight({"actuation": {"delay_s": 0.012, "motor_time_constant_up_s": 0.03, "thrust_scale": 0.95}}, duration_s=7.0, steps_per_update=1)
    keep = np.arange(1, len(truth), 5)
    channels = {name: values[keep] for name, values in truth.channels.items()}
    held = fit_actuation(FlightLog(truth.time[keep], channels), mass=MASS)
    sampled = fit_actuation(FlightLog(truth.time[keep], channels, {"commands": "sampled"}), mass=MASS)
    assert sampled.delay_s == pytest.approx(0.012, abs=1e-9)
    assert sampled.thrust_scale == pytest.approx(0.95, rel=0.01)
    assert held.delay_s <= 0.011
    with pytest.raises(ValueError, match='"commands"'):
        FlightLog(truth.time[keep], channels, {"commands": "interpolated"})
