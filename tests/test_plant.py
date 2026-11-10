"""The plant model (wheeled_uav.plant), checked against closed-form results."""

from __future__ import annotations

import json
import math

import mujoco
import numpy as np
import pytest

from wheeled_uav.config import load_vehicle_params, parse_total_mass
from wheeled_uav.plant import (
    FEATURE_NAMES,
    ActuationConfig,
    BatteryConfig,
    CorrectionModel,
    apply_randomization,
    fit_knn_correction,
    fit_linear_correction,
    parse_plant_config,
)
from wheeled_uav.plant.actuation import BatteryPacks, RotorActuation
from wheeled_uav.runtime.scene import SimulationRequest, load_simulation_scene
from wheeled_uav.sim import Simulator

DT = 0.001
# Airborne, no gravity: forces show up as clean accelerations.
FLOATING = {"simulation": {"gravity": [0.0, 0.0, 0.0]}, "drone": {"initial_position": [0.0, 0.0, 1.0]}}


def _merge(*parts):
    from wheeled_uav.config import merge_params

    merged: dict = {}
    for part in parts:
        merged = merge_params(merged, part)
    return merged


def _actuation(**kwargs) -> RotorActuation:
    options = dict(timestep=DT, num_uavs=1, rotors_per_uav=4, ctrl_limit=20.0, thrust_coefficient=3.0e-5)
    voltage_compensation = kwargs.pop("voltage_compensation", True)
    return RotorActuation(ActuationConfig(**kwargs), voltage_compensation=voltage_compensation, **options)


# -- configuration ----------------------------------------------------------------------------------


def test_no_plant_section_means_no_plant_model():
    assert parse_plant_config({}) is None
    assert parse_plant_config({"plant": {}}) is None
    assert parse_plant_config({"plant": {"enabled": False, "actuation": {"delay_s": 0.01}}}) is None
    assert parse_plant_config({"plant": {"actuation": {"enabled": False, "delay_s": 0.01}}}) is None


def test_plant_config_rejects_unknown_keys_and_bad_values():
    with pytest.raises(ValueError, match="motor_tau_up_s"):
        parse_plant_config({"plant": {"actuation": {"motor_tau_up_s": 0.03}}})
    with pytest.raises(ValueError, match="must be at least 0"):
        parse_plant_config({"plant": {"actuation": {"delay_s": -0.01}}})
    with pytest.raises(ValueError, match="probability"):
        parse_plant_config({"plant": {"sensors": {"pose_dropout_probability": 1.5}}})
    with pytest.raises(ValueError, match="below full_voltage_v"):
        parse_plant_config({"plant": {"battery": {"empty_voltage_v": 17.0}}})
    with pytest.raises(ValueError, match="list of 4"):
        parse_plant_config({"plant": {"actuation": {"rotor_thrust_scales": [1.0, 1.0]}}})
    config = parse_plant_config({"plant": {"_note": "comments are fine", "actuation": {"_source": "bench", "motor_time_constant_up_s": 0.03}}})
    assert config.actuation.motor_time_constant_up_s == 0.03
    assert config.actuation.motor_time_constant_down_s == 0.03  # defaults to the up value


def test_plant_and_hil_actuator_lag_cannot_both_model_the_rotors(tmp_path):
    params = load_vehicle_params()
    params["fidelity_mode"] = "hil"
    params["actuator_dynamics"]["motor_tau_ms"] = 30.0
    params["plant"] = {"actuation": {"motor_time_constant_up_s": 0.03}}
    with pytest.raises(ValueError, match="both model the rotors"):
        load_simulation_scene(SimulationRequest(generated_xml_dir=tmp_path), params=params)


# -- rotors -----------------------------------------------------------------------------------------


def test_command_delay_is_a_whole_number_of_steps():
    actuation = _actuation(delay_s=0.003)
    actuation.settle(np.zeros(4))
    delivered = [actuation.step(np.full(4, 2.0))[0] for _ in range(5)]
    assert delivered == pytest.approx([0.0, 0.0, 0.0, 2.0, 2.0], abs=1e-12)


def test_rotor_speed_follows_a_first_order_lag_up_and_down():
    tau_up, tau_down = 0.02, 0.05
    actuation = _actuation(motor_time_constant_up_s=tau_up, motor_time_constant_down_s=tau_down)
    actuation.settle(np.full(4, 1.0))
    for step in range(1, 41):
        thrust = actuation.step(np.full(4, 4.0))[0]
        # sqrt(T) ~ rotor speed: 1 -> 2 with time constant tau_up, exactly.
        assert math.sqrt(thrust) == pytest.approx(2.0 - math.exp(-step * DT / tau_up), rel=1e-12)
    start = math.sqrt(thrust)
    for step in range(1, 41):
        thrust = actuation.step(np.full(4, 1.0))[0]
        assert math.sqrt(thrust) == pytest.approx(1.0 + (start - 1.0) * math.exp(-step * DT / tau_down), rel=1e-12)


def test_scale_saturation_and_idle():
    actuation = _actuation(thrust_scale=0.9, rotor_thrust_scales=(1.0, 1.0, 1.0, 0.5), max_thrust_n=3.0, idle_thrust_n=0.2)
    delivered = actuation.step(np.array([2.0, 5.0, 0.1, 2.0]))
    assert delivered == pytest.approx([1.8, 3.0, 0.2, 0.9])
    assert actuation.saturated.tolist() == [False, True, False, False]
    assert actuation.step(np.zeros(4)) == pytest.approx(np.zeros(4))  # a zero command stops the rotors
    actuation.settle(np.full(4, 2.7))
    assert actuation.rotor_speeds() == pytest.approx(np.sqrt(np.array([2.43, 2.43, 2.43, 1.215]) / 3.0e-5))


def test_battery_voltage_solves_the_power_balance():
    config = BatteryConfig(full_voltage_v=16.8, empty_voltage_v=13.6, capacity_mah=1500.0, internal_resistance_ohm=0.04, power_coefficient_w_per_n1p5=12.0, auxiliary_power_w=5.0)
    packs = BatteryPacks(config, num_uavs=2)
    thrusts = np.array([[2.5, 2.5, 2.5, 2.5], [0.0, 0.0, 0.0, 0.0]])
    packs.update(thrusts, 0.01)
    power = 12.0 * 4 * 2.5**1.5 + 5.0
    open_circuit = 16.8  # full
    assert packs.power[0] == pytest.approx(power)
    assert packs.voltage[0] * (open_circuit - packs.voltage[0]) / 0.04 == pytest.approx(power, rel=1e-9)
    assert packs.current[0] == pytest.approx(power / packs.voltage[0])
    assert packs.charge[0] == pytest.approx(1.0 - packs.current[0] * 0.01 / (1500.0 * 3.6))
    assert packs.current[1] == pytest.approx(5.0 / packs.voltage[1])
    open_circuit = packs.open_circuit_voltage()
    packs.update(np.full((2, 4), 1.0e4), 0.001)  # far past the pack's maximum power
    assert packs.overloaded.all()
    assert packs.voltage == pytest.approx(0.5 * open_circuit)
    assert packs.current == pytest.approx(open_circuit / (2.0 * 0.04))  # the maximum power, V_oc^2 / 4R


def test_without_voltage_compensation_thrust_scales_with_voltage_squared():
    compensated = _actuation(max_thrust_n=8.0)
    uncompensated = _actuation(max_thrust_n=8.0, voltage_compensation=False)
    ratio = np.array([15.0 / 16.0])
    assert compensated.step(np.full(4, 2.0), ratio) == pytest.approx(np.full(4, 2.0))
    assert uncompensated.step(np.full(4, 2.0), ratio) == pytest.approx(np.full(4, 2.0 * ratio[0] ** 2))
    assert compensated.step(np.full(4, 9.0), ratio) == pytest.approx(np.full(4, 8.0 * ratio[0] ** 2))  # sagging maximum


def test_battery_at_its_power_limit_settles_without_rotor_lag():
    # No rotor lag: the voltage sets the thrust limit and the thrust sets the voltage within one step.
    # Solved together they settle; taking the previous step's voltage they would alternate between two states.
    with Simulator(overrides=_merge(FLOATING, {"plant": {"battery": {}}})) as sim:
        sim.reset()
        voltages, thrusts = [], []
        for _ in range(12):
            state = sim.step([14.0] * 4, steps=1)
            voltages.append(state["plant"]["battery_voltage_v"])
            thrusts.append(sim.runtime.plant.delivered_thrusts(0).copy())
        plant = sim.runtime.plant
        assert np.ptp(voltages) < 0.02 and np.ptp(np.array(thrusts)[:, 0]) < 0.02
        assert plant.actuation.saturated.all()
        # The thrust is the limit at that voltage, and the voltage is the pack's under that thrust.
        ratio = voltages[-1] / plant.battery.config.reference_voltage_v
        assert thrusts[-1] == pytest.approx(np.full(4, plant.actuation.max_thrust * ratio**2), rel=1e-6)
        assert plant.battery.terminal_voltage(thrusts[-1][None, :])[0] == pytest.approx(voltages[-1], abs=1e-3)


def test_battery_sags_and_drains_in_flight():
    plant = {"plant": {"battery": {"capacity_mah": 100.0, "internal_resistance_ohm": 0.05}}}
    with Simulator(overrides=_merge(FLOATING, plant)) as sim:
        first = sim.reset()["plant"]
        assert first["battery_voltage_v"] == pytest.approx(16.8)
        for _ in range(200):
            state = sim.step([2.45] * 4)
        late = state["plant"]
    assert late["battery_current_a"] > 5.0
    assert late["battery_voltage_v"] < 16.8 - 0.05 * late["battery_current_a"] + 1e-6
    assert late["battery_charge"] < 1.0


# -- forces -----------------------------------------------------------------------------------------


def _push(sim: Simulator, velocity) -> None:
    data = sim.scene.data
    data.qvel[0:3] = velocity
    mujoco.mj_forward(sim.scene.model, data)


def test_rotor_drag_decays_velocity_exponentially():
    mass = parse_total_mass(load_vehicle_params())
    plant = {"plant": {"aerodynamics": {"rotor_drag_coefficients_n_s_per_m": [0.3, 0.3, 0.0]}}}
    with Simulator(overrides=_merge(FLOATING, plant)) as sim:
        sim.reset()
        _push(sim, [2.0, 0.0, 0.0])
        for _ in range(100):
            sim.step([0.0] * 4)
        # Explicit force per step: v_k = v_0 (1 - d dt / m)^k.
        assert sim.truth()["velocity"][0] == pytest.approx(2.0 * (1.0 - 0.3 * DT / mass) ** 500, rel=1e-9)


def test_body_drag_is_quadratic_and_wind_drives_the_vehicle():
    mass = parse_total_mass(load_vehicle_params())
    plant = {"plant": {"aerodynamics": {"body_drag_area_m2": [0.02, 0.02, 0.02], "air_density_kg_per_m3": 1.2}}}
    with Simulator(overrides=_merge(FLOATING, plant)) as sim:
        sim.reset()
        _push(sim, [3.0, 0.0, 0.0])
        for _ in range(100):
            sim.step([0.0] * 4)
        k = 0.5 * 1.2 * 0.02 / mass
        assert sim.truth()["velocity"][0] == pytest.approx(3.0 / (1.0 + k * 3.0 * 0.5), rel=2e-3)
    windy = {"plant": {"aerodynamics": {"rotor_drag_coefficients_n_s_per_m": [0.4, 0.4, 0.4], "wind_m_per_s": [1.0, 0.0, 0.0]}}}
    with Simulator(overrides=_merge(FLOATING, windy)) as sim:
        sim.reset()
        state = sim.step([0.0] * 4, steps=1)
        for _ in range(400):
            state = sim.step([0.0] * 4)
        velocity = sim.truth()["velocity"]
    assert velocity[0] == pytest.approx(1.0 - math.exp(-0.4 * 2.005 / mass), abs=2e-3)  # towards the wind speed
    assert state["plant"]["plant_force_body"][0] > 0.0


def test_ground_effect_raises_thrust_near_the_ground():
    radius, height = 0.12, 0.06  # R / 4z = 0.5: T / T_inf = 1 / (1 - 0.25)
    propeller_z = load_vehicle_params()["drone"]["propeller"]["z"]
    plant = {"plant": {"aerodynamics": {"ground_effect": {"rotor_radius_m": radius, "max_gain": 2.0}}}}
    with Simulator(overrides=_merge(FLOATING, {"drone": {"initial_position": [0.0, 0.0, height - propeller_z]}}, plant)) as sim:
        sim.reset()
        state = sim.step([2.0] * 4, steps=1)
        assert sim.runtime.plant.delivered_thrusts(0) == pytest.approx(np.full(4, 2.0 / 0.75), rel=1e-9)
        assert state["position"][2] == pytest.approx(height - propeller_z)
    with Simulator(overrides=_merge(FLOATING, {"drone": {"initial_position": [0.0, 0.0, 0.01]}}, plant)) as sim:
        sim.reset()
        sim.step([2.0] * 4, steps=1)
        assert sim.runtime.plant.delivered_thrusts(0) == pytest.approx(np.full(4, 4.0))  # capped at max_gain
        # What is reported as applied stays within the actuator's range, as MuJoCo clamps it there.
        limit = load_vehicle_params()["actuation"]["max_rotor_thrust"]
        sim.step([limit] * 4, steps=1)
        assert sim.runtime.plant.delivered_thrusts(0) == pytest.approx(np.full(4, limit))
        assert sim.runtime.plant.requested_and_applied()[1] == pytest.approx(np.full(4, limit))


# -- correction -------------------------------------------------------------------------------------


def test_linear_and_knn_corrections_fit_and_round_trip(tmp_path):
    rng = np.random.default_rng(1)
    features = rng.normal(size=(400, 3))
    weights = np.array([[0.5, -0.2, 0.0], [0.0, 0.1, 0.3]])
    targets = features @ weights.T + np.array([0.05, -0.1])
    names = ("velocity_body_x", "velocity_body_y", "collective_thrust")
    outputs = ("force_body_x", "force_body_z")
    linear = fit_linear_correction(features, targets, names, outputs, ridge=0.0)
    full = np.zeros(len(FEATURE_NAMES))
    full[[0, 1, 10]] = [0.3, -0.4, 0.2]
    expected = weights @ np.array([0.3, -0.4, 0.2]) + np.array([0.05, -0.1])
    assert linear.evaluate(full)[[0, 2]] == pytest.approx(expected, abs=1e-9)
    knn = fit_knn_correction(features, targets, names, outputs, k=1)
    full[[0, 1, 10]] = features[7]
    assert knn.evaluate(full)[[0, 2]] == pytest.approx(targets[7])
    for model in (linear, knn):
        model.save(tmp_path / "correction.json")
        loaded = CorrectionModel.from_file(tmp_path / "correction.json")
        assert loaded.evaluate(full) == pytest.approx(model.evaluate(full))
    full[[0, 1, 10]] = [100.0, 0.0, 0.0]  # far outside the data: clipped to the range seen
    assert linear.evaluate(full)[0] == pytest.approx(np.max(targets[:, 0]))


def test_malformed_correction_files_are_value_errors(tmp_path):
    model = fit_linear_correction(np.random.default_rng(0).normal(size=(20, 2)), np.zeros((20, 1)), FEATURE_NAMES[:2], ("force_body_x",))
    good = model.to_dict()
    for broken in ([1, 2], {**good, "linear": None}, {**good, "features": 3}, {**good, "outputs": [1]}, {**good, "provenance": "x"}):
        (tmp_path / "broken.json").write_text(json.dumps(broken), encoding="utf-8")
        with pytest.raises(ValueError, match="correction"):
            CorrectionModel.from_file(tmp_path / "broken.json")


def test_correction_force_acts_on_the_vehicle(tmp_path):
    mass = parse_total_mass(load_vehicle_params())
    correction = {
        "format": "wheeled_uav.correction/1",
        "kind": "linear",
        "features": ["airspeed"],
        "outputs": ["force_body_z"],
        "feature_mean": [0.0],
        "feature_scale": [1.0],
        "output_min": [-1.0],
        "output_max": [1.0],
        "linear": {"weights": [[0.0]], "bias": [0.5]},
    }
    (tmp_path / "correction.json").write_text(json.dumps(correction), encoding="utf-8")
    plant = {"plant": {"correction": {"file": str(tmp_path / "correction.json")}}}
    with Simulator(overrides=_merge(FLOATING, plant)) as sim:
        sim.reset()
        for _ in range(100):
            sim.step([0.0] * 4)
        assert sim.truth()["velocity"][2] == pytest.approx(0.5 / mass * 0.5, rel=1e-9)


# -- sensors ----------------------------------------------------------------------------------------


def _positions(overrides, count=40, thrust=0.0):
    with Simulator(overrides=overrides) as sim:
        states = [sim.reset()] + [sim.step([thrust] * 4) for _ in range(count)]
    return states


def test_pose_delay_returns_the_state_from_that_long_ago():
    falling = {"drone": {"initial_position": [0.0, 0.0, 3.0]}}
    plain = _positions(falling)
    delayed = _positions(_merge(falling, {"plant": {"sensors": {"pose_delay_s": 0.010, "angular_velocity_delay_s": 0.0}}}))
    # 10 ms is two control periods of 5 ms.
    for index in range(3, len(plain)):
        assert delayed[index]["position"] == plain[index - 2]["position"]
        assert delayed[index]["plant"]["measurement_age_s"] == pytest.approx(0.010)
    assert delayed[index]["time"] == plain[index]["time"]  # stamped with the publish time


def test_sensor_noise_bias_and_seeds():
    noisy = _merge(FLOATING, {"plant": {"sensors": {"position_noise_std_m": 0.01, "angular_velocity_bias_std_rad_per_s": 0.05}}})
    with Simulator(overrides=noisy, seed=3) as sim:
        states = [sim.reset()] + [sim.step([0.0] * 4) for _ in range(400)]
        bias = sim.runtime.plant.sensors.gyro_bias[0].copy()
    errors = np.array([state["position"] for state in states]) - np.array([0.0, 0.0, 1.0])
    assert np.std(errors, axis=0) == pytest.approx([0.01, 0.01, 0.01], rel=0.15)
    rates = np.array([state["angular_velocity_body"] for state in states])
    assert rates == pytest.approx(np.tile(bias, (len(states), 1)), abs=1e-9)  # at rest: the gyro reads its bias
    with Simulator(overrides=noisy, seed=3) as sim:
        again = [sim.reset()] + [sim.step([0.0] * 4) for _ in range(400)]
    assert [state["position"] for state in again] == [state["position"] for state in states]
    with Simulator(overrides=noisy, seed=4) as sim:
        other = [sim.reset()] + [sim.step([0.0] * 4) for _ in range(5)]
    assert other[3]["position"] != states[3]["position"]


def test_pose_dropout_holds_the_last_pose():
    falling = {"drone": {"initial_position": [0.0, 0.0, 3.0]}}
    dropping = _merge(falling, {"plant": {"sensors": {"pose_dropout_probability": 1.0, "pose_dropout_mean_duration_s": 1000.0}}})
    states = _positions(dropping, count=20)
    assert all(state["position"] == states[0]["position"] for state in states)
    assert states[-1]["plant"]["measurement_age_s"] == pytest.approx(states[-1]["time"])
    assert states[-1]["angular_velocity_body"] is not None


# -- randomization ----------------------------------------------------------------------------------


def _randomized(seed, parameters):
    params = load_vehicle_params()
    params["randomization"] = {"parameters": parameters}
    return apply_randomization(params, seed=seed)


def test_randomization_is_reproducible_per_parameter():
    spec = {"drone.mass": {"uniform": [0.7, 0.9]}}
    first, draws = _randomized(1, spec)
    again, _ = _randomized(1, spec)
    other, _ = _randomized(2, spec)
    assert first["drone"]["mass"] == again["drone"]["mass"] != other["drone"]["mass"]
    assert 0.7 <= first["drone"]["mass"] <= 0.9
    assert draws == [{"path": "drone.mass", "value": first["drone"]["mass"], "distribution": {"uniform": [0.7, 0.9]}}]
    # Another parameter does not change this one's draw.
    more, _ = _randomized(1, {**spec, "plant.actuation.thrust_scale": {"normal": [1.0, 0.05]}})
    assert more["drone"]["mass"] == first["drone"]["mass"]


def test_randomization_scales_keep_proportions_and_clip():
    params, _ = _randomized(5, {"drone.inertia": {"scale_uniform": [0.5, 1.5]}, "simulation.timestep": {"normal": [0.001, 1.0], "clip": [0.0005, 0.002]}})
    base = np.array(load_vehicle_params()["drone"]["inertia"])
    scaled = np.array(params["drone"]["inertia"])
    factor = scaled[0, 0] / base[0, 0]
    assert scaled == pytest.approx(base * factor)
    assert 0.0005 <= params["simulation"]["timestep"] <= 0.002
    with pytest.raises(ValueError, match="does not exist"):
        _randomized(1, {"drone.no_such.mass": {"uniform": [0, 1]}})
    with pytest.raises(ValueError, match="not set"):
        _randomized(1, {"plant.actuation.thrust_scale": {"scale_uniform": [0.9, 1.1]}})
    # A misspelt last name is an error too (outside plant nothing would read the new key).
    with pytest.raises(ValueError, match="'drone.mas' does not exist"):
        _randomized(1, {"drone.mas": {"uniform": [0.85, 0.95]}})
    with pytest.raises(ValueError, match="exactly one"):
        _randomized(1, {"drone.mass": {"uniform": [0, 1], "normal": [0, 1]}})
    with pytest.raises(ValueError, match="scale_uniform"):
        _randomized(1, {"drone.mass": {"scale_uniform": [[0.9], [1.1]]}})


def test_randomized_mass_reaches_the_model_and_is_recorded(tmp_path):
    params = load_vehicle_params()
    params["randomization"] = {"parameters": {"drone.mass": {"uniform": [0.6, 0.7]}}}
    scene = load_simulation_scene(SimulationRequest(generated_xml_dir=tmp_path, seed=9), params=params)
    drawn = scene.params["drone"]["mass"]
    assert 0.6 <= drawn <= 0.7
    assert scene.model.body(scene.uav_specs[0].body_name).mass[0] == pytest.approx(drawn)
    record = json.loads(scene.model_xml_path.with_name(f"{scene.model_xml_path.stem}.randomization.json").read_text())
    assert record == {"seed": 9, "draws": [dict(scene.randomization_draws[0])]}


# -- the whole loop -----------------------------------------------------------------------------------


def test_state_packet_carries_the_plant_block_and_actuator_log():
    overrides = {
        "logging_config": {"log_actuator_stats": True},
        "plant": {
            "actuation": {"motor_time_constant_up_s": 0.03, "delay_s": 0.002},
            "battery": {},
            "sensors": {"pose_delay_s": 0.01},
        },
    }
    with Simulator(overrides=overrides) as sim:
        sim.reset()
        state = sim.step([3.0] * 4)
    plant = state["plant"]
    assert set(plant) == {"rotor_speeds_rad_s", "battery_voltage_v", "battery_current_a", "battery_charge", "measurement_age_s"}
    assert state["actuator"]["requested_rotor_thrusts"] == [3.0] * 4
    applied = state["actuator"]["applied_rotor_thrusts"]
    # The first command settles the rotors (a vehicle already flying), so the lag starts from it.
    assert applied == pytest.approx([3.0] * 4)
    assert plant["rotor_speeds_rad_s"] == pytest.approx(np.sqrt(np.array(applied) / 3.0e-5))


def test_an_empty_or_disabled_plant_changes_nothing():
    reference = _positions({}, count=30, thrust=2.6)
    for plant in ({}, {"enabled": False, "actuation": {"motor_time_constant_up_s": 0.05}}):
        states = _positions({"plant": plant}, count=30, thrust=2.6)
        assert [state["position"] for state in states] == [state["position"] for state in reference]


def test_plant_with_parallel_physics_matches_the_shared_world():
    overrides = _merge(
        {"plant": {"actuation": {"motor_time_constant_up_s": 0.03, "delay_s": 0.002}, "aerodynamics": {"rotor_drag_coefficients_n_s_per_m": [0.3, 0.3, 0.1]}}},
        {"drone": {"initial_position": [0.0, 0.0, 1.0]}},
    )
    runs = []
    for threads in (0, 2):
        with Simulator(overrides=_merge(overrides, {"simulation": {"physics_threads": threads}}), num_uavs=2, spawn_radius=3.0) as sim:
            sim.reset()
            for step in range(60):
                thrust = 2.0 + 0.5 * math.sin(0.2 * step)
                sim.step([[thrust, thrust, thrust + 0.1, thrust], [thrust, thrust, thrust, thrust - 0.1]])
            runs.append([sim.truth(index)["position"] for index in range(2)])
    np.testing.assert_allclose(runs[1], runs[0], atol=1e-6)


def test_binary_state_packets_carry_the_measured_state_without_the_plant_block(tmp_path):
    from wheeled_uav.runtime.loop import SimulationRuntime
    from wheeled_uav.state_packets import decode_state_packet

    params = load_vehicle_params()
    params["plant"] = {"sensors": {"pose_delay_s": 0.01}, "battery": {}}
    scene = load_simulation_scene(SimulationRequest(generated_xml_dir=tmp_path, state_format="binary", headless=True), params=params)
    runtime = SimulationRuntime(scene)
    try:
        for _ in range(20):
            runtime.step_physics_once()
        state = decode_state_packet(runtime.state_publisher.build_payload(1.0, 1, 0, runtime.actuator_snapshot()))
    finally:
        runtime.close()
    assert "plant" not in state
    assert len(state["position"]) == 3
