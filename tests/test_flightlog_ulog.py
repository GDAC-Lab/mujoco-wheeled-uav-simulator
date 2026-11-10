"""PX4 topics to a FlightLog: frames, time base, gaps, thrust curve (synthetic topic tables, no pyulog needed)."""

from __future__ import annotations

import numpy as np
import pytest

from wheeled_uav.flightlog import quaternion_to_rotation, rotation_to_quaternion
from wheeled_uav.flightlog_ulog import convert_topics

D = np.diag([1.0, -1.0, -1.0])


def _table():
    position_stamps = np.arange(0, 2_000_000, 10_000)  # 100 Hz, microseconds
    fast_stamps = np.arange(0, 2_000_000, 4_000)  # 250 Hz
    t = position_stamps * 1e-6
    yaw = 0.3
    # Body FRD to NED: a yaw of 0.3 rad about the NED down axis.
    q_ned = np.tile([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)], (fast_stamps.size, 1))
    motors = np.column_stack([0.3 + 0.01 * k + 0.0 * fast_stamps for k in range(4)])
    keep = (fast_stamps < 800_000) | (fast_stamps > 900_000)  # a 100 ms gap in the attitude
    return {
        "vehicle_local_position": {
            "timestamp": position_stamps,
            "x": t, "y": 2 * t, "z": -1.0 - t,
            "vx": np.ones_like(t), "vy": 2 * np.ones_like(t), "vz": -np.ones_like(t),
        },
        "vehicle_attitude": {"timestamp": fast_stamps[keep], **{f"q[{i}]": q_ned[keep, i] for i in range(4)}},
        "vehicle_angular_velocity": {"timestamp": fast_stamps, "xyz[0]": np.full(fast_stamps.size, 0.1), "xyz[1]": np.full(fast_stamps.size, 0.2), "xyz[2]": np.full(fast_stamps.size, 0.3)},
        "actuator_motors": {"timestamp": fast_stamps, **{f"control[{i}]": motors[:, i] for i in range(4)}},
        "battery_status": {"timestamp": position_stamps, "voltage_v": np.full(t.size, 16.0), "current_a": np.full(t.size, 10.0)},
    }


POSITION_BASE = {"time_base": {"topic": "vehicle_local_position"}}


def test_frames_are_nwu_and_flu():
    log = convert_topics(_table(), POSITION_BASE)
    assert log.time[0] == 0.0 and log.sample_period == pytest.approx(0.01)
    assert log["position"][50] == pytest.approx([0.5, -1.0, 1.5])  # NED -> NWU: [x, -y, -z]
    assert log["velocity"][50] == pytest.approx([1.0, -2.0, 1.0])
    assert log["angular_velocity_body"][50] == pytest.approx([0.1, -0.2, -0.3])  # FRD -> FLU
    rotation_ned = quaternion_to_rotation(np.array([[np.cos(0.15), 0.0, 0.0, np.sin(0.15)]]))[0]
    assert quaternion_to_rotation(log["quaternion"][50:51])[0] == pytest.approx(D @ rotation_ned @ D)
    # A yaw of +0.3 in NED (clockwise seen from above) is -0.3 about the up axis.
    assert rotation_to_quaternion(quaternion_to_rotation(log["quaternion"][50:51]))[0] == pytest.approx([np.cos(0.15), 0.0, 0.0, -np.sin(0.15)])


def test_gaps_mark_samples_invalid_and_motor_order_and_thrust_curve():
    mapping = {**POSITION_BASE, "motor_order": [3, 2, 1, 0], "thrust_curve": {"coefficients": [[0.0, 0.1], [2.0]], "voltage": "battery"}}
    log = convert_topics(_table(), mapping, start_s=0.5, end_s=1.5)
    assert log.time[0] == pytest.approx(0.5)
    invalid = log.time[~log.valid_mask()]
    # The attitude topic jumps from 0.796 s to 0.904 s: only the grid samples between them are invalid.
    assert invalid == pytest.approx(np.arange(0.80, 0.905, 0.01))
    assert log["rotor_command"][0] == pytest.approx([0.33, 0.32, 0.31, 0.30])
    # T = 0.1 V + 2 u
    assert log["rotor_thrust_command"][0] == pytest.approx(0.1 * 16.0 + 2.0 * np.array([0.33, 0.32, 0.31, 0.30]))


def test_missing_required_topics_and_unknown_keys_are_errors():
    table = _table()
    del table["vehicle_attitude"]
    with pytest.raises(ValueError, match="quaternion"):
        convert_topics(table)
    with pytest.raises(ValueError, match="unknown mapping key"):
        convert_topics(_table(), {"motor_ordr": [0, 1, 2, 3]})
    assert len(convert_topics(_table(), {"_source": {"vehicle": "x.json"}})) > 0  # keys starting with _ are comments


def test_default_time_base_is_the_command_topic():
    # Commands are held from the sample that records them, so their own instants are the samples.
    log = convert_topics(_table())
    assert log.metadata["time_base"] == {"topic": "actuator_motors"}
    assert log.sample_period == pytest.approx(0.004)
    assert log["position"][5] == pytest.approx([0.02, -0.04, 1.02])  # interpolated between the 100 Hz samples
    table = _table()
    del table["actuator_motors"]
    log = convert_topics(table)
    assert log.metadata["time_base"] == {"topic": "vehicle_local_position"} and log.sample_period == pytest.approx(0.01)


def test_measurements_sit_at_the_time_they_were_taken():
    table = _table()
    position = table["vehicle_local_position"]
    position["timestamp_sample"] = position["timestamp"] - 2_000  # taken 2 ms before it was published
    position["timestamp_sample"][0] = 1  # the first stamp is 0 here; a sample time of 0 would read as "not set"
    log = convert_topics(table)
    index = 100
    # x = t at the publication stamps, so at its sample time the value is 2 ms ahead of the log's clock.
    assert log["position"][index, 0] == pytest.approx(log.time[index] + 0.002)
    published = convert_topics(table, {"position": {"stamp": "timestamp"}})
    assert published["position"][index, 0] == pytest.approx(published.time[index])
    # Commands stay at their publication stamps, the instant they were issued.
    table["actuator_motors"]["timestamp_sample"] = table["actuator_motors"]["timestamp"] - 1_000
    table["actuator_motors"]["timestamp_sample"][0] = 1
    assert convert_topics(table).time[:3] == pytest.approx([0.0, 0.004, 0.008])
    with pytest.raises(ValueError, match="time stamp field"):
        convert_topics(table, {"position": {"stamp": "timestamp_smaple"}})


def test_sampled_commands_are_marked_and_interpolated():
    table = _table()
    table["actuator_motors"]["control[0]"] = table["actuator_motors"]["timestamp"] * 1e-6  # a ramp: u = t
    held = convert_topics(table, POSITION_BASE)
    sampled = convert_topics(table, {**POSITION_BASE, "commands": "sampled"})
    assert held.metadata["commands"] == "held" and sampled.metadata["commands"] == "sampled"
    # Between two log samples the held command stays, the sampled one follows the ramp.
    between = np.array([0.505])
    assert held.sample("rotor_command", between)[0, 0] == pytest.approx(0.5)
    assert sampled.sample("rotor_command", between)[0, 0] == pytest.approx(0.505)
    assert sampled.window(0.2, 1.0).metadata["commands"] == "sampled"
    with pytest.raises(ValueError, match='"commands"'):
        convert_topics(table, {"commands": "hold"})
