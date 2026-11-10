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


def test_frames_are_nwu_and_flu():
    log = convert_topics(_table())
    assert log.time[0] == 0.0 and log.sample_period == pytest.approx(0.01)
    assert log["position"][50] == pytest.approx([0.5, -1.0, 1.5])  # NED -> NWU: [x, -y, -z]
    assert log["velocity"][50] == pytest.approx([1.0, -2.0, 1.0])
    assert log["angular_velocity_body"][50] == pytest.approx([0.1, -0.2, -0.3])  # FRD -> FLU
    rotation_ned = quaternion_to_rotation(np.array([[np.cos(0.15), 0.0, 0.0, np.sin(0.15)]]))[0]
    assert quaternion_to_rotation(log["quaternion"][50:51])[0] == pytest.approx(D @ rotation_ned @ D)
    # A yaw of +0.3 in NED (clockwise seen from above) is -0.3 about the up axis.
    assert rotation_to_quaternion(quaternion_to_rotation(log["quaternion"][50:51]))[0] == pytest.approx([np.cos(0.15), 0.0, 0.0, -np.sin(0.15)])


def test_gaps_mark_samples_invalid_and_motor_order_and_thrust_curve():
    mapping = {"motor_order": [3, 2, 1, 0], "thrust_curve": {"coefficients": [[0.0, 0.1], [2.0]], "voltage": "battery"}}
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
