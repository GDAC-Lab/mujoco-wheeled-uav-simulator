from __future__ import annotations

import time

from wheeled_uav.clock import wall_time_ns


def test_wall_time_follows_the_system_clock_at_sub_millisecond_resolution():
    readings = [wall_time_ns() for _ in range(2000)]
    steps = [later - earlier for earlier, later in zip(readings, readings[1:]) if later != earlier]

    # time.time_ns() advances in 15.6 ms steps on Windows before Python 3.13;
    # wall_time_ns() must resolve far below a millisecond everywhere.
    assert steps and min(steps) < 1_000_000
    # ... and stay within one system timer tick of the system clock.
    assert abs(wall_time_ns() - time.time_ns()) <= 16_000_000
