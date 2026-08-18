from __future__ import annotations

import time
import unittest

from wheeled_uav.timing import (
    NullPacer,
    RealtimeTracker,
    SessionTimingTracker,
    SimulationTimingConfig,
    StateSampleTracker,
    StepPacer,
    build_pacer,
    build_state_sample_key,
    compute_control_dt_seconds,
    extract_sync_metrics,
    parse_simulation_timing,
)


class TimingModuleTests(unittest.TestCase):
    def test_parse_simulation_timing_resolves_control_period(self) -> None:
        timing = parse_simulation_timing(
            {
                "simulation": {
                    "timestep": 0.001,
                    "state_publish_every_n_steps": 5,
                    "viewer_fps": 60.0,
                }
            }
        )
        self.assertAlmostEqual(timing.control_period_seconds, 0.005)
        self.assertEqual(timing.state_publish_every_n_steps, 5)
        self.assertEqual(timing.pacing_mode, "realtime")

    def test_parse_simulation_timing_accepts_accelerated_mode(self) -> None:
        timing = parse_simulation_timing({"simulation": {"timestep": 0.001, "pacing_mode": "accelerated"}})
        self.assertEqual(timing.pacing_mode, "accelerated")

    def test_build_pacer_selects_implementation_by_mode(self) -> None:
        realtime_timing = SimulationTimingConfig(
            physics_timestep_seconds=0.001,
            state_publish_every_n_steps=1,
            control_period_seconds=0.001,
            viewer_fps=60.0,
            pacing_mode="realtime",
        )
        accelerated_timing = SimulationTimingConfig(
            physics_timestep_seconds=0.001,
            state_publish_every_n_steps=1,
            control_period_seconds=0.001,
            viewer_fps=60.0,
            pacing_mode="accelerated",
        )
        self.assertIsInstance(build_pacer(realtime_timing), StepPacer)
        self.assertIsInstance(build_pacer(accelerated_timing), NullPacer)

    def test_build_state_sample_key_uses_sequence_and_time(self) -> None:
        key = build_state_sample_key({"sequence": 42, "time": 0.12345678901234567})
        self.assertIn("seq=42", key)
        self.assertTrue(key.startswith("seq=42|t="))

    def test_state_sample_tracker_drops_duplicates(self) -> None:
        tracker = StateSampleTracker()
        state = {"sequence": 1, "time": 0.1}
        self.assertTrue(tracker.is_new(state))
        self.assertFalse(tracker.is_new(state))
        self.assertTrue(tracker.is_new({"sequence": 2, "time": 0.1}))

    def test_compute_control_dt_clamps_outliers(self) -> None:
        timing = SimulationTimingConfig(
            physics_timestep_seconds=0.001,
            state_publish_every_n_steps=5,
            control_period_seconds=0.005,
            viewer_fps=60.0,
            pacing_mode="realtime",
        )
        self.assertAlmostEqual(compute_control_dt_seconds(None, 0.1, timing), 0.005)
        self.assertAlmostEqual(compute_control_dt_seconds(0.1, 0.1, timing), 0.005)
        self.assertAlmostEqual(compute_control_dt_seconds(0.1, 0.105, timing), 0.005)
        self.assertAlmostEqual(compute_control_dt_seconds(0.1, 0.5, timing), 0.015)

    def test_extract_sync_metrics_reads_timing_block(self) -> None:
        metrics = extract_sync_metrics(
            {
                "time": 1.25,
                "realtime_factor": 0.98,
                "wall_time_send_ns": 1_000_000,
                "timing": {
                    "control_period_seconds": 0.005,
                    "sim_wall_skew_seconds": -0.02,
                    "session_wall_elapsed_seconds": 1.27,
                },
            },
            receive_time_ns=4_500_000,
        )
        self.assertAlmostEqual(metrics["sim_time_seconds"], 1.25)
        self.assertAlmostEqual(metrics["packet_age_ms"], 3.5)
        self.assertAlmostEqual(metrics["sim_wall_skew_seconds"], -0.02)

    def test_realtime_tracker_converges_to_unit_rate(self) -> None:
        tracker = RealtimeTracker()
        for _ in range(600):
            tracker.update(0.001, 0.001)
        self.assertAlmostEqual(tracker.realtime_factor, 1.0, delta=0.01)

    def test_session_timing_tracker_skew_near_zero_at_unit_rate(self) -> None:
        timing = SimulationTimingConfig(
            physics_timestep_seconds=0.001,
            state_publish_every_n_steps=1,
            control_period_seconds=0.001,
            viewer_fps=60.0,
            pacing_mode="realtime",
        )
        session = SessionTimingTracker.start(0.0, timing=timing)
        time.sleep(0.05)
        snapshot = session.snapshot(0.05, 1.0)
        self.assertLess(abs(snapshot["sim_wall_skew_seconds"]), 0.02)


if __name__ == "__main__":
    unittest.main()
