from __future__ import annotations

import json
import time
import unittest

import mujoco

from wheeled_uav.runtime import SimulationRequest, build_state_payload, load_simulation_scene, run_headless_loop
from wheeled_uav.timing import SessionTimingTracker
from wheeled_uav.protocol import create_udp_socket


class SimulationTimingTests(unittest.TestCase):
    def test_state_payload_includes_timing_block(self) -> None:
        scene = load_simulation_scene(SimulationRequest(headless=True))
        timing = SessionTimingTracker.start(
            scene.data.time,
            control_period_seconds=0.005,
            publish_every_n_steps=5,
        ).snapshot(scene.data.time, 1.0)
        payload = json.loads(
            build_state_payload(
                scene,
                1.0,
                sequence=1,
                wall_time_send_ns=1_000_000,
                timing=timing,
            ).decode("utf-8")
        )
        self.assertIn("timing", payload)
        self.assertAlmostEqual(payload["timing"]["control_period_seconds"], 0.005)
        self.assertAlmostEqual(payload["sim_time"], scene.data.time)
        self.assertAlmostEqual(payload["time"], scene.data.time)

    def test_publish_after_step_uses_integrated_time(self) -> None:
        scene = load_simulation_scene(SimulationRequest(headless=True))
        for _ in range(5):
            mujoco.mj_step(scene.model, scene.data)
        self.assertAlmostEqual(scene.data.time, 0.005)

    def test_headless_run_tracks_real_time(self) -> None:
        request = SimulationRequest(headless=True, duration_seconds=1.0, pacing_mode="realtime")
        scene = load_simulation_scene(request)
        recv_port, _ = request.resolved_ports()
        sock = create_udp_socket(recv_port=recv_port)
        try:
            wall_start = time.perf_counter()
            run_headless_loop(scene, sock)
            wall_elapsed = time.perf_counter() - wall_start
        finally:
            sock.close()
        self.assertAlmostEqual(scene.data.time, 1.0, delta=0.02)
        self.assertAlmostEqual(wall_elapsed, 1.0, delta=0.2)

    def test_headless_accelerated_mode_finishes_faster_than_real_time(self) -> None:
        request = SimulationRequest(headless=True, duration_seconds=1.0, pacing_mode="accelerated")
        scene = load_simulation_scene(request)
        recv_port, _ = request.resolved_ports()
        sock = create_udp_socket(recv_port=recv_port)
        try:
            wall_start = time.perf_counter()
            run_headless_loop(scene, sock)
            wall_elapsed = time.perf_counter() - wall_start
        finally:
            sock.close()
        self.assertAlmostEqual(scene.data.time, 1.0, delta=0.02)
        self.assertLess(wall_elapsed, 0.75)


if __name__ == "__main__":
    unittest.main()
