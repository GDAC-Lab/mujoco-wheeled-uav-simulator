from __future__ import annotations

import unittest
from pathlib import Path

import mujoco
import numpy as np

from wheeled_uav.config import load_vehicle_params
from wheeled_uav.model.builder import render_model_xml
from wheeled_uav.runtime import SimulationRequest, load_simulation_scene, run_headless_loop
from wheeled_uav.protocol import create_udp_socket


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROJECT_PARAMS = PROJECT_ROOT / "configs" / "vehicle" / "vehicle_params.project.json"


class WallSceneStabilityTests(unittest.TestCase):
    @unittest.skipUnless(PROJECT_PARAMS.is_file(), "project vehicle params not available")
    def test_project_wall_spawn_runs_startup_without_early_instability(self) -> None:
        generated_xml_dir = PROJECT_ROOT / "build" / "generated_xml"
        generated_xml_dir.mkdir(parents=True, exist_ok=True)
        request = SimulationRequest(
            headless=True,
            duration_seconds=12.0,
            params_path=PROJECT_PARAMS,
            generated_xml_dir=generated_xml_dir,
        )
        scene = load_simulation_scene(request)
        recv_port, _ = request.resolved_ports()
        sock = create_udp_socket(recv_port=recv_port)
        try:
            run_headless_loop(scene, sock)
        finally:
            sock.close()

        self.assertGreater(scene.data.time, 11.5)
        position = np.asarray(scene.data.qpos[0:3], dtype=float)
        self.assertTrue(np.all(np.isfinite(position)))
        self.assertGreater(position[0], 2.4)
        self.assertLess(position[0], 2.9)
        self.assertLess(abs(position[1]), 0.75)
        self.assertGreater(position[2], 0.0)
        self.assertLess(position[2], 2.5)

        params = load_vehicle_params(params_path=PROJECT_PARAMS)
        generated_xml_dir.mkdir(parents=True, exist_ok=True)
        model_xml_path, _, _ = render_model_xml(
            params,
            output_path=generated_xml_dir / "wall_spawn_check.generated.xml",
            generated_xml_dir=generated_xml_dir,
        )
        model = mujoco.MjModel.from_xml_path(str(model_xml_path))
        spawn_quat = model.body("drone").quat
        self.assertTrue(np.all(np.isfinite(spawn_quat)))
        self.assertNotAlmostEqual(float(spawn_quat[0]), 1.0, delta=0.05)


if __name__ == "__main__":
    unittest.main()
