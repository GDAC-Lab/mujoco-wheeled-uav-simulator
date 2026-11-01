"""uavsim.Protocol.decode_state (MATLAB) gives for a binary state packet what jsondecode gives for its JSON twin.

The MATLAB code runs in GNU Octave, or in MATLAB when Octave is not installed;
the test is skipped when neither is on PATH.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np
import pytest

from wheeled_uav.runtime.scene import SimulationRequest, load_simulation_scene, resolve_timing_config
from wheeled_uav.runtime.state_publisher import build_state_payload
from wheeled_uav.timing import SessionTimingTracker

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
OCTAVE = shutil.which("octave-cli")
MATLAB = shutil.which("matlab")

pytestmark = pytest.mark.skipif(OCTAVE is None and MATLAB is None, reason="neither GNU Octave nor MATLAB is installed")


def _write_packets(directory: Path, num_uavs: int) -> None:
    scene = load_simulation_scene(
        SimulationRequest(num_uavs=num_uavs, spawn_radius=1.5, generated_xml_dir=directory, headless=True)
    )
    # Binary packets carry the contact summaries only: the JSON twin without the per-contact details.
    logging = replace(scene.fidelity.logging, include_contact_details=False)
    scene = replace(scene, fidelity=replace(scene.fidelity, logging=logging))
    vehicle_mass = scene.model.body_subtreemass[scene.model.body(scene.uav_specs[0].body_name).id]
    scene.data.ctrl[:] = vehicle_mass * 9.81 / 4 * np.linspace(0.2, 0.6, scene.model.nu)  # rolling on the ground, with contacts
    mujoco.mj_step(scene.model, scene.data, nstep=300)
    timing = SessionTimingTracker.start(0.0, timing=resolve_timing_config(scene)).snapshot(scene.data.time, 0.97)
    timing.update(step_wall_seconds=0.0031, publish_wall_seconds=0.0012, command_wait_wall_seconds=0.0101)
    common = dict(sequence=12345, wall_time_send_ns=1_761_987_654_321_098_765, timing=timing)
    (directory / f"state_{num_uavs}.json").write_bytes(build_state_payload(scene, 0.97, **common))
    (directory / f"state_{num_uavs}.bin").write_bytes(build_state_payload(scene, 0.97, state_format="binary", **common))


def _run_matlab(script: str) -> dict:
    if OCTAVE is not None:
        command = [OCTAVE, "--no-gui", "--quiet", "--eval", script]
    else:
        command = [MATLAB, "-batch", script]
    result = subprocess.run(command, capture_output=True, text=True, timeout=600, check=True)
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_binary_and_json_packets_decode_to_the_same_struct(tmp_path):
    team_sizes = (1, 3)
    for num_uavs in team_sizes:
        _write_packets(tmp_path, num_uavs)
    script = (
        f"addpath('{(REPOSITORY_ROOT / 'matlab').as_posix()}');"
        "result = struct();"
        f"for n = [{' '.join(str(size) for size in team_sizes)}];"
        f" stem = sprintf('{tmp_path.as_posix()}/state_%d', n);"
        " f = fopen([stem '.json'], 'r'); json_bytes = fread(f, Inf, '*uint8')'; fclose(f);"
        " f = fopen([stem '.bin'], 'r'); binary_bytes = fread(f, Inf, '*uint8')'; fclose(f);"
        " expected = jsondecode(native2unicode(json_bytes, 'UTF-8'));"
        " from_binary = uavsim.Protocol.decode_state(binary_bytes);"
        " from_json = uavsim.Protocol.decode_state(json_bytes);"
        " same_fields = isequal(fieldnames(from_binary), fieldnames(expected));"
        " if isfield(expected, 'uavs'), same_fields = same_fields && isequal(fieldnames(from_binary.uavs), fieldnames(expected.uavs)); end;"
        " result.(sprintf('uavs_%d', n)) = struct('binary', isequaln(from_binary, expected),"
        " 'json', isequaln(from_json, expected), 'field_order', same_fields);"
        "end;"
        "fprintf('%s\\n', jsonencode(result));"
    )

    result = _run_matlab(script)

    for num_uavs in team_sizes:
        assert result[f"uavs_{num_uavs}"] == {"binary": True, "json": True, "field_order": True}
