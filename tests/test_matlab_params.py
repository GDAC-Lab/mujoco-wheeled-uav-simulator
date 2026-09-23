"""uavsim.Params.load (MATLAB) must see the calibrated values load_vehicle_params sees.

In command_mode "omega" the MATLAB controller converts thrust to rotor speed with
its kf and the simulator converts back with its own, so the two loaders must
apply actuation.calibration_file identically. The MATLAB code runs in GNU Octave;
the tests are skipped when octave-cli is not on PATH.
"""

from __future__ import annotations

import copy
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from wheeled_uav.config import clear_vehicle_params_cache, load_vehicle_params

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
OCTAVE = shutil.which("octave-cli")

pytestmark = pytest.mark.skipif(OCTAVE is None, reason="GNU Octave (octave-cli) is not installed")

_CALIBRATION = {
    "schema": "uav-propulsion-calibration/1",
    "source": {"mat_file": "thrust_20260101_test.mat"},
    "sim_params": {"thrust_coefficient": 3.5e-05, "yaw_moment_ratio": 0.025, "motor_tau_ms": 40.0},
}

# Octave's jsondecode can land one ulp away from the correctly rounded double.
_REL = 1e-12


def _write_case(directory: Path, sim_params: dict | None, drop_rotor_ratios: int = 0) -> Path:
    params = json.loads((REPOSITORY_ROOT / "vehicle_params.json").read_text(encoding="utf-8"))
    if sim_params is not None:
        params["actuation"]["calibration_file"] = "calibration/bench.json"
        calibration_path = directory / "calibration" / "bench.json"
        calibration_path.parent.mkdir()
        calibration_path.write_text(json.dumps({**_CALIBRATION, "sim_params": sim_params}), encoding="utf-8")
    # Rotors without their own yaw_moment_ratio make jsondecode return a cell array.
    for rotor in params["actuation"]["rotors"][:drop_rotor_ratios]:
        del rotor["yaw_moment_ratio"]
    params_path = directory / "vehicle_params.json"
    params_path.write_text(json.dumps(params), encoding="utf-8")
    return params_path


def _load_in_octave(params_path: Path) -> dict:
    script = (
        f"addpath('{REPOSITORY_ROOT / 'matlab'}');"
        "try;"
        f" p = uavsim.Params.load('{params_path.parent}', 'params_path', '{params_path}');"
        " out = struct('thrust_coefficient', p.thrust_coefficient, 'yaw_moment_ratio', p.yaw_moment_ratio,"
        " 'rotor_yaw_moment_ratio', [p.rotors.yaw_moment_ratio], 'raw', p.raw);"
        "catch err;"
        " out = struct('error', err.identifier);"
        "end;"
        "printf('%s\\n', jsonencode(out));"
    )
    result = subprocess.run(
        [OCTAVE, "--no-gui", "--quiet", "--eval", script], capture_output=True, text=True, timeout=120, check=True
    )
    return json.loads(result.stdout.strip().splitlines()[-1])


def _load_in_python(params_path: Path) -> dict:
    clear_vehicle_params_cache()
    return copy.deepcopy(load_vehicle_params(params_path))


@pytest.mark.parametrize("drop_rotor_ratios", [0, 2], ids=["rotor_struct_array", "rotor_cell_array"])
def test_matlab_applies_the_calibration_file_like_python(tmp_path, drop_rotor_ratios):
    params_path = _write_case(tmp_path, _CALIBRATION["sim_params"], drop_rotor_ratios)

    python = _load_in_python(params_path)
    matlab = _load_in_octave(params_path)

    actuation = python["actuation"]
    assert matlab["thrust_coefficient"] == pytest.approx(actuation["thrust_coefficient"], rel=_REL)
    assert matlab["raw"]["actuation"]["thrust_coefficient"] == pytest.approx(3.5e-05, rel=_REL)
    assert matlab["yaw_moment_ratio"] == pytest.approx(actuation["yaw_moment_ratio"], rel=_REL)
    expected_rotor_ratios = [
        rotor.get("yaw_moment_ratio", actuation["yaw_moment_ratio"]) for rotor in actuation["rotors"]
    ]
    assert matlab["rotor_yaw_moment_ratio"] == pytest.approx(expected_rotor_ratios, rel=_REL)
    assert expected_rotor_ratios == pytest.approx([0.025] * 4)
    assert matlab["raw"]["actuator_dynamics"]["motor_tau_ms"] == python["actuator_dynamics"]["motor_tau_ms"]

    applied = matlab["raw"]["calibration_applied"]
    assert Path(applied["file"]).resolve() == Path(python["calibration_applied"]["file"]).resolve()
    assert applied["source"] == python["calibration_applied"]["source"]
    assert applied["applied"] == pytest.approx(python["calibration_applied"]["applied"], rel=_REL)


def test_matlab_skips_null_calibration_entries(tmp_path):
    params_path = _write_case(tmp_path, {"thrust_coefficient": None, "yaw_moment_ratio": 0.03, "motor_tau_ms": None})

    python = _load_in_python(params_path)
    matlab = _load_in_octave(params_path)

    assert matlab["thrust_coefficient"] == pytest.approx(python["actuation"]["thrust_coefficient"], rel=_REL)
    assert matlab["yaw_moment_ratio"] == pytest.approx(0.03, rel=_REL)
    assert set(matlab["raw"]["calibration_applied"]["applied"]) == {"yaw_moment_ratio"}


def test_matlab_leaves_params_without_calibration_file_untouched(tmp_path):
    params_path = _write_case(tmp_path, None)

    python = _load_in_python(params_path)
    matlab = _load_in_octave(params_path)

    assert "calibration_applied" not in matlab["raw"]
    assert matlab["thrust_coefficient"] == pytest.approx(python["actuation"]["thrust_coefficient"], rel=_REL)


def test_matlab_reports_a_missing_calibration_file(tmp_path):
    params_path = _write_case(tmp_path, None)
    params = json.loads(params_path.read_text(encoding="utf-8"))
    params["actuation"]["calibration_file"] = "missing.json"
    params_path.write_text(json.dumps(params), encoding="utf-8")

    assert _load_in_octave(params_path) == {"error": "uavsim:Params:calibrationNotFound"}
