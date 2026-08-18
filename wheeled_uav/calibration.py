"""Apply measured propulsion calibration data to vehicle params.

A calibration file is produced by a bench thrust-test fitting step (any test
rig works as long as it writes the schema below) and carries the identified
propulsion parameters together with their provenance (which test session,
motor/prop/ESC, fit quality). Referencing it from
``actuation.calibration_file`` keeps vehicle_params.json in sync with the most
recent bench test without hand-copying numbers, and records where each number
came from so the source session stays traceable from the parameter file.

Schema (``uav-propulsion-calibration/1``)::

    {
      "schema": "uav-propulsion-calibration/1",
      "source": { ... free-form provenance: mat file, meta, fit quality ... },
      "models": { ... raw fitted models for reference ... },
      "sim_params": {
        "thrust_coefficient": 2.1e-05 | null,   # kf [N s^2/rad^2]
        "yaw_moment_ratio":   0.021   | null,   # km/kf [m]
        "motor_tau_ms":       null              # first-order motor lag [ms]
      }
    }

Entries in ``sim_params`` that are ``null`` (or absent) are skipped, so a
calibration overrides only what it actually measured.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

__all__ = ["apply_calibration_file"]

_SCHEMA_NAME = "uav-propulsion-calibration"
_SUPPORTED_SCHEMA_VERSIONS = {1}

# Announce each calibration file once per process, not on every params load.
_announced_paths: set[str] = set()


def apply_calibration_file(params: dict[str, Any], params_path: Path) -> dict[str, Any]:
    """Overlay calibration values referenced by ``actuation.calibration_file``.

    Relative paths resolve against the directory containing vehicle_params.json.
    Returns ``params`` (mutated in place) for call-site convenience; when no
    calibration is referenced the dict is returned untouched.
    """
    actuation = params.get("actuation")
    if not isinstance(actuation, dict):
        return params
    raw_path = actuation.get("calibration_file")
    if not raw_path:
        return params

    calibration_path = Path(str(raw_path))
    if not calibration_path.is_absolute():
        calibration_path = params_path.parent / calibration_path
    if not calibration_path.is_file():
        raise FileNotFoundError(f"actuation.calibration_file not found: {calibration_path}")

    with calibration_path.open("r", encoding="utf-8") as calibration_file:
        document = json.load(calibration_file)

    schema = str(document.get("schema", ""))
    schema_name, _, version_text = schema.partition("/")
    if schema_name != _SCHEMA_NAME or not version_text.isdigit() or int(version_text) not in _SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(
            f"Unsupported calibration schema '{schema}' in {calibration_path}; expected '{_SCHEMA_NAME}/1'"
        )

    sim_params = document.get("sim_params")
    if not isinstance(sim_params, dict):
        raise ValueError(f"Calibration file {calibration_path} has no 'sim_params' object")

    applied: dict[str, float] = {}

    thrust_coefficient = sim_params.get("thrust_coefficient")
    if thrust_coefficient is not None:
        value = float(thrust_coefficient)
        if value <= 0.0:
            raise ValueError("calibration sim_params.thrust_coefficient must be positive")
        actuation["thrust_coefficient"] = value
        applied["thrust_coefficient"] = value

    yaw_moment_ratio = sim_params.get("yaw_moment_ratio")
    if yaw_moment_ratio is not None:
        value = float(yaw_moment_ratio)
        if value <= 0.0:
            raise ValueError("calibration sim_params.yaw_moment_ratio must be positive")
        actuation["yaw_moment_ratio"] = value
        # The bench test measures one km/kf for the whole propulsion set, and the
        # model builder prefers per-rotor overrides, so align those too — a stale
        # per-rotor value would silently win otherwise.
        rotors = actuation.get("rotors")
        if isinstance(rotors, list):
            for rotor in rotors:
                if isinstance(rotor, dict) and "yaw_moment_ratio" in rotor:
                    rotor["yaw_moment_ratio"] = value
        applied["yaw_moment_ratio"] = value

    motor_tau_ms = sim_params.get("motor_tau_ms")
    if motor_tau_ms is not None:
        value = float(motor_tau_ms)
        if value < 0.0:
            raise ValueError("calibration sim_params.motor_tau_ms must be >= 0")
        actuator_section = params.setdefault("actuator_dynamics", {})
        if isinstance(actuator_section, dict):
            actuator_section["motor_tau_ms"] = value
            applied["motor_tau_ms"] = value

    source = document.get("source")
    params["calibration_applied"] = {
        "file": str(calibration_path),
        "source": source if isinstance(source, dict) else {},
        "applied": dict(applied),
    }

    announce_key = str(calibration_path)
    if announce_key not in _announced_paths:
        _announced_paths.add(announce_key)
        if applied:
            summary = ", ".join(f"{name}={value:g}" for name, value in applied.items())
            print(f"calibration: {calibration_path.name} -> {summary}")
        else:
            print(f"calibration: {calibration_path.name} has no applicable sim_params; nothing overridden")

    return params
