"""PX4 ULog to FlightLog.

Reading a ``.ulg`` needs pyulog (``pip install pyulog``, or the ``logs``
extra); the conversion itself works on a topic table
``{topic: {field: array, "timestamp": microseconds}}``, so it can be fed by
any reader and tested without one.

PX4 works in NED (world) and FRD (body); the simulator in a z-up world and
FLU body. With ``D = diag(1, -1, -1)``: ``p = D p_ned`` (north, west, up),
``R = D R_ned D``, ``w = D w_frd``. Each channel is interpolated from its
topic's own time stamps onto the time base (commands are held), and grid
samples inside a gap of a topic (longer than ``max_gap_factor`` times its
median period) are marked invalid. The motor command becomes a thrust in N
only with a ``thrust_curve``: ``T = sum_ij c[i][j] u^i V^j`` (V from
``battery_status`` or a fixed voltage), the form of a bench calibration
such as a 2-D polynomial in command and voltage.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np

from .flightlog import FlightLog, quaternion_to_rotation, rotation_to_quaternion

__all__ = ["DEFAULT_MAPPING", "convert_topics", "convert_ulog", "read_ulog_topics"]

DEFAULT_MAPPING: dict[str, Any] = {
    "time_base": {"topic": "vehicle_local_position"},  # or {"rate_hz": 200.0}
    "max_gap_factor": 3.0,
    "position": {"topic": "vehicle_local_position", "fields": ["x", "y", "z"], "frame": "ned"},
    "velocity": {"topic": "vehicle_local_position", "fields": ["vx", "vy", "vz"], "frame": "ned"},
    "quaternion": {"topic": "vehicle_attitude", "fields": ["q[0]", "q[1]", "q[2]", "q[3]"], "frame": "ned_frd"},
    "angular_velocity_body": {"topic": "vehicle_angular_velocity", "fields": ["xyz[0]", "xyz[1]", "xyz[2]"], "frame": "frd"},
    "specific_force_body": {"topic": "vehicle_acceleration", "fields": ["xyz[0]", "xyz[1]", "xyz[2]"], "frame": "frd"},
    "rotor_command": {"topic": "actuator_motors", "fields": ["control[0]", "control[1]", "control[2]", "control[3]"]},
    "rotor_speed": {
        "topic": "esc_status",
        "fields": ["esc[0].esc_rpm", "esc[1].esc_rpm", "esc[2].esc_rpm", "esc[3].esc_rpm"],
        "scale": 2.0 * np.pi / 60.0,
    },
    "battery_voltage": {"topic": "battery_status", "fields": ["voltage_v"]},
    "battery_current": {"topic": "battery_status", "fields": ["current_a"]},
    # For each simulator rotor (actuation.rotors order), the log's motor index.
    "motor_order": [0, 1, 2, 3],
    # {"coefficients": [[c00, c01, ...], [c10, ...], ...], "voltage": "battery" | <volts>}: T = sum c[i][j] u^i V^j
    "thrust_curve": None,
}
_D = np.diag([1.0, -1.0, -1.0])
_HELD = ("rotor_command",)


def read_ulog_topics(path: str | Path, topics: list[str]) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    """Read the named topics (instance 0) with pyulog; returns the table and some log information."""
    try:
        from pyulog import ULog
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise SystemExit("reading .ulg files needs pyulog: pip install pyulog") from error
    ulog = ULog(str(path), sorted(set(topics)))
    table: dict[str, dict[str, np.ndarray]] = {}
    for dataset in ulog.data_list:
        if dataset.multi_id != 0 or dataset.name in table:
            continue
        table[dataset.name] = {name: np.asarray(values) for name, values in dataset.data.items()}
    info = {"start_timestamp_us": int(ulog.start_timestamp), "message_formats": len(ulog.message_formats)}
    return table, info


def _merged(mapping: dict[str, Any] | None) -> dict[str, Any]:
    merged = copy.deepcopy(DEFAULT_MAPPING)
    for key, value in (mapping or {}).items():
        if key.startswith("_"):
            continue  # comments and provenance
        if key not in merged:
            raise ValueError(f"unknown mapping key {key!r}; known: {', '.join(merged)}")
        if isinstance(value, dict) and isinstance(merged[key], dict):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value
    return merged


def _columns(table, spec) -> tuple[np.ndarray, np.ndarray] | None:
    topic = table.get(spec["topic"])
    if topic is None or any(field not in topic for field in spec["fields"]):
        return None
    stamps = np.asarray(topic["timestamp"], dtype=np.float64) * 1.0e-6
    values = np.column_stack([np.asarray(topic[field], dtype=float) for field in spec["fields"]]) * float(spec.get("scale", 1.0))
    order = np.argsort(stamps, kind="stable")
    stamps, values = stamps[order], values[order]
    unique = np.concatenate([[True], np.diff(stamps) > 0.0])
    return stamps[unique], values[unique]


def _onto(grid: np.ndarray, stamps: np.ndarray, values: np.ndarray, held: bool, max_gap_factor: float) -> tuple[np.ndarray, np.ndarray]:
    """Values on the grid, and which grid samples are valid (inside the topic's span, outside its gaps)."""
    if held:
        index = np.clip(np.searchsorted(stamps, grid, side="right") - 1, 0, stamps.size - 1)
        result = values[index]
    else:
        result = np.column_stack([np.interp(grid, stamps, values[:, column]) for column in range(values.shape[1])])
    valid = (grid >= stamps[0]) & (grid <= stamps[-1])
    if stamps.size > 2:
        gaps = np.diff(stamps)
        limit = max_gap_factor * float(np.median(gaps))
        for start in np.nonzero(gaps > limit)[0]:
            valid &= ~((grid > stamps[start]) & (grid < stamps[start + 1]))
    return result, valid


def _to_simulator_frame(name: str, frame: str | None, values: np.ndarray) -> np.ndarray:
    if frame in ("ned", "frd"):
        return values @ _D
    if frame == "ned_frd":
        rotation = quaternion_to_rotation(values)
        return rotation_to_quaternion(_D @ rotation @ _D)
    if frame in (None, "simulator"):
        return values
    raise ValueError(f"unknown frame {frame!r} for {name}")


def convert_topics(
    table: dict[str, dict[str, np.ndarray]],
    mapping: dict[str, Any] | None = None,
    *,
    start_s: float | None = None,
    end_s: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> FlightLog:
    """Convert a topic table into a FlightLog (times in seconds from the first time-base sample)."""
    config = _merged(mapping)
    base = config["time_base"]
    if "rate_hz" in base:
        spans = [_columns(table, config[name]) for name in ("position", "quaternion")]
        if any(span is None for span in spans):
            raise ValueError("the position and attitude topics are required")
        first = max(span[0][0] for span in spans)
        last = min(span[0][-1] for span in spans)
        grid = np.arange(first, last, 1.0 / float(base["rate_hz"]))
    else:
        topic = table.get(base["topic"])
        if topic is None:
            raise ValueError(f"time-base topic {base['topic']!r} not in the log")
        grid = np.unique(np.asarray(topic["timestamp"], dtype=np.float64) * 1.0e-6)
    origin = float(grid[0])
    if start_s is not None:
        grid = grid[grid >= origin + start_s]
    if end_s is not None:
        grid = grid[grid <= origin + end_s]
    if grid.size < 2:
        raise ValueError("fewer than two samples in the chosen window")
    channels: dict[str, np.ndarray] = {}
    valid = np.ones(grid.size, dtype=bool)
    present = []
    for name in ("position", "velocity", "quaternion", "angular_velocity_body", "specific_force_body", "rotor_command", "rotor_speed", "battery_voltage", "battery_current"):
        spec = config.get(name)
        if spec is None:
            continue
        columns = _columns(table, spec)
        if columns is None:
            if name in ("position", "quaternion"):
                raise ValueError(f"{name}: topic {spec['topic']!r} or its fields are missing")
            continue
        stamps, values = columns
        values = _to_simulator_frame(name, spec.get("frame"), values)
        on_grid, channel_valid = _onto(grid, stamps, values, name in _HELD, float(config["max_gap_factor"]))
        if name in ("rotor_command", "rotor_speed"):
            on_grid = on_grid[:, list(config["motor_order"])]
        channels[name] = on_grid
        valid &= channel_valid
        present.append(name)
    curve = config.get("thrust_curve")
    if curve is not None and "rotor_command" in channels:
        coefficients = curve["coefficients"]
        voltage_source = curve.get("voltage", "battery")
        if voltage_source == "battery":
            if "battery_voltage" not in channels:
                raise ValueError("thrust_curve uses the battery voltage, but battery_status is not in the log")
            voltage = channels["battery_voltage"][:, 0]
        else:
            voltage = np.full(grid.size, float(voltage_source))
        command = channels["rotor_command"]
        thrust = np.zeros_like(command)
        for i, row in enumerate(coefficients):
            for j, coefficient in enumerate(row):
                thrust += float(coefficient) * command**i * (voltage[:, None] ** j)
        channels["rotor_thrust_command"] = np.maximum(thrust, 0.0)
    channels["valid"] = valid.astype(float)
    info = {
        "source": "px4-ulog",
        "converter": "wheeled_uav.flightlog_ulog",
        "frames": "world NWU (from NED), body FLU (from FRD)",
        "time_origin_s": origin,
        "channels_from_topics": present,
        "mapping": json.loads(json.dumps(config, default=float)),
        **(metadata or {}),
    }
    return FlightLog(grid - origin, channels, info)


def convert_ulog(path: str | Path, mapping: dict[str, Any] | None = None, *, start_s: float | None = None, end_s: float | None = None) -> FlightLog:
    config = _merged(mapping)
    topics = sorted(
        {config["time_base"].get("topic", "")} | {spec["topic"] for name, spec in config.items() if isinstance(spec, dict) and "topic" in spec}
        - {""}
    )
    table, info = read_ulog_topics(path, topics)
    return convert_topics(table, mapping, start_s=start_s, end_s=end_s, metadata={"ulog": str(Path(path).name), **info})
