"""Binary state packets: an opt-in, compact alternative to the JSON format.

A JSON state packet grows by about 1.5 KiB per UAV and passes the 64 KiB limit
of a UDP datagram at around 40 UAVs. With ``simulation.state_format = "binary"``
the simulator sends the same content in a fixed little-endian layout instead
(about a quarter of the size, see docs/STATE_PACKETS.md):

    header    magic b"WUAV", layout version, protocol version, UAV count,
              record length, fidelity and pacing codes, timing-field flags,
              instance id, sequence, send time [ns], simulation time, realtime
              factor, then the ten timing values (NaN where absent)
    names     UTF-8 UAV names joined by newlines (uint16 length first), padded
              to a multiple of 8 bytes
    records   one row of float64 per UAV: position, velocity, angular velocity
              (world, then body), rotation matrix (row-major), and the contact
              summary (totals, then the left_wheel, right_wheel, surface and
              wall groups, five values each)

``decode_state_packet`` turns a datagram of either format into the dict that
``json.loads`` gives for the JSON packet of the same state, so a controller can
accept both. Binary packets carry no per-contact details and no actuator,
sensor-truth or aerodynamics logs.

This module is free of MuJoCo and NumPy imports, like :mod:`wheeled_uav.protocol`.
"""

from __future__ import annotations

import json
import math
import struct
from typing import Any

__all__ = [
    "STATE_PACKET_MAGIC",
    "decode_state_packet",
    "encode_state_packet",
]

STATE_PACKET_MAGIC = b"WUAV"
_LAYOUT_VERSION = 1
_HEADER = struct.Struct("<4sHHHHBBHiqqdd")
_TIMING_FIELDS = (
    "physics_timestep_seconds",
    "control_period_seconds",
    "publish_every_n_steps",
    "session_wall_elapsed_seconds",
    "session_sim_elapsed_seconds",
    "sim_wall_skew_seconds",
    "realtime_factor",
    "step_wall_seconds",
    "publish_wall_seconds",
    "command_wait_wall_seconds",
)
_TIMING = struct.Struct(f"<{len(_TIMING_FIELDS)}d")
# Flag bits: one per timing field, then pacing_mode, and "the packet has a timing block".
_FLAG_PACING_MODE = 1 << len(_TIMING_FIELDS)
_FLAG_TIMING = 1 << 15
_FIDELITY_MODES = ("baseline", "hil")
_NO_FIDELITY_MODE = 255
_PACING_MODES = ("realtime", "accelerated", "lockstep")
_VECTORS = (
    ("position", 3),
    ("velocity", 3),
    ("angular_velocity_world", 3),
    ("angular_velocity_body", 3),
    ("rotation_matrix", 9),
)
_CONTACT_FIELDS = ("count", "total_force_magnitude", "max_force_magnitude", "total_normal_force", "max_normal_force")
_CONTACT_GROUPS = ("left_wheel", "right_wheel", "surface", "wall")
RECORD_LENGTH = sum(size for _, size in _VECTORS) + len(_CONTACT_FIELDS) * (1 + len(_CONTACT_GROUPS))
_UAV_KEYS = frozenset(
    (
        "name", "time", "position", "velocity", "angular_velocity_world", "angular_velocity_body",
        "rotation_matrix", "z", "vz", "yaw_rate", "realtime_factor", "instance_id", "contact_summary", "contacts",
    )
)


def encode_state_packet(packet: dict[str, Any], uav_states: list[dict[str, Any]]) -> bytes:
    """Binary form of the state whose JSON form is ``packet`` plus ``uav_states``.

    ``packet`` holds the packet-level fields (metadata, ``sim_time`` and the
    optional ``timing`` block) and ``uav_states`` one dict per UAV, as the
    state publisher builds them.
    """
    for uav_state in uav_states:
        extra = set(uav_state) - _UAV_KEYS
        if extra or uav_state["contacts"]:
            raise ValueError(
                "binary state packets carry no per-contact details and no actuator, sensor-truth or "
                "aerodynamics logs; turn those off or use the JSON state format "
                f"(found {sorted(extra) if extra else ['contacts']})"
            )
    flags = 0
    timing_values = [math.nan] * len(_TIMING_FIELDS)
    pacing_code = 0
    timing = packet.get("timing")
    if timing is not None:
        unknown = set(timing) - set(_TIMING_FIELDS) - {"pacing_mode"}
        if unknown:
            raise ValueError(f"binary state packets cannot carry the timing fields {sorted(unknown)}")
        flags |= _FLAG_TIMING
        for index, name in enumerate(_TIMING_FIELDS):
            if name in timing:
                flags |= 1 << index
                timing_values[index] = float(timing[name])
        if "pacing_mode" in timing:
            flags |= _FLAG_PACING_MODE
            pacing_code = _PACING_MODES.index(timing["pacing_mode"])
    first = uav_states[0]
    header = _HEADER.pack(
        STATE_PACKET_MAGIC,
        _LAYOUT_VERSION,
        int(packet["protocol_version"]),
        len(uav_states),
        RECORD_LENGTH,
        _NO_FIDELITY_MODE if packet["fidelity_mode"] is None else _FIDELITY_MODES.index(packet["fidelity_mode"]),
        pacing_code,
        flags,
        int(first["instance_id"]),
        int(packet["sequence"]),
        int(packet["wall_time_send_ns"]),
        float(packet["sim_time"]),
        float(first["realtime_factor"]),
    )
    names = "\n".join(str(uav_state["name"]) for uav_state in uav_states).encode("utf-8")
    preamble = header + _TIMING.pack(*timing_values) + struct.pack("<H", len(names)) + names
    preamble += bytes(-len(preamble) % 8)
    record = struct.Struct(f"<{RECORD_LENGTH}d")
    rows = []
    for uav_state in uav_states:
        summary = uav_state["contact_summary"]
        values: list[float] = []
        for name, _ in _VECTORS:
            values.extend(uav_state[name])
        values.extend(summary[field] for field in _CONTACT_FIELDS)
        for group in _CONTACT_GROUPS:
            values.extend(summary[group][field] for field in _CONTACT_FIELDS)
        rows.append(record.pack(*values))
    return preamble + b"".join(rows)


def _decode_binary(datagram: bytes) -> dict[str, Any]:
    (
        _,
        layout_version,
        protocol_version,
        num_uavs,
        record_length,
        fidelity_code,
        pacing_code,
        flags,
        instance_id,
        sequence,
        wall_time_send_ns,
        sim_time,
        realtime_factor,
    ) = _HEADER.unpack_from(datagram, 0)
    if layout_version != _LAYOUT_VERSION or record_length < RECORD_LENGTH:
        raise ValueError(f"unsupported binary state packet (layout {layout_version}, record length {record_length})")
    offset = _HEADER.size
    timing_values = _TIMING.unpack_from(datagram, offset)
    offset += _TIMING.size
    (names_length,) = struct.unpack_from("<H", datagram, offset)
    offset += 2
    names = datagram[offset : offset + names_length].decode("utf-8").split("\n")
    offset += names_length
    offset += -offset % 8

    packet: dict[str, Any] = {
        "protocol_version": protocol_version,
        "sequence": sequence,
        "wall_time_send_ns": wall_time_send_ns,
        "fidelity_mode": None if fidelity_code == _NO_FIDELITY_MODE else _FIDELITY_MODES[fidelity_code],
        "sim_time": sim_time,
    }
    if flags & _FLAG_TIMING:
        timing: dict[str, Any] = {}
        for index, name in enumerate(_TIMING_FIELDS):
            if name == "session_wall_elapsed_seconds" and flags & _FLAG_PACING_MODE:
                timing["pacing_mode"] = _PACING_MODES[pacing_code]
            if flags & (1 << index):
                timing[name] = timing_values[index]
        packet["timing"] = timing

    record = struct.Struct(f"<{record_length}d")
    uav_states = []
    for uav_index in range(num_uavs):
        values = record.unpack_from(datagram, offset + uav_index * record.size)
        uav_state: dict[str, Any] = {"name": names[uav_index], "time": sim_time}
        position = 0
        for name, size in _VECTORS:
            uav_state[name] = list(values[position : position + size])
            position += size
        totals = values[position : position + len(_CONTACT_FIELDS)]
        position += len(_CONTACT_FIELDS)
        summary: dict[str, Any] = dict(zip(_CONTACT_FIELDS, totals))
        summary["count"] = int(summary["count"])
        for group in _CONTACT_GROUPS:
            summary[group] = dict(zip(_CONTACT_FIELDS, values[position : position + len(_CONTACT_FIELDS)]))
            position += len(_CONTACT_FIELDS)
        uav_state.update(
            {
                "z": uav_state["position"][2],
                "vz": uav_state["velocity"][2],
                "yaw_rate": uav_state["angular_velocity_world"][2],
                "realtime_factor": realtime_factor,
                "instance_id": instance_id,
                "contact_summary": summary,
                "contacts": [],
            }
        )
        uav_states.append(uav_state)

    if num_uavs == 1:
        return {**packet, **uav_states[0]}
    return {
        **packet,
        "time": sim_time,
        "instance_id": instance_id,
        "num_uavs": num_uavs,
        "realtime_factor": realtime_factor,
        "uavs": uav_states,
    }


def decode_state_packet(datagram: bytes) -> dict[str, Any]:
    """Decode a state datagram of either format into the JSON packet's dict."""
    if datagram.startswith(STATE_PACKET_MAGIC):
        return _decode_binary(datagram)
    return json.loads(datagram.decode("utf-8"))
