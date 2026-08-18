"""UDP wire protocol shared by the simulator and every controller.

Owns the port conventions, socket setup, and the JSON command-packet format.
Recognized command fields (single-UAV packets; multi-UAV packets nest the same
per-UAV fields under ``uavs`` or use flat nested lists):

  rotor_thrusts             4 thrusts [N], one per rotor
  rotor_omega/rotor_omegas  4 rotor speeds [rad/s], converted via thrust_coefficient
  wrench                    body wrench [f_z, M_x, M_y, M_z] mixed to rotor
                            thrusts through the geometry-derived allocation matrix
  thrust                    scalar collective [N] split evenly across rotors
  body_wrench(es)           6-element WORLD-frame external wrench applied via
                            xfrc_applied in addition to the rotor thrusts
                            (tilt-rotor emulation) — distinct from ``wrench``

This module is deliberately free of MuJoCo imports so controller processes can
run on machines without a MuJoCo installation.
"""

from __future__ import annotations

import json
import socket
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

from .paths import ROTOR_NAMES

__all__ = [
    "UDP_IP",
    "PORT_SEND",
    "PORT_RECV",
    "RECV_BUFFER_SIZE",
    "CommandPacket",
    "PacketMetrics",
    "build_packet_metadata",
    "get_instance_ports",
    "create_udp_socket",
    "parse_control_input",
    "parse_command_packet",
    "parse_multi_uav_control_input",
    "parse_multi_uav_command_packet",
    "receive_control_command",
    "receive_multi_uav_control_command",
]

UDP_IP = "127.0.0.1"
PORT_SEND = 5001
PORT_RECV = 5000
# Maximum UDP datagram size: state packets with contact details can exceed
# several KiB, and truncated datagrams silently corrupt JSON parsing.
RECV_BUFFER_SIZE = 65535


@dataclass(frozen=True)
class PacketMetrics:
    receive_time_ns: int
    protocol_version: int
    sequence: int | None
    source_state_sequence: int | None
    wall_time_send_ns: int | None
    fidelity_mode: str | None
    age_ms: float | None
    is_stale: bool


@dataclass(frozen=True)
class CommandPacket:
    rotor_thrusts: list[float] | list[list[float]]
    metrics: PacketMetrics
    # Optional per-UAV external body wrench [Fx, Fy, Fz, Mx, My, Mz] in the WORLD
    # frame, applied via data.xfrc_applied in addition to the rotor thrusts.
    # Used to emulate thrust-vectoring (tilt-rotor) drones preliminarily: the
    # rotors keep the body aloft/upright while the wrench supplies the
    # tangential drive and yaw that a tilting rotor set would produce.
    body_wrenches: list[list[float]] | None = None


def build_packet_metadata(
    *,
    sequence: int,
    wall_time_send_ns: int,
    fidelity_mode: str,
    protocol_version: int = 2,
) -> dict[str, int | str]:
    return {
        "protocol_version": int(protocol_version),
        "sequence": int(sequence),
        "wall_time_send_ns": int(wall_time_send_ns),
        "fidelity_mode": fidelity_mode,
    }


def get_instance_ports(instance_id: int) -> tuple[int, int]:
    if instance_id < 0:
        raise ValueError("instance_id must be non-negative")
    port_offset = 2 * instance_id
    return PORT_RECV + port_offset, PORT_SEND + port_offset


def create_udp_socket(udp_ip: str = UDP_IP, recv_port: int = PORT_RECV) -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((udp_ip, recv_port))
    sock.setblocking(False)
    return sock


def _parse_rotor_vector(control_input: dict[str, object], field_names: tuple[str, ...]) -> list[float] | None:
    for field_name in field_names:
        field_value = control_input.get(field_name)
        if isinstance(field_value, list) and len(field_value) == len(ROTOR_NAMES):
            return [float(value) for value in field_value]
    return None


def _get_thrust_coefficient(params: dict[str, Any]) -> float:
    thrust_coefficient = float(params["actuation"]["thrust_coefficient"])
    if thrust_coefficient <= 0.0:
        raise ValueError("actuation.thrust_coefficient must be positive")
    return thrust_coefficient


@lru_cache(maxsize=8)
def _cached_wrench_mixer(rotor_specs_key: tuple) -> "np.ndarray":
    import numpy as np

    from .model.builder import build_allocation_matrix

    return np.linalg.pinv(build_allocation_matrix(list(rotor_specs_key)))


def _wrench_to_rotor_thrusts(wrench_values: list[float], params: dict[str, Any]) -> list[float]:
    # Body wrench [f_z, M_x, M_y, M_z] mapped to rotor thrusts through the
    # pseudo-inverse of the geometry-derived allocation matrix, so controllers can
    # command wrenches without knowing the rotor layout. Imports are deferred so
    # this module stays importable on numpy-free controller hosts until a wrench
    # command actually arrives (model.builder itself is MuJoCo-free).
    import numpy as np

    from .model.builder import build_rotor_specs

    wrench = np.asarray([float(value) for value in wrench_values], dtype=float)
    mixer = _cached_wrench_mixer(tuple(build_rotor_specs(params)))
    max_rotor_thrust = float(params["actuation"]["max_rotor_thrust"])
    rotor_thrusts = np.clip(mixer @ wrench, 0.0, max_rotor_thrust)
    return [float(thrust) for thrust in rotor_thrusts]


def _parse_single_uav_control_input(control_input: dict[str, object], params: dict[str, Any]) -> list[float] | None:
    rotor_thrusts = control_input.get("rotor_thrusts")
    if isinstance(rotor_thrusts, list) and len(rotor_thrusts) == len(ROTOR_NAMES):
        return [float(thrust) for thrust in rotor_thrusts]

    rotor_omega = _parse_rotor_vector(control_input, ("rotor_omega", "rotor_omegas"))
    if rotor_omega is not None:
        return _rotor_omega_to_thrust(rotor_omega, _get_thrust_coefficient(params))

    wrench_values = control_input.get("wrench")
    if isinstance(wrench_values, list) and len(wrench_values) == 4:
        return _wrench_to_rotor_thrusts(wrench_values, params)

    thrust = control_input.get("thrust")
    if thrust is None:
        return None

    scalar_thrust = float(thrust)
    return [scalar_thrust] * len(ROTOR_NAMES)


def _rotor_omega_to_thrust(rotor_omega: list[float], thrust_coefficient: float) -> list[float]:
    return [max(0.0, thrust_coefficient * omega * omega) for omega in rotor_omega]


def parse_control_input(control_input: dict[str, object], params: dict[str, Any]) -> list[float] | None:
    return _parse_single_uav_control_input(control_input, params)


def _build_packet_metrics(
    control_input: dict[str, object],
    *,
    receive_time_ns: int,
    stale_command_threshold_ms: float | None,
) -> PacketMetrics:
    protocol_version = int(control_input.get("protocol_version", 1))
    sequence = control_input.get("sequence")
    source_state_sequence = control_input.get("source_state_sequence")
    wall_time_send_ns = control_input.get("wall_time_send_ns")
    fidelity_mode = control_input.get("fidelity_mode")

    normalized_sequence = int(sequence) if sequence is not None else None
    normalized_source_state_sequence = int(source_state_sequence) if source_state_sequence is not None else None
    normalized_wall_time_send_ns = int(wall_time_send_ns) if wall_time_send_ns is not None else None
    normalized_fidelity_mode = None if fidelity_mode is None else str(fidelity_mode)
    age_ms = None
    if normalized_wall_time_send_ns is not None:
        age_ms = max(0.0, (receive_time_ns - normalized_wall_time_send_ns) / 1.0e6)
    is_stale = age_ms is not None and stale_command_threshold_ms is not None and age_ms > stale_command_threshold_ms
    return PacketMetrics(
        receive_time_ns=receive_time_ns,
        protocol_version=protocol_version,
        sequence=normalized_sequence,
        source_state_sequence=normalized_source_state_sequence,
        wall_time_send_ns=normalized_wall_time_send_ns,
        fidelity_mode=normalized_fidelity_mode,
        age_ms=age_ms,
        is_stale=is_stale,
    )


def _parse_single_body_wrench(control_input: dict[str, object]) -> list[list[float]] | None:
    # Optional world-frame body wrench [Fx,Fy,Fz,Mx,My,Mz] on a single-UAV
    # packet (tilt-rotor emulation), normalized to the per-UAV batch shape.
    body_wrench = control_input.get("body_wrench")
    if not isinstance(body_wrench, list) or len(body_wrench) != 6:
        return None
    return [[float(component) for component in body_wrench]]


def parse_command_packet(
    control_input: dict[str, object],
    params: dict[str, Any],
    *,
    receive_time_ns: int | None = None,
    stale_command_threshold_ms: float | None = None,
) -> CommandPacket | None:
    rotor_thrusts = parse_control_input(control_input, params)
    if rotor_thrusts is None:
        return None
    resolved_receive_time_ns = time.time_ns() if receive_time_ns is None else int(receive_time_ns)
    return CommandPacket(
        rotor_thrusts=rotor_thrusts,
        body_wrenches=_parse_single_body_wrench(control_input),
        metrics=_build_packet_metrics(
            control_input,
            receive_time_ns=resolved_receive_time_ns,
            stale_command_threshold_ms=stale_command_threshold_ms,
        ),
    )


def _receive_latest_packet(sock: socket.socket) -> bytes | None:
    latest_packet: bytes | None = None
    while True:
        try:
            received_data, _ = sock.recvfrom(RECV_BUFFER_SIZE)
            latest_packet = received_data
        except BlockingIOError:
            break
        except ConnectionResetError:
            break
    return latest_packet


def _decode_packet(received_data: bytes) -> dict[str, object] | None:
    # A truncated or foreign datagram must not kill the receive loop; the next
    # packet arrives within one control period anyway.
    try:
        control_input = json.loads(received_data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(control_input, dict):
        return None
    return control_input


_warned_unrecognized_key_sets: set[tuple[str, ...]] = set()


def _warn_unrecognized_command(control_input: dict[str, object]) -> None:
    # An unparseable command is otherwise indistinguishable from "no controller
    # yet": with hold_until_first_command the vehicle then sits frozen forever
    # with no diagnostic. Warn once per distinct key set.
    key_set = tuple(sorted(control_input.keys()))
    if key_set in _warned_unrecognized_key_sets:
        return
    _warned_unrecognized_key_sets.add(key_set)
    print(f"warning: ignoring command packet with unrecognized fields {list(key_set)}")


def receive_control_command(
    sock: socket.socket,
    params: dict[str, Any],
    *,
    stale_command_threshold_ms: float | None = None,
) -> CommandPacket | None:
    received_data = _receive_latest_packet(sock)
    if received_data is None:
        return None

    control_input = _decode_packet(received_data)
    if control_input is None:
        return None
    command_packet = parse_command_packet(
        control_input,
        params,
        receive_time_ns=time.time_ns(),
        stale_command_threshold_ms=stale_command_threshold_ms,
    )
    if command_packet is None:
        _warn_unrecognized_command(control_input)
    return command_packet


def parse_multi_uav_control_input(control_input: dict[str, object], params: dict[str, Any], num_uavs: int) -> list[list[float]] | None:
    rotor_thrusts = control_input.get("rotor_thrusts")
    if isinstance(rotor_thrusts, list) and len(rotor_thrusts) == num_uavs:
        parsed_commands: list[list[float]] = []
        for thrust_vector in rotor_thrusts:
            if not isinstance(thrust_vector, list) or len(thrust_vector) != len(ROTOR_NAMES):
                return None
            parsed_commands.append([float(thrust) for thrust in thrust_vector])
        return parsed_commands

    # Accept both spellings, mirroring the single-UAV parser, so a controller
    # that works at num_uavs=1 does not silently go quiet at num_uavs=2.
    for omega_field in ("rotor_omegas", "rotor_omega"):
        rotor_omegas = control_input.get(omega_field)
        if isinstance(rotor_omegas, list) and len(rotor_omegas) == num_uavs:
            thrust_coefficient = _get_thrust_coefficient(params)
            parsed_commands = []
            for rotor_omega in rotor_omegas:
                if not isinstance(rotor_omega, list) or len(rotor_omega) != len(ROTOR_NAMES):
                    return None
                parsed_commands.append(_rotor_omega_to_thrust([float(value) for value in rotor_omega], thrust_coefficient))
            return parsed_commands

    uavs = control_input.get("uavs")
    if isinstance(uavs, list) and len(uavs) == num_uavs:
        parsed_commands = []
        for uav_control_input in uavs:
            if not isinstance(uav_control_input, dict):
                return None
            rotor_command = _parse_single_uav_control_input(uav_control_input, params)
            if rotor_command is None:
                return None
            parsed_commands.append(rotor_command)
        return parsed_commands

    return None


def parse_multi_uav_command_packet(
    control_input: dict[str, object],
    params: dict[str, Any],
    num_uavs: int,
    *,
    receive_time_ns: int | None = None,
    stale_command_threshold_ms: float | None = None,
) -> CommandPacket | None:
    rotor_thrusts = parse_multi_uav_control_input(control_input, params, num_uavs)
    if rotor_thrusts is None:
        return None
    body_wrenches = _parse_body_wrenches(control_input, num_uavs)
    resolved_receive_time_ns = time.time_ns() if receive_time_ns is None else int(receive_time_ns)
    return CommandPacket(
        rotor_thrusts=rotor_thrusts,
        body_wrenches=body_wrenches,
        metrics=_build_packet_metrics(
            control_input,
            receive_time_ns=resolved_receive_time_ns,
            stale_command_threshold_ms=stale_command_threshold_ms,
        ),
    )


def _parse_body_wrenches(control_input: dict[str, object], num_uavs: int) -> list[list[float]] | None:
    # Optional per-UAV world-frame body wrench [Fx,Fy,Fz,Mx,My,Mz] (tilt-rotor
    # emulation). Absent -> None (no external wrench, unchanged behavior).
    body_wrenches = control_input.get("body_wrenches")
    if not isinstance(body_wrenches, list) or len(body_wrenches) != num_uavs:
        return _parse_uav_entry_body_wrenches(control_input, num_uavs)
    parsed: list[list[float]] = []
    for wrench in body_wrenches:
        if not isinstance(wrench, list) or len(wrench) != 6:
            return None
        parsed.append([float(component) for component in wrench])
    return parsed


def _parse_uav_entry_body_wrenches(control_input: dict[str, object], num_uavs: int) -> list[list[float]] | None:
    # Fallback for the `uavs` encoding: each entry may carry its own
    # `body_wrench` (the natural per-UAV shape a controller builds before
    # flattening). Entries without one get a zero wrench.
    uavs = control_input.get("uavs")
    if not isinstance(uavs, list) or len(uavs) != num_uavs:
        return None
    parsed: list[list[float]] = []
    any_wrench = False
    for uav_control_input in uavs:
        body_wrench = uav_control_input.get("body_wrench") if isinstance(uav_control_input, dict) else None
        if isinstance(body_wrench, list) and len(body_wrench) == 6:
            parsed.append([float(component) for component in body_wrench])
            any_wrench = True
        else:
            parsed.append([0.0] * 6)
    return parsed if any_wrench else None


def receive_multi_uav_control_command(
    sock: socket.socket,
    params: dict[str, Any],
    num_uavs: int,
    *,
    stale_command_threshold_ms: float | None = None,
) -> CommandPacket | None:
    received_data = _receive_latest_packet(sock)
    if received_data is None:
        return None

    control_input = _decode_packet(received_data)
    if control_input is None:
        return None
    command_packet = parse_multi_uav_command_packet(
        control_input,
        params,
        num_uavs,
        receive_time_ns=time.time_ns(),
        stale_command_threshold_ms=stale_command_threshold_ms,
    )
    if command_packet is None:
        _warn_unrecognized_command(control_input)
    return command_packet


