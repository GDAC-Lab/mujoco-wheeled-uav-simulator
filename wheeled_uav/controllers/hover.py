from __future__ import annotations

"""Reference UDP hover controller for single-UAV simulator packets.

Intended scenarios (see docs/CONTROLLERS.md):
  A/B/C — interactive dev, remote HIL, headless real-time checks with pacing_mode=realtime
  E     — batch / gain sweeps with pacing_mode=accelerated on the simulator side

Timing contract (always, regardless of simulator pacing_mode):
  - Advance control on state.time only; integrator dt = delta(state.time) between unique samples.
  - One control evaluation per new (sequence, time) via StateSampleTracker.
  - Never pause to align controller wall clock with sim_wall_skew or simulator time.
  - Send commands immediately after compute (short duplicate/idle sleeps are poll throttles only).

This module is a geometric-hover reference. For a minimal wall-riding example
(wheels rolling on the wall), see the MATLAB wall_demo_controller entry point.
"""

import json
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..config import build_fidelity_config, load_vehicle_params, parse_total_mass
from ..model.builder import build_allocation_matrix, build_rotor_specs
from ..protocol import RECV_BUFFER_SIZE, UDP_IP, create_udp_socket, get_instance_ports
from ..timing import (
    StateSampleTracker,
    extract_sync_metrics,
    parse_simulation_timing,
    sim_time_seconds,
)

__all__ = [
    "build_hover_command_payload",
    "build_hover_command_payload_with_metadata",
    "build_hover_controller_config",
    "compute_hover_control",
    "ControllerRuntimeStats",
    "StatePacketMetrics",
    "resolve_controller_ports",
    "run_hover_controller",
]

_DEFAULT_DESIRED_HEADING = np.array([1.0, 0.0, 0.0], dtype=float)
_DEFAULT_BODY_Z_AXIS = np.array([0.0, 0.0, 1.0], dtype=float)
_DEFAULT_BODY_Y_AXIS = np.array([0.0, 1.0, 0.0], dtype=float)
_CONTROLLER_DEFAULTS: dict[str, tuple[float, float, float]] = {
    "desired_heading": (1.0, 0.0, 0.0),
    "position_gain": (3.0, 3.0, 6.0),
    "velocity_gain": (2.2, 2.2, 4.0),
    "attitude_gain": (0.8, 0.8, 0.25),
    "angular_velocity_gain": (0.12, 0.12, 0.08),
}
# Large-displacement safety clamps (controller.* scalars, <= 0 disables).
# They keep the desired thrust vector sane when the vehicle is far from its
# target (e.g. dragged in the viewer): without them the raw PD produces a
# near-horizontal or downward force, the collective collapses, and the vehicle
# flips instead of recovering.
_CONTROLLER_SCALAR_DEFAULTS: dict[str, float] = {
    "position_error_limit_m": 1.5,
    "max_tilt_deg": 35.0,
}
# Fraction of hover force kept as the minimum vertical component while the
# tilt clamp is active; bounds commanded descent to (1 - 0.25) g and keeps the
# desired body z-axis pointing up.
_MIN_VERTICAL_FORCE_FACTOR = 0.25


@dataclass(frozen=True)
class HoverControllerConfig:
    mass: float
    gravity: float
    max_rotor_thrust: float
    thrust_coefficient: float
    command_mode: str
    desired_heading: np.ndarray
    position_gain: np.ndarray
    velocity_gain: np.ndarray
    attitude_gain: np.ndarray
    angular_velocity_gain: np.ndarray
    position_error_limit_m: float
    max_tilt_deg: float
    mixer: np.ndarray


@dataclass(frozen=True)
class StatePacketMetrics:
    receive_time_ns: int
    protocol_version: int
    sequence: int | None
    wall_time_send_ns: int | None
    fidelity_mode: str | None
    age_ms: float | None


@dataclass
class ControllerRuntimeStats:
    last_state_sequence: int | None = None
    last_state_age_ms: float | None = None
    last_state_sequence_gap: int = 0
    state_sequence_gap_count: int = 0
    duplicate_state_skip_count: int = 0
    timeout_count: int = 0
    last_controller_compute_ms: float | None = None
    last_command_sequence: int | None = None
    last_source_state_sequence: int | None = None
    last_sim_wall_skew_seconds: float | None = None
    last_realtime_factor: float | None = None


def resolve_controller_ports(instance_id: int, local_port: int | None, target_port: int | None) -> tuple[int, int]:
    default_target_port, default_local_port = get_instance_ports(instance_id)
    resolved_local_port = default_local_port if local_port is None else local_port
    resolved_target_port = default_target_port if target_port is None else target_port
    return resolved_local_port, resolved_target_port


def _normalize_vector(vector: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    vector_norm = np.linalg.norm(vector)
    if vector_norm < 1.0e-6:
        return fallback
    return vector / vector_norm


def _state_vector(state: dict[str, Any], field_name: str) -> np.ndarray:
    return np.asarray(state[field_name], dtype=float).reshape(3)


def _parse_controller_vector(controller_params: dict[str, Any], field_name: str) -> np.ndarray:
    default_vector = _CONTROLLER_DEFAULTS[field_name]
    return np.asarray(controller_params.get(field_name, default_vector), dtype=float).reshape(3)


def _parse_controller_scalar(controller_params: dict[str, Any], field_name: str) -> float:
    return float(controller_params.get(field_name, _CONTROLLER_SCALAR_DEFAULTS[field_name]))


def _clamp_position_error(position_error: np.ndarray, limit: float) -> np.ndarray:
    """Saturate the position-error norm so far-away targets command a bounded pull."""
    if limit <= 0.0:
        return position_error
    error_norm = float(np.linalg.norm(position_error))
    if error_norm <= limit:
        return position_error
    return position_error * (limit / error_norm)


def _apply_force_safety_clamps(desired_force: np.ndarray, min_vertical_force: float, max_tilt_deg: float) -> np.ndarray:
    """Floor the vertical force component and cap the tilt of the desired thrust.

    Guarantees the desired body z-axis stays within max_tilt_deg of vertical and
    never points down, so recovery from a large displacement is a bounded, upright
    maneuver instead of a flip. Disabled when max_tilt_deg <= 0; values >= 90
    keep only the vertical floor (tan would go negative and flip the lateral
    force instead of capping it).
    """
    if max_tilt_deg <= 0.0:
        return desired_force
    clamped_force = np.array(desired_force, dtype=float)
    clamped_force[2] = max(clamped_force[2], min_vertical_force)
    if max_tilt_deg >= 90.0:
        return clamped_force
    max_horizontal_force = clamped_force[2] * np.tan(np.radians(max_tilt_deg))
    horizontal_norm = float(np.linalg.norm(clamped_force[:2]))
    if horizontal_norm > max_horizontal_force:
        clamped_force[:2] *= max_horizontal_force / horizontal_norm
    return clamped_force


def _build_mixer(params: dict[str, Any]) -> np.ndarray:
    # Wrench-to-thrust map derived from the actual rotor geometry (position, tilt,
    # spin direction) so it always matches the simulator actuator ordering, even
    # when the JSON lists rotors in a non-standard order or uses tilted rotors.
    return np.linalg.pinv(build_allocation_matrix(build_rotor_specs(params)))


def _thrust_to_rotor_omega(rotor_thrusts: np.ndarray, thrust_coefficient: float) -> np.ndarray:
    if thrust_coefficient <= 0.0:
        raise ValueError("actuation.thrust_coefficient must be positive when command_mode is omega")
    return np.sqrt(np.maximum(0.0, rotor_thrusts) / thrust_coefficient)


def build_hover_controller_config(params: dict[str, Any]) -> HoverControllerConfig:
    controller_params = params.get("controller", {})
    mass = parse_total_mass(params)
    gravity = abs(float(params["simulation"]["gravity"][2]))
    max_rotor_thrust = float(params["actuation"]["max_rotor_thrust"])
    thrust_coefficient = float(params["actuation"]["thrust_coefficient"])
    command_mode = str(params["actuation"]["command_mode"])

    return HoverControllerConfig(
        mass=mass,
        gravity=gravity,
        max_rotor_thrust=max_rotor_thrust,
        thrust_coefficient=thrust_coefficient,
        command_mode=command_mode,
        desired_heading=_parse_controller_vector(controller_params, "desired_heading"),
        position_gain=_parse_controller_vector(controller_params, "position_gain"),
        velocity_gain=_parse_controller_vector(controller_params, "velocity_gain"),
        attitude_gain=_parse_controller_vector(controller_params, "attitude_gain"),
        angular_velocity_gain=_parse_controller_vector(controller_params, "angular_velocity_gain"),
        position_error_limit_m=_parse_controller_scalar(controller_params, "position_error_limit_m"),
        max_tilt_deg=_parse_controller_scalar(controller_params, "max_tilt_deg"),
        mixer=_build_mixer(params),
    )


def compute_hover_control(state: dict[str, Any], target_position: np.ndarray, config: HoverControllerConfig) -> np.ndarray:
    position = _state_vector(state, "position")
    velocity = _state_vector(state, "velocity")
    angular_velocity = _state_vector(state, "angular_velocity_body")
    rotation_matrix = np.asarray(state["rotation_matrix"], dtype=float).reshape(3, 3)
    desired_heading = _normalize_vector(config.desired_heading, _DEFAULT_DESIRED_HEADING)

    position_error = _clamp_position_error(target_position - position, config.position_error_limit_m)
    velocity_error = -velocity
    desired_force = config.position_gain * position_error + config.velocity_gain * velocity_error + np.array([0.0, 0.0, config.mass * config.gravity], dtype=float)
    desired_force = _apply_force_safety_clamps(
        desired_force,
        _MIN_VERTICAL_FORCE_FACTOR * config.mass * config.gravity,
        config.max_tilt_deg,
    )

    body_z_axis = rotation_matrix[:, 2]
    collective_thrust = max(0.0, float(np.dot(desired_force, body_z_axis)))

    desired_body_z = _normalize_vector(desired_force, _DEFAULT_BODY_Z_AXIS)
    desired_body_y = np.cross(desired_body_z, desired_heading)
    if np.linalg.norm(desired_body_y) < 1.0e-6:
        desired_body_y = np.cross(desired_body_z, _DEFAULT_BODY_Y_AXIS)
    desired_body_y = desired_body_y / np.linalg.norm(desired_body_y)
    desired_body_x = np.cross(desired_body_y, desired_body_z)
    desired_body_x = desired_body_x / np.linalg.norm(desired_body_x)
    desired_rotation = np.column_stack((desired_body_x, desired_body_y, desired_body_z))

    attitude_error_matrix = 0.5 * (desired_rotation.T @ rotation_matrix - rotation_matrix.T @ desired_rotation)
    attitude_error = np.array(
        [
            attitude_error_matrix[2, 1],
            attitude_error_matrix[0, 2],
            attitude_error_matrix[1, 0],
        ],
        dtype=float,
    )
    moment_command = -config.attitude_gain * attitude_error - config.angular_velocity_gain * angular_velocity
    wrench = np.concatenate(([collective_thrust], moment_command))
    rotor_thrusts = config.mixer @ wrench
    return np.clip(rotor_thrusts, 0.0, config.max_rotor_thrust)


def build_hover_command_payload(rotor_thrusts: np.ndarray, config: HoverControllerConfig) -> bytes:
    return build_hover_command_payload_with_metadata(rotor_thrusts, config)


def build_hover_command_payload_with_metadata(
    rotor_thrusts: np.ndarray,
    config: HoverControllerConfig,
    *,
    sequence: int | None = None,
    source_state_sequence: int | None = None,
    wall_time_send_ns: int | None = None,
    fidelity_mode: str | None = None,
    protocol_version: int = 2,
) -> bytes:
    payload: dict[str, Any] = {}
    if sequence is not None:
        payload["protocol_version"] = int(protocol_version)
        payload["sequence"] = int(sequence)
    if source_state_sequence is not None:
        payload["source_state_sequence"] = int(source_state_sequence)
    if wall_time_send_ns is not None:
        payload["wall_time_send_ns"] = int(wall_time_send_ns)
    if fidelity_mode is not None:
        payload["fidelity_mode"] = fidelity_mode

    if config.command_mode == "omega":
        rotor_omega = _thrust_to_rotor_omega(rotor_thrusts, config.thrust_coefficient)
        payload["rotor_omega"] = rotor_omega.tolist()
    else:
        payload["rotor_thrusts"] = rotor_thrusts.tolist()
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _build_state_packet_metrics(state: dict[str, Any], receive_time_ns: int) -> StatePacketMetrics:
    protocol_version = int(state.get("protocol_version", 1))
    sequence = state.get("sequence")
    wall_time_send_ns = state.get("wall_time_send_ns")
    fidelity_mode = state.get("fidelity_mode")
    normalized_sequence = int(sequence) if sequence is not None else None
    normalized_wall_time_send_ns = int(wall_time_send_ns) if wall_time_send_ns is not None else None
    age_ms = None
    if normalized_wall_time_send_ns is not None:
        age_ms = max(0.0, (receive_time_ns - normalized_wall_time_send_ns) / 1.0e6)
    return StatePacketMetrics(
        receive_time_ns=receive_time_ns,
        protocol_version=protocol_version,
        sequence=normalized_sequence,
        wall_time_send_ns=normalized_wall_time_send_ns,
        fidelity_mode=None if fidelity_mode is None else str(fidelity_mode),
        age_ms=age_ms,
    )


def _update_runtime_stats_for_state(stats: ControllerRuntimeStats, metrics: StatePacketMetrics, compute_time_ms: float) -> None:
    previous_sequence = stats.last_state_sequence
    current_sequence = metrics.sequence
    if previous_sequence is not None and current_sequence is not None:
        sequence_gap = max(0, current_sequence - previous_sequence - 1)
        stats.last_state_sequence_gap = sequence_gap
        stats.state_sequence_gap_count += sequence_gap
    else:
        stats.last_state_sequence_gap = 0
    stats.last_state_sequence = current_sequence
    stats.last_state_age_ms = metrics.age_ms
    stats.last_controller_compute_ms = compute_time_ms


def _read_latest_state(sock: socket.socket) -> tuple[dict[str, Any], StatePacketMetrics] | None:
    latest_packet: bytes | None = None
    while True:
        try:
            received_data, _ = sock.recvfrom(RECV_BUFFER_SIZE)
            latest_packet = received_data
        except BlockingIOError:
            break
        except ConnectionResetError:
            break

    if latest_packet is None:
        return None

    receive_time_ns = time.time_ns()
    try:
        state = json.loads(latest_packet.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        # A truncated or foreign datagram must not kill the controller; the
        # next state packet arrives within one publish period anyway.
        return None
    if not isinstance(state, dict):
        return None
    if "uavs" in state:
        raise ValueError("hover-controller expects single-UAV state packets, but received a multi-UAV packet")
    return state, _build_state_packet_metrics(state, receive_time_ns)


def run_hover_controller(
    *,
    instance_id: int = 0,
    bind_ip: str = UDP_IP,
    target_ip: str = UDP_IP,
    local_port: int | None = None,
    target_port: int | None = None,
    params_path: str | Path | None = None,
    target_position: list[float] | tuple[float, float, float] | np.ndarray = (0.0, 0.0, 1.5),
    duration_seconds: float | None = None,
    state_timeout_seconds: float = 10.0,
    status_display_interval: float = 2.0,
    fidelity_mode: str | None = None,
) -> None:
    """Run the reference hover controller until duration or interrupt.

    Pair with simulator ``pacing_mode=realtime`` for HIL (scenarios A/B/C) or
    ``accelerated`` for batch runs (scenario E). This function never aligns to
    wall clock; see docs/CONTROLLERS.md. ``fidelity_mode=None`` tags outgoing
    packets with the params file's ``fidelity_mode``.
    """
    params = load_vehicle_params(params_path=params_path)
    config = build_hover_controller_config(params)
    resolved_fidelity_mode = fidelity_mode or build_fidelity_config(params).mode
    timing_config = parse_simulation_timing(params)
    resolved_target_position = np.asarray(target_position, dtype=float).reshape(3)
    resolved_local_port, resolved_target_port = resolve_controller_ports(instance_id, local_port, target_port)
    sock = create_udp_socket(udp_ip=bind_ip, recv_port=resolved_local_port)
    duplicate_sleep_seconds = min(0.001, 0.25 * timing_config.control_period_seconds)

    print(
        f"Starting Python hover controller (instance={instance_id}, bind={bind_ip}:{resolved_local_port}, "
        f"target={target_ip}:{resolved_target_port}, control_period={timing_config.control_period_seconds:.4f} s)."
    )

    start_simulation_time: float | None = None
    last_state_wall_time = time.monotonic()
    next_status_time = 0.0
    command_sequence = 0
    runtime_stats = ControllerRuntimeStats()
    sample_tracker = StateSampleTracker()

    try:
        while True:
            state_packet = _read_latest_state(sock)
            if state_packet is None:
                if time.monotonic() - last_state_wall_time >= state_timeout_seconds:
                    runtime_stats.timeout_count += 1
                    raise TimeoutError(f"No simulator state received within {state_timeout_seconds:.1f} s")
                time.sleep(0.001)
                continue

            state, state_metrics = state_packet
            if not sample_tracker.is_new(state):
                runtime_stats.duplicate_state_skip_count += 1
                time.sleep(duplicate_sleep_seconds)
                continue

            last_state_wall_time = time.monotonic()
            simulation_time = sim_time_seconds(state)
            if simulation_time is None:
                raise ValueError("Simulator state packet is missing time/sim_time")
            if start_simulation_time is None:
                start_simulation_time = simulation_time
            elapsed_simulation_time = simulation_time - start_simulation_time
            if duration_seconds is not None and elapsed_simulation_time >= duration_seconds:
                print(f"Python hover controller complete at t={elapsed_simulation_time:.2f} s")
                break

            sync_metrics = extract_sync_metrics(state, receive_time_ns=state_metrics.receive_time_ns)
            runtime_stats.last_sim_wall_skew_seconds = sync_metrics["sim_wall_skew_seconds"]
            runtime_stats.last_realtime_factor = sync_metrics["realtime_factor"]

            compute_start = time.perf_counter()
            rotor_thrusts = compute_hover_control(state, resolved_target_position, config)
            compute_time_ms = (time.perf_counter() - compute_start) * 1.0e3
            _update_runtime_stats_for_state(runtime_stats, state_metrics, compute_time_ms)
            command_sequence += 1
            runtime_stats.last_command_sequence = command_sequence
            runtime_stats.last_source_state_sequence = state_metrics.sequence
            sock.sendto(
                build_hover_command_payload_with_metadata(
                    rotor_thrusts,
                    config,
                    sequence=command_sequence,
                    source_state_sequence=state_metrics.sequence,
                    wall_time_send_ns=time.time_ns(),
                    fidelity_mode=resolved_fidelity_mode,
                ),
                (target_ip, resolved_target_port),
            )

            if simulation_time >= next_status_time:
                position = np.asarray(state["position"], dtype=float).reshape(3)
                position_error = resolved_target_position - position
                rtf_val = sync_metrics["realtime_factor"]
                skew_val = sync_metrics["sim_wall_skew_seconds"]
                # Thrusts are always printed in newtons; in omega mode the
                # wire payload carries the converted rotor speeds instead.
                print(
                    "[python-hover t=%.2f s, rtf=%.2f, age=%.2f ms, skew=%+.3f s] "
                    "pos=[%.3f %.3f %.3f] m, err=[%.3f %.3f %.3f] m, "
                    "state_gap=%d, dup_skip=%d, compute=%.3f ms, cmd_mode=%s, thrusts_N=[%.3f %.3f %.3f %.3f]"
                    % (
                        elapsed_simulation_time,
                        rtf_val,
                        sync_metrics["packet_age_ms"],
                        skew_val,
                        position[0],
                        position[1],
                        position[2],
                        position_error[0],
                        position_error[1],
                        position_error[2],
                        runtime_stats.last_state_sequence_gap,
                        runtime_stats.duplicate_state_skip_count,
                        -1.0 if runtime_stats.last_controller_compute_ms is None else runtime_stats.last_controller_compute_ms,
                        config.command_mode,
                        rotor_thrusts[0],
                        rotor_thrusts[1],
                        rotor_thrusts[2],
                        rotor_thrusts[3],
                    )
                )
                next_status_time = simulation_time + status_display_interval
    finally:
        sock.close()