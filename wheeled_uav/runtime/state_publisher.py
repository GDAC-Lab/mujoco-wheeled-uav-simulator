"""Builds and sends simulator state packets (with optional network fidelity)."""

from __future__ import annotations

import json
import random
import socket
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..clock import wall_time_ns
from ..protocol import build_packet_metadata
from ..state_packets import encode_state_packet
from ..types import SensorLayout, UAVModelSpec
from .contact import build_contact_reports
from .fidelity import ActuatorSnapshot, apply_sensor_fidelity
from .scene import SimulationScene, resolve_state_format

__all__ = [
    "StatePayloadPublisher",
    "build_state_payload",
]


# Largest UDP payload over IPv4 (65535 minus the IP and UDP headers).
_MAX_DATAGRAM_BYTES = 65507


@dataclass(frozen=True)
class PendingDatagram:
    release_time_ns: int
    payload: bytes


def _build_uav_state(
    scene: SimulationScene,
    sensor_layout: SensorLayout,
    uav_spec: UAVModelSpec,
    realtime_factor: float,
    sensor_rng: np.random.Generator | None,
    actuator_snapshot: ActuatorSnapshot | None,
    contact_report: dict[str, Any],
    aero_entry: dict[str, object] | None = None,
) -> dict[str, object]:
    data, fidelity = scene.data, scene.fidelity
    true_position = np.array(data.sensordata[sensor_layout.position], copy=True)
    true_linear_velocity = np.array(data.sensordata[sensor_layout.linear_velocity], copy=True)
    true_angular_velocity_world = np.array(data.sensordata[sensor_layout.angular_velocity], copy=True)
    true_rotation_matrix = np.column_stack(
        (
            data.sensordata[sensor_layout.x_axis],
            data.sensordata[sensor_layout.y_axis],
            data.sensordata[sensor_layout.z_axis],
        )
    )
    position, linear_velocity, angular_velocity_world, rotation_matrix = apply_sensor_fidelity(
        true_position,
        true_linear_velocity,
        true_angular_velocity_world,
        true_rotation_matrix,
        fidelity,
        sensor_rng,
    )
    angular_velocity_body = rotation_matrix.T @ angular_velocity_world
    true_angular_velocity_body = true_rotation_matrix.T @ true_angular_velocity_world

    uav_state: dict[str, object] = {
        "name": uav_spec.name,
        "time": data.time,
        "position": position.tolist(),
        "velocity": linear_velocity.tolist(),
        "angular_velocity_world": angular_velocity_world.tolist(),
        "angular_velocity_body": angular_velocity_body.tolist(),
        "rotation_matrix": rotation_matrix.reshape(-1).tolist(),
        "z": float(position[2]),
        "vz": float(linear_velocity[2]),
        "yaw_rate": float(angular_velocity_world[2]),
        "realtime_factor": float(realtime_factor),
        "instance_id": int(scene.request.instance_id),
        "contact_summary": {
            "count": contact_report["count"],
            "total_force_magnitude": contact_report["total_force_magnitude"],
            "max_force_magnitude": contact_report["max_force_magnitude"],
            "total_normal_force": contact_report["total_normal_force"],
            "max_normal_force": contact_report["max_normal_force"],
            "left_wheel": contact_report["left_wheel"],
            "right_wheel": contact_report["right_wheel"],
            "surface": contact_report["surface"],
            "wall": contact_report["wall"],
        },
        "contacts": contact_report["contacts"],
    }
    if actuator_snapshot is not None and fidelity.logging.log_actuator_stats:
        uav_state["actuator"] = {
            "requested_rotor_thrusts": actuator_snapshot.requested_ctrl.tolist(),
            "applied_rotor_thrusts": actuator_snapshot.applied_ctrl.tolist(),
            "tracking_error": (actuator_snapshot.requested_ctrl - actuator_snapshot.applied_ctrl).tolist(),
        }
    if aero_entry is not None:
        uav_state["aero"] = aero_entry
    if fidelity.logging.log_sensor_truth:
        uav_state["sensor_truth"] = {
            "position": true_position.tolist(),
            "velocity": true_linear_velocity.tolist(),
            "angular_velocity_world": true_angular_velocity_world.tolist(),
            "angular_velocity_body": true_angular_velocity_body.tolist(),
            "rotation_matrix": true_rotation_matrix.reshape(-1).tolist(),
        }
    return uav_state


def build_state_payload(
    scene: SimulationScene,
    realtime_factor: float,
    *,
    sequence: int,
    wall_time_send_ns: int,
    sensor_rng: np.random.Generator | None = None,
    actuator_snapshot: ActuatorSnapshot | None = None,
    aero_snapshot: list[dict[str, object]] | None = None,
    timing: dict[str, float] | None = None,
    contact_reports: list[dict[str, Any]] | None = None,
    state_format: str = "json",
) -> bytes:
    """Serialize the scene state into a single-UAV or multi-UAV state packet.

    `contact_reports` (one per UAV) replaces the reports built from the scene's
    own contacts, for runtimes that step the vehicles elsewhere. `state_format`
    "binary" encodes the same content compactly (see wheeled_uav.state_packets),
    with the contact summaries but without the per-contact details.
    """
    if contact_reports is None:
        contact_reports = build_contact_reports(
            scene.model,
            scene.data,
            scene.surface_evaluator,
            scene.uav_specs,
            geom_names=scene.geom_names,
            include_details=scene.fidelity.logging.include_contact_details and state_format == "json",
            attribution=scene.contact_attribution,
        )
    uav_states = [
        _build_uav_state(
            scene,
            sensor_layout,
            uav_spec,
            realtime_factor,
            sensor_rng,
            actuator_snapshot,
            contact_reports[uav_index],
            aero_entry=None if aero_snapshot is None else aero_snapshot[uav_index],
        )
        for uav_index, (sensor_layout, uav_spec) in enumerate(zip(scene.sensor_layouts, scene.uav_specs, strict=True))
    ]
    packet: dict[str, object] = {
        **build_packet_metadata(
            sequence=sequence,
            wall_time_send_ns=wall_time_send_ns,
            fidelity_mode=scene.fidelity.mode,
        ),
        "sim_time": scene.data.time,
    }
    if timing is not None:
        packet["timing"] = timing

    if state_format == "binary":
        return encode_state_packet(packet, uav_states)
    if len(uav_states) == 1:
        return json.dumps(
            {
                **packet,
                **uav_states[0],
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")

    return json.dumps(
        {
            **packet,
            "time": scene.data.time,
            "instance_id": int(scene.request.instance_id),
            "num_uavs": len(uav_states),
            "realtime_factor": float(realtime_factor),
            "uavs": uav_states,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


class StatePayloadPublisher:
    """Publishes state packets, optionally through the HIL network model."""

    def __init__(self, scene: SimulationScene):
        self._scene = scene
        self._sequence = 0
        self._last_payload: bytes | None = None
        self._pending_datagrams: list[PendingDatagram] = []
        self._rng = random.Random(0)
        self._sensor_rng = np.random.default_rng(0)
        self._state_format = resolve_state_format(scene)
        if self._state_format == "binary":
            logging = scene.fidelity.logging
            if logging.log_actuator_stats or logging.log_sensor_truth:
                raise ValueError(
                    'simulation.state_format "binary" carries no actuator stats or sensor truth: turn off '
                    "logging_config.log_actuator_stats and log_sensor_truth, or use the JSON state format"
                )
            # Per-contact details are on unless switched off; binary packets
            # carry the contact summaries only, so the details are not built.
            if scene.params.get("logging_config", {}).get("include_contact_details") is True:
                print("binary state packets carry the contact summaries only; per-contact details are not sent")

    @property
    def include_contact_details(self) -> bool:
        """Whether the packets list every contact (JSON packets with logging_config.include_contact_details)."""
        return self._state_format == "json" and self._scene.fidelity.logging.include_contact_details

    def _network_fidelity_enabled(self) -> bool:
        return self._scene.fidelity.mode == "hil" and self._scene.fidelity.network.enabled

    def _sample_delay_ms(self, base_latency_ms: float) -> float:
        jitter_ms = self._scene.fidelity.network.jitter_std_dev_ms
        sampled_delay_ms = base_latency_ms
        if jitter_ms > 0.0:
            sampled_delay_ms += self._rng.gauss(0.0, jitter_ms)
        return max(0.0, sampled_delay_ms)

    def _should_drop_packet(self) -> bool:
        loss_percent = self._scene.fidelity.network.packet_loss_percent
        return loss_percent > 0.0 and self._rng.random() * 100.0 < loss_percent

    def _flush_due_datagrams(self, sock: socket.socket, target_ip: str, send_port: int, now_ns: int) -> None:
        remaining_datagrams: list[PendingDatagram] = []
        for pending_datagram in self._pending_datagrams:
            if pending_datagram.release_time_ns <= now_ns:
                sock.sendto(pending_datagram.payload, (target_ip, send_port))
            else:
                remaining_datagrams.append(pending_datagram)
        self._pending_datagrams = remaining_datagrams

    def build_payload(
        self,
        realtime_factor: float,
        sequence: int,
        wall_time_send_ns: int,
        actuator_snapshot: ActuatorSnapshot | None,
        *,
        aero_snapshot: list[dict[str, object]] | None = None,
        timing: dict[str, float] | None = None,
        contact_reports: list[dict[str, Any]] | None = None,
    ) -> bytes:
        return build_state_payload(
            self._scene,
            realtime_factor,
            sequence=sequence,
            wall_time_send_ns=wall_time_send_ns,
            sensor_rng=self._sensor_rng,
            actuator_snapshot=actuator_snapshot,
            aero_snapshot=aero_snapshot,
            timing=timing,
            contact_reports=contact_reports,
            state_format=self._state_format,
        )

    def send_state(
        self,
        sock: socket.socket,
        target_ip: str,
        send_port: int,
        realtime_factor: float,
        actuator_snapshot: ActuatorSnapshot | None,
        *,
        aero_snapshot: list[dict[str, object]] | None = None,
        timing: dict[str, float] | None = None,
        contact_reports: list[dict[str, Any]] | None = None,
    ) -> None:
        self._sequence += 1
        now_ns = wall_time_ns()
        payload = self.build_payload(
            realtime_factor,
            self._sequence,
            now_ns,
            actuator_snapshot,
            aero_snapshot=aero_snapshot,
            timing=timing,
            contact_reports=contact_reports,
        )
        self._last_payload = payload
        self._dispatch(sock, target_ip, send_port, payload, now_ns)

    def resend_last(self, sock: socket.socket, target_ip: str, send_port: int) -> None:
        """Send the latest state packet again, byte for byte.

        Lockstep pacing re-sends the current state while it waits for a
        command, so that a controller connecting late still gets one. Keeping
        the sequence number lets controllers drop the copy as a duplicate
        sample instead of answering it with a second command.
        """
        if self._last_payload is not None:
            self._dispatch(sock, target_ip, send_port, self._last_payload, wall_time_ns())

    def _dispatch(self, sock: socket.socket, target_ip: str, send_port: int, payload: bytes, now_ns: int) -> None:
        if len(payload) > _MAX_DATAGRAM_BYTES:
            raise ValueError(
                f"the state packet is {len(payload)} bytes, more than one UDP datagram holds "
                f'({_MAX_DATAGRAM_BYTES}); for large teams set simulation.state_format to "binary"'
            )
        if not self._network_fidelity_enabled():
            sock.sendto(payload, (target_ip, send_port))
            return

        if not self._should_drop_packet():
            delay_ms = self._sample_delay_ms(self._scene.fidelity.network.state_tx_latency_ms)
            release_time_ns = now_ns + int(delay_ms * 1.0e6)
            self._pending_datagrams.append(PendingDatagram(release_time_ns=release_time_ns, payload=payload))
        self._flush_due_datagrams(sock, target_ip, send_port, now_ns)
