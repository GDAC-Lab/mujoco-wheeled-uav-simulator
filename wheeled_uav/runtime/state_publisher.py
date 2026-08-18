"""Builds and sends simulator state packets (with optional network fidelity)."""

from __future__ import annotations

import json
import random
import socket
import time
from dataclasses import dataclass

import numpy as np

from ..protocol import build_packet_metadata
from ..types import SensorLayout, UAVModelSpec
from .contact import build_contact_report
from .fidelity import ActuatorSnapshot, apply_sensor_fidelity
from .scene import SimulationScene

__all__ = [
    "StatePayloadPublisher",
    "build_state_payload",
]


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
    aero_entry: dict[str, object] | None = None,
) -> dict[str, object]:
    model, data, fidelity = scene.model, scene.data, scene.fidelity
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
    contact_report = build_contact_report(
        model,
        data,
        scene.surface_evaluator,
        contact_prefix=uav_spec.contact_prefix,
        geom_names=scene.geom_names,
        include_details=fidelity.logging.include_contact_details,
    )

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
) -> bytes:
    """Serialize the scene state into a single-UAV or multi-UAV JSON packet."""
    uav_states = [
        _build_uav_state(
            scene,
            sensor_layout,
            uav_spec,
            realtime_factor,
            sensor_rng,
            actuator_snapshot,
            aero_entry=None if aero_snapshot is None else aero_snapshot[uav_index],
        )
        for uav_index, (sensor_layout, uav_spec) in enumerate(zip(scene.sensor_layouts, scene.uav_specs, strict=True))
    ]
    packet: dict[str, object] = {
        **build_packet_metadata(
            sequence=sequence,
            wall_time_send_ns=wall_time_send_ns,
            fidelity_mode=scene.request.fidelity_mode,
        ),
        "sim_time": scene.data.time,
    }
    if timing is not None:
        packet["timing"] = timing

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
        self._pending_datagrams: list[PendingDatagram] = []
        self._rng = random.Random(0)
        self._sensor_rng = np.random.default_rng(0)

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
    ) -> None:
        self._sequence += 1
        now_ns = time.time_ns()
        payload = self.build_payload(
            realtime_factor,
            self._sequence,
            now_ns,
            actuator_snapshot,
            aero_snapshot=aero_snapshot,
            timing=timing,
        )
        if not self._network_fidelity_enabled():
            sock.sendto(payload, (target_ip, send_port))
            return

        if not self._should_drop_packet():
            delay_ms = self._sample_delay_ms(self._scene.fidelity.network.state_tx_latency_ms)
            release_time_ns = now_ns + int(delay_ms * 1.0e6)
            self._pending_datagrams.append(PendingDatagram(release_time_ns=release_time_ns, payload=payload))
        self._flush_due_datagrams(sock, target_ip, send_port, now_ns)
