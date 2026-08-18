"""Receives controller command packets and applies them to the MuJoCo data.

Handles the HIL network model on the command path (receive latency, jitter,
packet loss) and the stale-command policies. Single-UAV and multi-UAV packets
share one code path: single-UAV commands are treated as a one-element batch.
"""

from __future__ import annotations

import random
import socket
import time
from dataclasses import dataclass
from typing import Any

import mujoco

from ..config import parse_total_mass
from ..protocol import (
    CommandPacket,
    receive_control_command,
    receive_multi_uav_control_command,
)
from ..types import FidelityConfig, UAVModelSpec

__all__ = [
    "CommandRuntimeStats",
    "ControlCommandDispatcher",
    "apply_body_wrenches",
    "apply_thrust_command",
]


@dataclass
class CommandRuntimeStats:
    last_receive_time_ns: int | None = None
    last_packet_age_ms: float | None = None
    last_packet_sequence: int | None = None
    last_sequence_gap: int = 0
    missed_command_updates: int = 0
    stale_command_count: int = 0
    stale_command_apply_count: int = 0
    command_timeout_count: int = 0
    last_applied_policy: str = "fresh"


@dataclass(frozen=True)
class PendingCommandDelivery:
    release_time_ns: int
    packet: CommandPacket


def apply_thrust_command(data: mujoco.MjData, rotor_thrusts_by_uav: list[list[float]]) -> None:
    data.ctrl[:] = [thrust for rotor_thrusts in rotor_thrusts_by_uav for thrust in rotor_thrusts]


def apply_body_wrenches(data: mujoco.MjData, body_ids: list[int], body_wrenches: list[list[float]] | None) -> None:
    # Rebuild the command-owned world-frame external wrench on each drone
    # body's CoM. Only the UAV body rows are touched, so perturbations other
    # writers place on different bodies (e.g. viewer drag on a wheel) survive;
    # the aerodynamics model adds its force on top of these rows afterwards.
    if body_wrenches is None:
        for body_id in body_ids:
            data.xfrc_applied[body_id, :] = 0.0
        return
    for body_id, wrench in zip(body_ids, body_wrenches, strict=True):
        data.xfrc_applied[body_id, :] = wrench


class ControlCommandDispatcher:
    def __init__(self, params: dict[str, Any], fidelity: FidelityConfig, num_uavs: int):
        self._params = params
        self._fidelity = fidelity
        self._num_uavs = num_uavs
        self._stats = CommandRuntimeStats()
        self._last_command: list[list[float]] | None = None
        self._last_body_wrenches: list[list[float]] | None = None
        # True while the UAV xfrc rows still hold values we wrote; lets
        # refresh_body_wrenches zero them exactly once after wrench commands
        # stop, instead of clobbering the rows forever.
        self._wrench_rows_dirty = False
        self._pending_commands: list[PendingCommandDelivery] = []
        self._rng = random.Random(0)
        self.has_received_command = False
        # Counter of fresh (non-stale) commands applied; lockstep mode blocks on it.
        self.applied_command_count = 0
        # Body ids for optional external body-wrench application (tilt-rotor
        # emulation); populated by bind_bodies once the model is built.
        self._uav_body_ids: list[int] | None = None

    def bind_bodies(self, model: mujoco.MjModel, uav_specs: list[UAVModelSpec]) -> None:
        self._uav_body_ids = []
        for spec in uav_specs:
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, spec.body_name)
            if body_id < 0:
                # -1 would silently write wrenches onto the LAST body instead.
                raise ValueError(f"UAV body {spec.body_name!r} not found in the compiled model")
            self._uav_body_ids.append(body_id)

    @property
    def stats(self) -> CommandRuntimeStats:
        return self._stats

    def _network_fidelity_enabled(self) -> bool:
        return self._fidelity.mode == "hil" and self._fidelity.network.enabled

    def _sample_delay_ms(self, base_latency_ms: float) -> float:
        jitter_ms = self._fidelity.network.jitter_std_dev_ms
        sampled_delay_ms = base_latency_ms
        if jitter_ms > 0.0:
            sampled_delay_ms += self._rng.gauss(0.0, jitter_ms)
        return max(0.0, sampled_delay_ms)

    def _should_drop_packet(self) -> bool:
        loss_percent = self._fidelity.network.packet_loss_percent
        return loss_percent > 0.0 and self._rng.random() * 100.0 < loss_percent

    def _observed_command_age_ms(self, command_packet: CommandPacket, now_ns: int) -> float | None:
        wall_time_send_ns = command_packet.metrics.wall_time_send_ns
        if wall_time_send_ns is None:
            return command_packet.metrics.age_ms
        return max(0.0, (now_ns - wall_time_send_ns) / 1.0e6)

    def _is_stale(self, command_packet: CommandPacket, now_ns: int) -> bool:
        threshold_ms = self._fidelity.network.stale_command_threshold_ms
        observed_age_ms = self._observed_command_age_ms(command_packet, now_ns)
        return threshold_ms is not None and observed_age_ms is not None and observed_age_ms > threshold_ms

    def _queue_incoming_command(self, command_packet: CommandPacket, now_ns: int) -> None:
        if self._should_drop_packet():
            return
        delay_ms = self._sample_delay_ms(self._fidelity.network.command_rx_latency_ms)
        release_time_ns = now_ns + int(delay_ms * 1.0e6)
        self._pending_commands.append(PendingCommandDelivery(release_time_ns=release_time_ns, packet=command_packet))

    def _pop_latest_due_command(self, now_ns: int) -> CommandPacket | None:
        due_packets: list[CommandPacket] = []
        remaining_packets: list[PendingCommandDelivery] = []
        for pending_command in self._pending_commands:
            if pending_command.release_time_ns <= now_ns:
                due_packets.append(pending_command.packet)
            else:
                remaining_packets.append(pending_command)
        self._pending_commands = remaining_packets
        if not due_packets:
            return None
        return due_packets[-1]

    def _update_metrics(self, command_packet: CommandPacket, now_ns: int) -> bool:
        metrics = command_packet.metrics
        self._stats.last_receive_time_ns = metrics.receive_time_ns
        self._stats.last_packet_age_ms = self._observed_command_age_ms(command_packet, now_ns)
        previous_sequence = self._stats.last_packet_sequence
        current_sequence = metrics.sequence
        if previous_sequence is not None and current_sequence is not None:
            sequence_gap = max(0, current_sequence - previous_sequence - 1)
            self._stats.last_sequence_gap = sequence_gap
            self._stats.missed_command_updates += sequence_gap
        else:
            self._stats.last_sequence_gap = 0
        self._stats.last_packet_sequence = current_sequence
        is_stale = self._is_stale(command_packet, now_ns)
        if is_stale:
            self._stats.stale_command_count += 1
        return is_stale

    def _apply_stale_policy(self, data: mujoco.MjData) -> None:
        policy = self._fidelity.network.stale_command_policy
        self._stats.stale_command_apply_count += 1
        self._stats.last_applied_policy = policy
        if policy == "hold-last-command":
            if self._last_command is not None:
                apply_thrust_command(data, self._last_command)
            return

        if policy == "zero-thrust":
            data.ctrl[:] = [0.0] * data.ctrl.shape[0]
            self._last_body_wrenches = None
            return

        if policy == "hover-fallback":
            hover_value = parse_total_mass(self._params) * abs(float(self._params["simulation"]["gravity"][2])) / 4.0
            data.ctrl[:] = [hover_value] * data.ctrl.shape[0]
            self._last_body_wrenches = None

    def _handle_missing_command(self, data: mujoco.MjData, now_ns: int) -> None:
        threshold_ms = self._fidelity.network.stale_command_threshold_ms
        if threshold_ms is None or self._stats.last_receive_time_ns is None:
            return
        age_since_receive_ms = (now_ns - self._stats.last_receive_time_ns) / 1.0e6
        if age_since_receive_ms <= threshold_ms:
            return
        self._stats.command_timeout_count += 1
        self._apply_stale_policy(data)

    def _receive_latest_command(self, sock: socket.socket) -> CommandPacket | None:
        stale_threshold_ms = None if self._network_fidelity_enabled() else self._fidelity.network.stale_command_threshold_ms
        if self._num_uavs == 1:
            return receive_control_command(sock, self._params, stale_command_threshold_ms=stale_threshold_ms)
        return receive_multi_uav_control_command(
            sock,
            self._params,
            self._num_uavs,
            stale_command_threshold_ms=stale_threshold_ms,
        )

    def _normalize_rotor_thrusts(self, command_packet: CommandPacket) -> list[list[float]]:
        rotor_thrusts = command_packet.rotor_thrusts
        if self._num_uavs == 1:
            if not isinstance(rotor_thrusts, list) or (rotor_thrusts and isinstance(rotor_thrusts[0], list)):
                raise ValueError("Single-UAV command packet must contain a flat rotor thrust list")
            return [[float(thrust) for thrust in rotor_thrusts]]
        if not isinstance(rotor_thrusts, list) or not rotor_thrusts or not isinstance(rotor_thrusts[0], list):
            raise ValueError("Multi-UAV command packet must contain a nested rotor thrust list")
        return [[float(thrust) for thrust in uav_command] for uav_command in rotor_thrusts]

    def apply_next_command(self, sock: socket.socket, data: mujoco.MjData) -> None:
        now_ns = time.time_ns()
        incoming_command = self._receive_latest_command(sock)
        if incoming_command is not None:
            self.has_received_command = True
        if incoming_command is not None and self._network_fidelity_enabled():
            self._queue_incoming_command(incoming_command, now_ns)
            control_command = self._pop_latest_due_command(now_ns)
        elif incoming_command is not None:
            control_command = incoming_command
        else:
            control_command = self._pop_latest_due_command(now_ns) if self._network_fidelity_enabled() else None

        if control_command is None:
            self._handle_missing_command(data, now_ns)
            return

        is_stale = self._update_metrics(control_command, now_ns)
        if is_stale:
            # _last_command must keep the last FRESH command here: overwriting
            # it first would make hold-last-command replay the stale packet.
            self._apply_stale_policy(data)
            return

        self._last_command = self._normalize_rotor_thrusts(control_command)
        apply_thrust_command(data, self._last_command)
        # Wrenches are recorded here and written into MjData once per physics
        # step by refresh_body_wrenches, so the aerodynamics contribution can
        # be composed on top without either writer clobbering the other.
        self._last_body_wrenches = control_command.body_wrenches
        self._stats.last_applied_policy = "fresh"
        self.applied_command_count += 1

    def refresh_body_wrenches(self, data: mujoco.MjData, *, force_rebuild: bool = False) -> None:
        """Re-assert the command-owned external wrench rows, once per physics step.

        MjData.xfrc_applied persists across steps and the aerodynamics model adds
        its force on top of these rows every step, so while wrench commands are
        active the base wrench must be rewritten from scratch each step to keep
        the composition well-defined.

        While no wrench command is active (the normal case), the rows are left
        completely untouched so external writers — most importantly viewer drag
        perturbations on the vehicle — behave natively. The one exception is a
        single zeroing pass right after wrench commands stop, and callers that
        need a per-step rebase anyway (active aerodynamics) pass force_rebuild.
        """
        if self._uav_body_ids is None:
            return
        if not force_rebuild and self._last_body_wrenches is None and not self._wrench_rows_dirty:
            return
        apply_body_wrenches(data, self._uav_body_ids, self._last_body_wrenches)
        self._wrench_rows_dirty = self._last_body_wrenches is not None

    def wait_for_command(self, sock: socket.socket, data: mujoco.MjData, republish=None, timeout_seconds: float = 120.0) -> bool:
        """Block until a fresh control command has been applied (lockstep mode).

        While waiting, optionally re-invokes `republish` every second so a
        late-connecting controller still receives the current state.
        Returns False on timeout.
        """
        target_count = self.applied_command_count + 1
        start_time = time.monotonic()
        last_republish = start_time
        while time.monotonic() - start_time < timeout_seconds:
            self.apply_next_command(sock, data)
            if self.applied_command_count >= target_count:
                return True
            now = time.monotonic()
            if republish is not None and now - last_republish >= 1.0:
                republish()
                last_republish = now
            time.sleep(0.0002)
        return False
