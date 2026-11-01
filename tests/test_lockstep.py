"""End-to-end lockstep co-simulation over loopback UDP."""

from __future__ import annotations

import copy
import json
import socket
import threading
import time
from unittest.mock import patch

from wheeled_uav.config import load_vehicle_params
from wheeled_uav.protocol import create_udp_socket
from wheeled_uav.runtime.loop import run_headless_loop
from wheeled_uav.runtime.scene import SimulationRequest, load_simulation_scene
from wheeled_uav.runtime.state_publisher import StatePayloadPublisher

_COMMAND = json.dumps({"rotor_thrusts": [0.0, 0.0, 0.0, 0.0]}).encode("utf-8")


def _run_lockstep(tmp_path, *, params_path=None, silent_seconds: float = 0.0, duration_seconds: float = 0.05):
    """Run a lockstep simulator in a thread and answer every new state with a command.

    With ``silent_seconds`` the controller first waits that long without
    answering. Returns the raw datagrams in order of arrival, duplicates kept.
    """
    datagrams: list[bytes] = []
    with create_udp_socket(recv_port=0) as simulator_socket, create_udp_socket(recv_port=0) as controller_socket:
        controller_socket.setblocking(True)
        scene = load_simulation_scene(
            SimulationRequest(
                send_port=controller_socket.getsockname()[1],
                params_path=params_path,
                generated_xml_dir=tmp_path,
                pacing_mode="lockstep",
                duration_seconds=duration_seconds,
                headless=True,
            )
        )
        simulator = threading.Thread(target=run_headless_loop, args=(scene, simulator_socket))
        simulator.start()
        simulator_address = simulator_socket.getsockname()

        controller_socket.settimeout(10.0)
        datagrams.append(controller_socket.recvfrom(65535)[0])
        silent_until = time.monotonic() + silent_seconds
        while (remaining := silent_until - time.monotonic()) > 0.0:
            controller_socket.settimeout(remaining)
            try:
                datagrams.append(controller_socket.recvfrom(65535)[0])
            except socket.timeout:
                break
        # The simulator goes quiet once its duration is reached.
        controller_socket.settimeout(2.0)
        answered = set()
        while True:
            state = json.loads(datagrams[-1])
            if state["sequence"] not in answered:
                answered.add(state["sequence"])
                controller_socket.sendto(_COMMAND, simulator_address)
            try:
                datagrams.append(controller_socket.recvfrom(65535)[0])
            except socket.timeout:
                break
        simulator.join(timeout=10.0)
    assert not simulator.is_alive()
    return datagrams


def test_lockstep_resends_the_unchanged_state_while_it_waits(tmp_path):
    datagrams = _run_lockstep(tmp_path, silent_seconds=1.5)

    # Silent past the once-a-second re-send: the copy is byte for byte the first state.
    assert datagrams[1] == datagrams[0]
    states = [json.loads(datagram) for datagram in datagrams[1:]]
    sequences = [state["sequence"] for state in states]
    assert sequences == sorted(set(sequences))  # no further duplicates once commands flow
    assert states[-1]["sim_time"] >= 0.05 - 1e-9


def test_lockstep_reports_its_own_timing(tmp_path):
    states = [json.loads(datagram) for datagram in _run_lockstep(tmp_path, silent_seconds=0.2)]

    assert states[0]["timing"]["command_wait_wall_seconds"] == 0.0  # the first state follows no command
    assert states[0]["timing"]["step_wall_seconds"] == 0.0
    for state in states[1:]:
        timing = state["timing"]
        assert timing["step_wall_seconds"] > 0.0
        assert timing["publish_wall_seconds"] > 0.0
        assert timing["command_wait_wall_seconds"] >= 0.0
    assert states[1]["timing"]["command_wait_wall_seconds"] >= 0.2  # the controller's deliberate pause


def test_lockstep_asks_again_when_a_command_is_dropped(tmp_path):
    params = copy.deepcopy(load_vehicle_params())
    params["fidelity_mode"] = "hil"
    params["network_fidelity"].update({"enabled": True, "packet_loss_percent": 30.0})
    params_path = tmp_path / "vehicle_params.json"
    params_path.write_text(json.dumps(params), encoding="utf-8")

    # Lose commands only: the state path stays lossless so the run ends on a delivered state.
    with patch.object(StatePayloadPublisher, "_should_drop_packet", return_value=False):
        datagrams = _run_lockstep(tmp_path, params_path=str(params_path))

    states = [json.loads(datagram) for datagram in datagrams]
    assert states[-1]["sim_time"] >= 0.05 - 1e-9
    # Every dropped command was answered by a re-published state with a new
    # sequence, so the run needed more states than its 11 control periods.
    assert len({state["sequence"] for state in states}) > 11
