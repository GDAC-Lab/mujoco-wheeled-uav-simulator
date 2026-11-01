"""Binary state packets (simulation.state_format = "binary") against the JSON packets."""

from __future__ import annotations

import copy
import json
from dataclasses import replace

import mujoco
import numpy as np
import pytest

from wheeled_uav.config import load_vehicle_params
from wheeled_uav.runtime.scene import (
    SimulationRequest,
    load_simulation_scene,
    resolve_state_format,
    resolve_timing_config,
)
from wheeled_uav.runtime.state_publisher import StatePayloadPublisher, build_state_payload
from wheeled_uav.state_packets import STATE_PACKET_MAGIC, decode_state_packet, encode_state_packet
from wheeled_uav.timing import SessionTimingTracker


def _rolling_scene(tmp_path, num_uavs: int):
    scene = load_simulation_scene(
        SimulationRequest(num_uavs=num_uavs, spawn_radius=1.5, generated_xml_dir=tmp_path, headless=True)
    )
    # Binary packets carry the contact summaries only: compare with JSON packets without the per-contact details.
    scene = replace(scene, fidelity=replace(scene.fidelity, logging=replace(scene.fidelity.logging, include_contact_details=False)))
    # Below the hover thrust, unevenly: the vehicles roll on the ground and have contacts.
    vehicle_mass = scene.model.body_subtreemass[scene.model.body(scene.uav_specs[0].body_name).id]
    scene.data.ctrl[:] = vehicle_mass * 9.81 / 4 * np.linspace(0.2, 0.6, scene.model.nu)
    mujoco.mj_step(scene.model, scene.data, nstep=300)
    return scene


def _payloads(scene) -> tuple[bytes, bytes]:
    timing = SessionTimingTracker.start(0.0, timing=resolve_timing_config(scene)).snapshot(scene.data.time, 0.97)
    timing.update(step_wall_seconds=0.0031, publish_wall_seconds=0.0012, command_wait_wall_seconds=0.0101)
    common = dict(sequence=12345, wall_time_send_ns=1_761_987_654_321_098_765, timing=timing)
    return (
        build_state_payload(scene, 0.97, **common),
        build_state_payload(scene, 0.97, state_format="binary", **common),
    )


def _uav_state(name: str, offset: float) -> dict:
    group = {"count": 1.0, "total_force_magnitude": 2.5 + offset, "max_force_magnitude": 2.5, "total_normal_force": 2.4, "max_normal_force": 2.4}
    return {
        "name": name,
        "time": 0.25,
        "position": [offset, -offset, 0.1],
        "velocity": [0.1, 0.0, -0.01],
        "angular_velocity_world": [0.0, 0.2, 0.3],
        "angular_velocity_body": [0.0, 0.3, 0.2],
        "rotation_matrix": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        "z": 0.1,
        "vz": -0.01,
        "yaw_rate": 0.3,
        "realtime_factor": 1.0,
        "instance_id": 4,
        "contact_summary": {
            "count": 2,
            "total_force_magnitude": 5.0,
            "max_force_magnitude": 2.5,
            "total_normal_force": 4.8,
            "max_normal_force": 2.4,
            "left_wheel": dict(group),
            "right_wheel": dict(group),
            "surface": {**group, "count": 2.0},
            "wall": {**group, "count": 0.0},
        },
        "contacts": [],
    }


_HEADER = {"protocol_version": 2, "sequence": 7, "wall_time_send_ns": 1_700_000_000_000_000_000, "fidelity_mode": None, "sim_time": 0.25}


@pytest.mark.parametrize("num_uavs", [1, 3])
def test_binary_packet_decodes_to_the_json_packet(tmp_path, num_uavs):
    json_payload, binary_payload = _payloads(_rolling_scene(tmp_path, num_uavs))

    assert binary_payload.startswith(STATE_PACKET_MAGIC)
    assert len(binary_payload) < len(json_payload) / 2
    expected = json.loads(json_payload)
    decoded = decode_state_packet(binary_payload)
    # Same keys in the same order, same values, ints and floats where JSON has them.
    assert json.dumps(decoded) == json.dumps(expected)
    assert decode_state_packet(json_payload) == expected


def test_packets_without_timing_or_fidelity_mode_round_trip():
    uavs = [_uav_state("uav_0", 0.0), _uav_state("uav_1", 1.5)]
    decoded = decode_state_packet(encode_state_packet(dict(_HEADER), uavs))

    assert decoded == {**_HEADER, "time": 0.25, "instance_id": 4, "num_uavs": 2, "realtime_factor": 1.0, "uavs": uavs}
    single = decode_state_packet(encode_state_packet(dict(_HEADER), uavs[:1]))
    assert single == {**_HEADER, **uavs[0]}


def test_a_team_of_a_hundred_fits_one_datagram():
    # 46 doubles per UAV plus its name: JSON packets of real states pass the
    # 65507-byte datagram limit at about 40 UAVs.
    uavs = [_uav_state(f"uav_{index}", 0.1 * index) for index in range(100)]

    assert len(encode_state_packet(dict(_HEADER), uavs)) < 65507


@pytest.mark.parametrize(
    "change",
    [
        lambda uav, packet: uav.update(actuator={"requested_rotor_thrusts": [0.0] * 4}),
        lambda uav, packet: uav.update(contacts=[{"geom1": "a", "geom2": "b"}]),
        lambda uav, packet: packet.update(timing={"unknown_seconds": 0.1}),
    ],
    ids=["actuator_log", "contact_details", "unknown_timing_field"],
)
def test_content_without_a_binary_field_is_refused(change):
    uav, packet = _uav_state("uav_0", 0.0), dict(_HEADER)
    change(uav, packet)
    with pytest.raises(ValueError):
        encode_state_packet(packet, [uav])


def test_state_format_comes_from_the_request_before_the_params(tmp_path):
    scene = load_simulation_scene(SimulationRequest(generated_xml_dir=tmp_path, headless=True))
    params = {**scene.params, "simulation": {**scene.params.get("simulation", {}), "state_format": "Binary"}}

    assert resolve_state_format(scene) == "json"
    assert resolve_state_format(replace(scene, params=params)) == "binary"
    assert resolve_state_format(replace(scene, params=params, request=replace(scene.request, state_format="json"))) == "json"
    with pytest.raises(ValueError):
        resolve_state_format(replace(scene, request=replace(scene.request, state_format="msgpack")))


def _binary_scene(tmp_path, **logging_config):
    params = copy.deepcopy(load_vehicle_params())
    params["simulation"]["state_format"] = "binary"
    params.setdefault("logging_config", {}).update(logging_config)
    params_path = tmp_path / "vehicle_params.json"
    params_path.write_text(json.dumps(params), encoding="utf-8")
    return load_simulation_scene(SimulationRequest(params_path=str(params_path), generated_xml_dir=tmp_path, headless=True))


def test_binary_publisher_refuses_the_logs_it_cannot_carry(tmp_path):
    with pytest.raises(ValueError, match="log_actuator_stats"):
        StatePayloadPublisher(_binary_scene(tmp_path, log_actuator_stats=True))


def test_binary_packets_send_contact_summaries_without_the_details(tmp_path, capsys):
    scene = _binary_scene(tmp_path, include_contact_details=True)
    scene.data.ctrl[:] = 0.0
    mujoco.mj_step(scene.model, scene.data, nstep=300)  # settle on the floor

    publisher = StatePayloadPublisher(scene)
    state = decode_state_packet(publisher.build_payload(1.0, 1, 0, None))

    assert "per-contact details are not sent" in capsys.readouterr().out
    assert not publisher.include_contact_details
    assert state["contacts"] == []
    assert state["contact_summary"]["count"] > 0


def test_a_json_packet_too_large_for_one_datagram_names_the_binary_format(tmp_path):
    publisher = StatePayloadPublisher(load_simulation_scene(SimulationRequest(generated_xml_dir=tmp_path, headless=True)))

    with pytest.raises(ValueError, match='state_format to "binary"'):
        publisher._dispatch(None, "127.0.0.1", 0, b"x" * 65508, 0)
