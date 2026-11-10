"""Parallel physics (simulation.physics_threads): a world per vehicle, stepped on a thread pool."""

from __future__ import annotations

import json
import socket
import threading
from dataclasses import replace

import mujoco
import numpy as np
import pytest

from wheeled_uav.protocol import create_udp_socket
from wheeled_uav.runtime.contact import build_contact_reports
from wheeled_uav.runtime.loop import run_headless_loop
from wheeled_uav.runtime.parallel import ParallelPhysics
from wheeled_uav.runtime.scene import SimulationRequest, load_simulation_scene, resolve_physics_threads
from wheeled_uav.state_packets import decode_state_packet


def _scene(directory, num_uavs: int = 3, spawn_radius: float = 2.0, **request):
    directory.mkdir(parents=True, exist_ok=True)
    return load_simulation_scene(
        SimulationRequest(
            num_uavs=num_uavs, spawn_radius=spawn_radius, generated_xml_dir=directory, headless=True, **request
        )
    )


def _below_hover_thrusts(scene) -> np.ndarray:
    # Every rotor at a different share of the hover thrust: the vehicles stay
    # on the ground, pressed unevenly, and roll and turn.
    vehicle_mass = scene.model.body_subtreemass[scene.model.body(scene.uav_specs[0].body_name).id]
    return vehicle_mass * 9.81 / 4 * np.linspace(0.2, 0.6, scene.model.nu)


def _drive(scene, physics: ParallelPhysics | None, periods: int = 60, nstep: int = 5) -> None:
    scene.data.ctrl[:] = _below_hover_thrusts(scene)
    for _ in range(periods):
        if physics is None:
            mujoco.mj_step(scene.model, scene.data, nstep=nstep)
        else:
            physics.step(scene.data, nstep)


def _parallel_run(scene, threads: int, **drive) -> None:
    physics = ParallelPhysics(scene, threads)
    try:
        _drive(scene, physics, **drive)
    finally:
        physics.close()


def test_results_do_not_depend_on_the_thread_count(tmp_path):
    one_thread = _scene(tmp_path / "one")
    three_threads = _scene(tmp_path / "three")

    _parallel_run(one_thread, 1)
    _parallel_run(three_threads, 3)

    assert np.array_equal(one_thread.data.qpos, three_threads.data.qpos)
    assert np.array_equal(one_thread.data.qvel, three_threads.data.qvel)
    assert np.array_equal(one_thread.data.sensordata, three_threads.data.sensordata)
    assert one_thread.data.time == three_threads.data.time > 0.0


def test_vehicles_far_apart_move_as_in_the_shared_world(tmp_path):
    shared = _scene(tmp_path / "shared")
    separate = _scene(tmp_path / "separate")
    start = shared.data.qpos.copy()

    _drive(shared, None)
    _parallel_run(separate, 2)

    assert separate.data.time == pytest.approx(shared.data.time)
    assert np.max(np.abs(shared.data.qpos - start)) > 0.01  # the vehicles did move
    # The shared world solves all contacts in one system, the worlds one vehicle
    # each: the same motion up to the solver's tolerance.
    np.testing.assert_allclose(separate.data.qpos, shared.data.qpos, atol=1e-6)
    np.testing.assert_allclose(separate.data.qvel, shared.data.qvel, atol=1e-4)


def test_contact_reports_use_the_shared_models_geom_names(tmp_path):
    shared = _scene(tmp_path / "shared")
    separate = _scene(tmp_path / "separate")
    _drive(shared, None, periods=80)
    physics = ParallelPhysics(separate, 3)
    try:
        _drive(separate, physics, periods=80)
        reports = physics.contact_reports(separate.surface_evaluator, include_details=True)
    finally:
        physics.close()

    expected = build_contact_reports(
        shared.model, shared.data, shared.surface_evaluator, shared.uav_specs, geom_names=shared.geom_names, include_details=True
    )
    # Whether a given vehicle touches the ground at the last instant depends on the
    # MuJoCo version (the unevenly pushed ones roll, bounce and flip), so compare
    # the reports vehicle by vehicle and only require that some vehicle touches.
    assert any(reference["count"] > 0 for reference in expected)
    for report, reference, uav_spec in zip(reports, expected, separate.uav_specs):
        assert report["count"] == reference["count"]
        assert report["left_wheel"]["count"] == reference["left_wheel"]["count"]
        assert report["right_wheel"]["count"] == reference["right_wheel"]["count"]
        geoms = {name for record in report["contacts"] for name in (record["geom1"], record["geom2"])}
        assert geoms == {name for record in reference["contacts"] for name in (record["geom1"], record["geom2"])}
        assert all(name in separate.geom_names for name in geoms)
        if report["count"] > 0:
            assert any(name.startswith(uav_spec.contact_prefix) for name in geoms)


def test_warns_once_when_two_vehicles_come_close(tmp_path, capsys):
    scene = _scene(tmp_path, num_uavs=2, spawn_radius=0.05)
    physics = ParallelPhysics(scene, 2)
    try:
        physics.step(scene.data, 1)
        physics.step(scene.data, 1)
    finally:
        physics.close()

    assert capsys.readouterr().out.count("do not collide with each other") == 1


def test_physics_threads_come_from_the_request_before_the_params(tmp_path):
    scene = _scene(tmp_path, num_uavs=1)
    params = {**scene.params, "simulation": {**scene.params.get("simulation", {}), "physics_threads": 4}}

    assert resolve_physics_threads(scene) == 0
    assert resolve_physics_threads(replace(scene, params=params)) == 4
    assert resolve_physics_threads(replace(scene, params=params, request=replace(scene.request, physics_threads=0))) == 0
    with pytest.raises(ValueError):
        resolve_physics_threads(replace(scene, request=replace(scene.request, physics_threads=-1)))


def test_recording_is_refused_with_parallel_physics(tmp_path):
    scene = _scene(tmp_path, num_uavs=2, physics_threads=2, record_path=str(tmp_path / "run.mp4"))
    with create_udp_socket(recv_port=0) as sock, pytest.raises(ValueError, match="--record"):
        run_headless_loop(scene, sock)


@pytest.mark.parametrize("state_format", ["json", "binary"])
def test_lockstep_run_with_parallel_physics(tmp_path, state_format):
    states = []
    with create_udp_socket(recv_port=0) as simulator_socket, create_udp_socket(recv_port=0) as controller_socket:
        controller_socket.setblocking(True)
        controller_socket.settimeout(10.0)
        scene = _scene(
            tmp_path,
            num_uavs=2,
            send_port=controller_socket.getsockname()[1],
            pacing_mode="lockstep",
            duration_seconds=0.3,
            physics_threads=2,
            state_format=state_format,
        )
        simulator = threading.Thread(target=run_headless_loop, args=(scene, simulator_socket))
        simulator.start()
        command = json.dumps({"rotor_thrusts": [[0.5, 0.5, 0.5, 0.5]] * 2}).encode("utf-8")
        answered = set()
        try:
            while True:
                state = decode_state_packet(controller_socket.recvfrom(65535)[0])
                states.append(state)
                if state["sequence"] not in answered:
                    answered.add(state["sequence"])
                    controller_socket.sendto(command, simulator_socket.getsockname())
                controller_socket.settimeout(2.0)  # the simulator goes quiet once its duration is reached
        except socket.timeout:
            pass
        simulator.join(timeout=10.0)
    assert not simulator.is_alive()

    assert states[-1]["sim_time"] >= 0.3 - 1e-9
    assert all(len(state["uavs"]) == 2 for state in states)
    # On the ground under the light thrust: the worlds' contacts reach the packets. (Not
    # necessarily in the last one: a vehicle tipping over its axle may be in the air then.)
    assert all(any(state["uavs"][index]["contact_summary"]["count"] > 0 for state in states) for index in range(2))
