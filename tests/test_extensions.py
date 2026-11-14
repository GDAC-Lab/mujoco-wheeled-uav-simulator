"""Scene extensions (wheeled_uav.extensions, docs/EXTENSIONS.md): loading, the calls, commands and states."""

from __future__ import annotations

import json
import shutil
import socket
import subprocess
import threading
from pathlib import Path

import mujoco
import numpy as np
import pytest

from wheeled_uav.config import load_vehicle_params
from wheeled_uav.extensions import Extension, ExtensionError, load_extensions, to_jsonable
from wheeled_uav.extensions.marker import Marker
from wheeled_uav.protocol import create_udp_socket, parse_command_packet, parse_multi_uav_command_packet
from wheeled_uav.runtime.command_dispatcher import ControlCommandDispatcher
from wheeled_uav.runtime.loop import SimulationRuntime, run_headless_loop
from wheeled_uav.runtime.scene import SimulationRequest, load_simulation_scene, validate_model
from wheeled_uav.runtime.state_publisher import build_state_payload
from wheeled_uav.sim import Simulator

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MARKER = "wheeled_uav.extensions.marker:Marker"
AIRBORNE = {"drone": {"initial_position": [0.0, 0.0, 1.0]}}
THRUSTS = [1.9, 2.1, 2.0, 2.05]


class Recorder(Extension):
    """Records the calls the simulator makes and what it sees at each."""

    def __init__(self, name, config):
        super().__init__(name, config)
        self.calls: list[str] = []
        self.commands: list = []
        self.context = None
        self.ctrl_seen = None

    def edit_spec(self, spec, context):
        self.calls.append("edit_spec")
        self.context = context

    def bind(self, model):
        self.calls.append("bind")

    def reset(self, model, data):
        self.calls.append("reset")

    def apply_command(self, command):
        self.calls.append("apply_command")
        self.commands.append(command)

    def before_step(self, model, data):
        self.calls.append("before_step")
        self.ctrl_seen = np.array(data.ctrl)

    def state(self, model, data):
        self.calls.append("state")
        return {"time": data.time, "array": np.arange(3), "count": np.int64(2), "pair": (np.float64(0.5), 1)}


def _write_params(directory: Path, extensions: dict, **sections) -> Path:
    params = load_vehicle_params()
    params.update(sections)
    params["extensions"] = extensions
    path = directory / f"vehicle_params.{len(list(directory.glob('vehicle_params.*.json')))}.json"
    path.write_text(json.dumps(params), encoding="utf-8")
    return path


def _vehicle_trajectory(overrides: dict, control_periods: int = 60) -> np.ndarray:
    with Simulator(overrides=overrides) as sim:
        sim.reset()
        rows = []
        for _ in range(control_periods):
            sim.step(THRUSTS)
            rows.append(np.array(sim.scene.data.qpos[:7]))
        return np.array(rows)


def test_without_extensions_nothing_changes(tmp_path):
    scene = load_simulation_scene(SimulationRequest(generated_xml_dir=tmp_path, headless=True))
    assert scene.extensions is None
    assert not scene.model_xml_path.name.endswith(".extended.xml")
    plain = scene.model_xml_path.read_bytes()
    for section in ({}, {"goal": {"class": MARKER, "enabled": False}}):
        params = load_vehicle_params()
        params["extensions"] = section
        other = load_simulation_scene(SimulationRequest(generated_xml_dir=tmp_path / "other", headless=True), params=params)
        assert other.extensions is None
        assert other.model_xml_path.read_bytes() == plain


def test_extensions_leave_the_vehicle_physics_bit_identical():
    # The scene compiles from the edited spec, so what the vehicles are made of is exactly the
    # model without extensions; a marker adds a body that touches nothing.
    base = _vehicle_trajectory(AIRBORNE)
    assert np.abs(base[-1] - base[0]).max() > 0.01  # the vehicle does move
    for extensions in ({"idle": {"class": Extension}}, {"watch": {"class": Recorder}}, {"goal": {"class": MARKER}}):
        assert np.array_equal(_vehicle_trajectory({**AIRBORNE, "extensions": extensions}), base), extensions


def test_the_calls_come_in_order_with_this_steps_inputs():
    with Simulator(overrides={**AIRBORNE, "extensions": {"rec": {"class": Recorder, "anything": 1}}}, seed=7) as sim:
        state = sim.reset()
        recorder = sim.extensions["rec"]
        assert recorder.config == {"anything": 1}
        assert recorder.calls == ["edit_spec", "bind", "reset", "state"]
        context = recorder.context
        assert [(vehicle.index, vehicle.name, vehicle.body) for vehicle in context.vehicles] == [
            (0, sim.scene.uav_specs[0].name, sim.scene.uav_specs[0].body_name)
        ]
        assert context.timestep == pytest.approx(float(sim.scene.model.opt.timestep))
        assert context.seed == 7
        assert context.params_dir == REPOSITORY_ROOT
        assert context.params["extensions"]["rec"]["anything"] == 1
        assert state["extensions"]["rec"] == {"time": state["time"], "array": [0, 1, 2], "count": 2, "pair": [0.5, 1]}

        recorder.calls.clear()
        state = sim.step(THRUSTS, steps=3, extensions={"rec": {"go": True}})
        assert recorder.calls == ["apply_command", "before_step", "before_step", "before_step", "state"]
        assert recorder.commands == [{"go": True}]
        np.testing.assert_array_equal(recorder.ctrl_seen, THRUSTS)
        assert state["extensions"]["rec"]["time"] == pytest.approx(3 * float(sim.scene.model.opt.timestep))


def test_the_marker_moves_at_its_speed_and_reports_the_distances():
    extensions = {"goal": {"class": MARKER, "position": [1.0, 0.0, 1.0], "max_speed": 0.5}, "flag": {"class": MARKER}}
    with Simulator(overrides={**AIRBORNE, "extensions": extensions}) as sim:
        state = sim.reset()
        model = sim.scene.model
        geom = model.geom("goal_marker_geom")
        assert (int(geom.contype[0]), int(geom.conaffinity[0])) == (0, 0)
        assert model.body_mocapid[model.body("goal_marker").id] >= 0
        assert state["extensions"]["goal"]["position"] == [1.0, 0.0, 1.0]
        vehicle = sim.truth()["position"]
        assert state["extensions"]["goal"]["distance"][0] == pytest.approx(np.linalg.norm(vehicle - [1.0, 0.0, 1.0]))

        command = {"goal": {"position": [1.0, 1.0, 1.0]}, "flag": {"position": [2.0, 2.0, 2.0]}}
        state = sim.step(THRUSTS, steps=200, extensions=command)
        np.testing.assert_allclose(state["extensions"]["goal"]["position"], [1.0, 0.1, 1.0], atol=1e-9)
        assert state["extensions"]["goal"]["target"] == [1.0, 1.0, 1.0]
        assert state["extensions"]["flag"]["position"] == [2.0, 2.0, 2.0]  # no max_speed: it jumps
        state = sim.step(THRUSTS, steps=2000)
        assert state["extensions"]["goal"]["position"] == [1.0, 1.0, 1.0]


def test_a_command_an_extension_cannot_use_raises_in_process():
    with Simulator(overrides={"extensions": {"goal": {"class": MARKER}, "idle": {"class": Extension}}}) as sim:
        sim.reset()
        for command, message in (
            ({"nope": {}}, "no extension 'nope'"),
            ({"goal": {"position": [1.0, 2.0]}}, "3 finite numbers"),
            ({"goal": {"speed": 1.0}}, "takes"),
            ({"idle": {"x": 1}}, "takes no commands"),
        ):
            with pytest.raises(ExtensionError, match=message):
                sim.step(THRUSTS, extensions=command)
        assert sim.time == 0.0  # nothing was applied or stepped
    with Simulator() as sim:
        sim.reset()
        with pytest.raises(ValueError, match="no extensions"):
            sim.step(THRUSTS, extensions={"goal": {}})


def test_loader_errors_name_the_entry(tmp_path):
    cases = [
        ([], "must be an object"),
        ({"1goal": {"class": MARKER}}, "extension name '1goal'"),
        ({"my-goal": {"class": MARKER}}, "extension name 'my-goal'"),
        ({"goal": [MARKER]}, "extensions.goal must be an object"),
        ({"goal": {"class": MARKER, "enabled": "yes"}}, "enabled must be true or false"),
        ({"goal": {"position": [0, 0, 1]}}, "no class"),
        ({"goal": {"class": "wheeled_uav.extensions.marker"}}, "module:ClassName"),
        ({"goal": {"class": 42}}, "module:ClassName"),
        ({"goal": {"class": "no_such_module_for_extensions:Thing"}}, "cannot import 'no_such_module_for_extensions'"),
        ({"goal": {"class": "wheeled_uav.extensions.marker:Nope"}}, "has no 'Nope'"),
        ({"goal": {"class": "wheeled_uav.extensions.marker:__doc__"}}, "is not a class"),
        ({"goal": {"class": MARKER, "path": "missing_folder"}}, "is not a directory"),
        ({"goal": {"class": MARKER, "path": 3}}, "path must be"),
        ({"goal": {"class": MARKER, "colour": [1, 0, 0, 1]}}, r"extensions.goal: unknown keys \['colour'\]"),
        ({"goal": {"class": MARKER, "radius": -1}}, "radius must be positive"),
    ]
    for section, message in cases:
        with pytest.raises(ExtensionError, match=message):
            load_extensions({"extensions": section}, tmp_path)
    assert load_extensions({}, tmp_path) is None


class _AddsAnActuator(Extension):
    def edit_spec(self, spec, context):
        body = spec.worldbody.add_body(name="spinner")
        body.add_joint(name="spinner_joint", type=mujoco.mjtJoint.mjJNT_HINGE)
        body.add_geom(size=[0.05, 0.0, 0.0])
        spec.add_actuator(target="spinner_joint", trntype=mujoco.mjtTrn.mjTRN_JOINT)


class _ChangesTheTimestep(Extension):
    def edit_spec(self, spec, context):
        spec.option.timestep = context.timestep / 2


class _RepeatsAName(Extension):
    def edit_spec(self, spec, context):
        spec.worldbody.add_body(name="twice")
        spec.worldbody.add_body(name="twice")


class _BreaksTheModel(Extension):
    def edit_spec(self, spec, context):
        spec.worldbody.add_body(name="massless").add_freejoint()  # a moving body without mass


def test_extensions_may_not_take_the_controls_or_the_timestep_or_break_the_model(tmp_path):
    for factory, message in (
        (_AddsAnActuator, "must not add actuators"),
        (_ChangesTheTimestep, "must not change the timestep"),
        (_RepeatsAName, "extensions.bad: editing the model: .*repeated name 'twice'"),
        (_BreaksTheModel, r"does not compile after the extensions' edits \(bad\)"),
    ):
        params = load_vehicle_params()
        params["extensions"] = {"bad": {"class": factory}}
        with pytest.raises(ExtensionError, match=message):
            load_simulation_scene(SimulationRequest(generated_xml_dir=tmp_path, headless=True), params=params)


def test_a_class_is_imported_from_a_folder_next_to_the_params_file(tmp_path):
    folder = tmp_path / "my_extensions"
    folder.mkdir()
    (folder / "probe_extension_module.py").write_text(
        "from wheeled_uav.extensions import Extension\n"
        "class Probe(Extension):\n"
        "    keys = frozenset({'gain'})\n"
        "    def state(self, model, data):\n"
        "        return {'gain': self.config['gain']}\n",
        encoding="utf-8",
    )
    params_path = _write_params(
        tmp_path,
        {
            "_note": "names and keys starting with _ are comments",
            "probe": {"class": "probe_extension_module:Probe", "path": "my_extensions", "gain": 3, "_why": "a comment"},
        },
    )
    scene = load_simulation_scene(SimulationRequest(params_path=params_path, generated_xml_dir=tmp_path, headless=True))
    probe = scene.extensions["probe"]
    assert type(probe).__name__ == "Probe" and probe.config == {"gain": 3}
    assert scene.extensions.states(scene.model, scene.data) == {"probe": {"gain": 3}}


def test_extension_settings_can_be_randomized(tmp_path):
    params = load_vehicle_params()
    params["extensions"] = {"goal": {"class": MARKER, "radius": 0.05}}
    params["randomization"] = {"parameters": {"extensions.goal.radius": {"uniform": [0.1, 0.2]}}}
    sizes = []
    for seed in (1, 2):
        scene = load_simulation_scene(SimulationRequest(generated_xml_dir=tmp_path, headless=True, seed=seed), params=params)
        sizes.append(float(scene.model.geom("goal_marker_geom").size[0]))
        assert scene.extensions["goal"].radius == sizes[-1]
    assert all(0.1 <= size <= 0.2 for size in sizes) and sizes[0] != sizes[1]


def test_check_model_lists_the_extensions_and_writes_the_extended_xml(tmp_path, capsys):
    params_path = _write_params(tmp_path, {"goal": {"class": MARKER}})
    assert validate_model(SimulationRequest(params_path=params_path, generated_xml_dir=tmp_path)) == 0
    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert summary["extensions"] == ["goal"]
    extended = Path(summary["xml_path"])
    assert extended.name.endswith(".extended.xml")
    assert extended.with_name(extended.name.replace(".extended.xml", ".xml")).is_file()  # the vehicles' XML, as before
    model = mujoco.MjModel.from_xml_path(str(extended))
    assert model.body("goal_marker").id > 0


def test_the_runtime_refuses_parallel_physics_and_binary_states(tmp_path):
    params = load_vehicle_params()
    params["extensions"] = {"goal": {"class": MARKER}}
    for request, message in (
        (SimulationRequest(generated_xml_dir=tmp_path, headless=True, physics_threads=1), "parallel physics"),
        (SimulationRequest(generated_xml_dir=tmp_path, headless=True, state_format="binary"), "binary"),
    ):
        scene = load_simulation_scene(request, params=params)
        with pytest.raises(ValueError, match=message):
            SimulationRuntime(scene)
    scene = load_simulation_scene(SimulationRequest(generated_xml_dir=tmp_path, headless=True), params=params)
    with pytest.raises(ValueError, match="no extension states"):
        build_state_payload(scene, 1.0, sequence=1, wall_time_send_ns=0, state_format="binary", extension_states={"goal": {}})


def test_command_packets_carry_the_extension_commands(capsys):
    params = load_vehicle_params()
    command = {"goal": {"position": [1.0, 2.0, 3.0]}}
    packet = parse_command_packet({"rotor_thrusts": THRUSTS, "extensions": command}, params)
    assert packet.extension_commands == command
    assert parse_command_packet({"rotor_thrusts": THRUSTS}, params).extension_commands is None
    assert parse_command_packet({"rotor_thrusts": THRUSTS, "extensions": {}}, params).extension_commands is None
    for packet in (
        parse_multi_uav_command_packet({"rotor_thrusts": [THRUSTS, THRUSTS], "extensions": command}, params, 2),
        parse_multi_uav_command_packet({"uavs": [{"rotor_thrusts": THRUSTS}] * 2, "extensions": command}, params, 2),
    ):
        assert packet.extension_commands == command
    for _ in range(2):
        malformed = parse_command_packet({"rotor_thrusts": THRUSTS, "extensions": [1, 2]}, params)
        assert malformed.rotor_thrusts == THRUSTS and malformed.extension_commands is None
    assert capsys.readouterr().out.count("ignoring the extensions field") == 1


class _QueuedSocket:
    """Stands in for the simulator's non-blocking socket: hands out the queued datagrams."""

    def __init__(self):
        self.datagrams: list[bytes] = []

    def recvfrom(self, size):
        if not self.datagrams:
            raise BlockingIOError
        return self.datagrams.pop(0), ("127.0.0.1", 1)


def test_network_commands_reach_the_extensions_and_bad_ones_are_warned_about_once(tmp_path, capsys):
    params = load_vehicle_params()
    params["extensions"] = {"goal": {"class": MARKER}}
    scene = load_simulation_scene(SimulationRequest(generated_xml_dir=tmp_path, headless=True), params=params)
    dispatcher = ControlCommandDispatcher(scene.params, scene.fidelity, 1)
    dispatcher.bind_bodies(scene.model, scene.uav_specs)
    dispatcher.bind_extensions(scene.extensions)
    sock = _QueuedSocket()

    def send(extensions):
        sock.datagrams.append(json.dumps({"rotor_thrusts": THRUSTS, "extensions": extensions}).encode())
        dispatcher.apply_next_command(sock, scene.data)

    send({"goal": {"position": [0.5, 0.5, 0.5]}})
    assert scene.extensions["goal"].target.tolist() == [0.5, 0.5, 0.5]
    for _ in range(2):
        send({"nope": {"position": [0.0, 0.0, 0.0]}})
        send({"goal": {"position": "here"}})
    assert scene.extensions["goal"].target.tolist() == [0.5, 0.5, 0.5]
    assert dispatcher.applied_command_count == 5  # the rotor commands were applied all the same
    out = capsys.readouterr().out
    assert out.count("warning: ignoring no extension 'nope'") == 1
    assert out.count("warning: ignoring the command for extension 'goal'") == 1

    plain = load_simulation_scene(SimulationRequest(generated_xml_dir=tmp_path / "plain", headless=True))
    dispatcher = ControlCommandDispatcher(plain.params, plain.fidelity, 1)
    for _ in range(2):
        sock.datagrams.append(json.dumps({"rotor_thrusts": THRUSTS, "extensions": {"goal": {}}}).encode())
        dispatcher.apply_next_command(sock, plain.data)
    assert capsys.readouterr().out.count("the params have no extensions") == 1


def test_state_packets_carry_the_extension_states_at_the_top_level(tmp_path):
    states = {"goal": {"position": [1.0, 2.0, 3.0]}}
    for num_uavs in (1, 2):
        scene = load_simulation_scene(SimulationRequest(generated_xml_dir=tmp_path, headless=True, num_uavs=num_uavs))
        packet = json.loads(build_state_payload(scene, 1.0, sequence=1, wall_time_send_ns=0, extension_states=states))
        assert packet["extensions"] == states
        assert all("extensions" not in uav for uav in packet.get("uavs", []))
        assert "extensions" not in json.loads(build_state_payload(scene, 1.0, sequence=1, wall_time_send_ns=0))


def test_lockstep_over_udp_commands_the_marker_and_reports_it(tmp_path):
    params_path = _write_params(tmp_path, {"goal": {"class": MARKER, "position": [0.0, 0.0, 2.0]}})
    target = [0.5, -0.5, 1.5]
    command = json.dumps({"rotor_thrusts": [0.0] * 4, "extensions": {"goal": {"position": target}}}).encode()
    with create_udp_socket(recv_port=0) as simulator_socket, create_udp_socket(recv_port=0) as controller_socket:
        controller_socket.setblocking(True)
        scene = load_simulation_scene(
            SimulationRequest(
                send_port=controller_socket.getsockname()[1],
                params_path=params_path,
                generated_xml_dir=tmp_path,
                pacing_mode="lockstep",
                duration_seconds=0.03,
                headless=True,
            )
        )
        simulator = threading.Thread(target=run_headless_loop, args=(scene, simulator_socket))
        simulator.start()
        controller_socket.settimeout(10.0)
        states = [json.loads(controller_socket.recvfrom(65535)[0])]
        answered = set()
        controller_socket.settimeout(2.0)
        while True:
            if states[-1]["sequence"] not in answered:
                answered.add(states[-1]["sequence"])
                controller_socket.sendto(command, simulator_socket.getsockname())
            try:
                states.append(json.loads(controller_socket.recvfrom(65535)[0]))
            except socket.timeout:
                break
        simulator.join(timeout=10.0)
    assert not simulator.is_alive()
    assert states[0]["extensions"]["goal"]["position"] == [0.0, 0.0, 2.0]
    assert states[-1]["extensions"]["goal"]["position"] == target
    assert len(states) > 3


def test_to_jsonable_turns_numpy_values_into_plain_ones():
    value = to_jsonable({"a": np.array([[1.0, 2.0]]), "b": (np.int32(3), np.bool_(True)), 4: [np.float32(0.5)]})
    assert value == {"a": [[1.0, 2.0]], "b": [3, True], "4": [0.5]}
    assert json.dumps(value)


def test_marker_is_an_extension_with_a_closed_set_of_keys():
    assert issubclass(Marker, Extension) and Marker.keys == {"position", "radius", "rgba", "max_speed"}


OCTAVE = shutil.which("octave-cli")
MATLAB = shutil.which("matlab")


@pytest.mark.skipif(OCTAVE is None and MATLAB is None, reason="neither GNU Octave nor MATLAB is installed")
def test_matlab_commands_carry_the_extension_commands():
    script = (
        f"addpath('{(REPOSITORY_ROOT / 'matlab').as_posix()}');"
        "options = uavsim.Protocol.build_command_options('thrust', 2e-5);"
        "goal = struct('goal', struct('position', [1 2 3]));"
        "single = uavsim.Protocol.build_control_command([1;2;3;4], options, 'extensions', goal);"
        "plain = uavsim.Protocol.build_control_command([1;2;3;4], options);"
        "other = uavsim.Protocol.build_control_command([5;6;7;8], options, 'extensions', struct('flag', struct('position', [0 0 1])));"
        "disp(uavsim.Protocol.encode_control_command(single));"
        "disp(uavsim.Protocol.encode_control_command(plain));"
        "disp(uavsim.Protocol.encode_multi_uav_control_command({single, plain, other}, options));"
    )
    command = [OCTAVE, "--no-gui", "--quiet", "--eval", script] if OCTAVE else [MATLAB, "-batch", script]
    lines = subprocess.run(command, capture_output=True, text=True, timeout=600, check=True).stdout.strip().splitlines()
    single, plain, multi = (json.loads(line) for line in lines[-3:])
    params = load_vehicle_params()
    assert parse_command_packet(single, params).extension_commands == {"goal": {"position": [1, 2, 3]}}
    assert "extensions" not in plain
    packet = parse_multi_uav_command_packet(multi, params, 3)
    assert packet.extension_commands == {"goal": {"position": [1, 2, 3]}, "flag": {"position": [0, 0, 1]}}
    assert packet.rotor_thrusts[2] == [5.0, 6.0, 7.0, 8.0]
