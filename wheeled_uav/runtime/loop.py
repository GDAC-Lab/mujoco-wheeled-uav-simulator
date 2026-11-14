"""Top-level simulation loops: real-time/accelerated stepping and lockstep."""

from __future__ import annotations

import time
from contextlib import nullcontext

import mujoco
import mujoco.viewer

from ..config import build_aerodynamics_config
from ..paths import PathResolver
from ..protocol import create_udp_socket
from ..timing import (
    RealtimeLagMonitor,
    RealtimeTracker,
    SessionTimingTracker,
    build_pacer,
    high_resolution_os_timer,
    resolve_viewer_sync_every_n_steps,
)
from .aerodynamics import AerodynamicsModel
from .command_dispatcher import ControlCommandDispatcher
from ..plant import PlantModel
from .fidelity import ActuatorModel, ActuatorSnapshot
from .parallel import ParallelPhysics
from .scene import (
    SimulationRequest,
    SimulationScene,
    load_simulation_scene,
    resolve_physics_threads,
    resolve_seed,
    resolve_state_format,
    resolve_timing_config,
)
from .state_publisher import StatePayloadPublisher
from .visuals import FrameRecorder, WallOverlayAnimator, configure_viewer, open_passive_viewer

__all__ = [
    "SimulationRuntime",
    "run_headless_loop",
    "run_simulation",
    "run_viewer_loop",
]


class SimulationRuntime:
    """Bundles the per-run runtime components shared by the loops and the in-process Simulator."""

    def __init__(self, scene: SimulationScene):
        physics_threads = resolve_physics_threads(scene)
        if physics_threads > 0 and scene.request.record_path is not None:
            raise ValueError("--record is not supported with parallel physics (simulation.physics_threads)")
        self.scene = scene
        self.plant = None
        if scene.plant is not None:
            self.plant = PlantModel(
                scene.model,
                scene.params,
                scene.plant,
                scene.uav_specs,
                scene.sensor_layouts,
                seed=resolve_seed(scene),
                surface_evaluator=scene.surface_evaluator,
            )
        self.state_publisher = StatePayloadPublisher(scene, plant=self.plant)
        self.command_dispatcher = ControlCommandDispatcher(
            scene.params, scene.fidelity, scene.request.num_uavs, seed=resolve_seed(scene)
        )
        self.command_dispatcher.bind_bodies(scene.model, scene.uav_specs)
        self.actuator_model = ActuatorModel(
            scene.model,
            scene.fidelity,
            float(scene.params["actuation"]["thrust_coefficient"]),
        )
        self.aerodynamics_model = AerodynamicsModel(
            scene.model,
            scene.params,
            build_aerodynamics_config(scene.params),
            scene.uav_specs,
            scene.sensor_layouts,
        )
        if self.aerodynamics_model.active and resolve_state_format(scene) == "binary":
            raise ValueError(
                'simulation.state_format "binary" carries no aerodynamics log; use the JSON state format '
                "while the aerodynamics section is enabled"
            )
        self.overlay_animator = WallOverlayAnimator(scene)
        self.frame_recorder = FrameRecorder(scene, scene.request.record_path)
        self.timing_config = resolve_timing_config(scene)
        _, self.send_port = scene.request.resolved_ports()
        # Wall time of the simulator's own work, reported in the timing block
        # of each state packet: the steps since the last publish, the previous
        # packet's build and send, and (lockstep) the wait for the command.
        self.step_wall_seconds = 0.0
        self.publish_wall_seconds = 0.0
        self.command_wait_wall_seconds: float | None = None
        self.parallel_physics = ParallelPhysics(scene, physics_threads) if physics_threads > 0 else None

    def _hooks_change_within_a_step_batch(self) -> bool:
        # Actuator lag (HIL), the plant model and aerodynamics rewrite the
        # controls or wrenches every physics step (and the plant keeps a sensor
        # history); without them a batch of steps can run in one go.
        return self.scene.fidelity.mode == "hil" or self.aerodynamics_model.active or self.plant is not None

    def _rebuild_wrench_rows(self) -> bool:
        # Writers that add to xfrc_applied need the rows rebuilt from the command every step.
        return self.aerodynamics_model.active or (self.plant is not None and self.plant.applies_body_forces)

    def advance(self, nstep: int) -> None:
        """Step the physics `nstep` times (one control period in lockstep)."""
        if self.parallel_physics is None or self._hooks_change_within_a_step_batch():
            for _ in range(nstep):
                self.step_physics_once()
            return
        started = time.perf_counter()
        scene = self.scene
        active = self.command_dispatcher.applied_command_count > 0
        self.command_dispatcher.refresh_body_wrenches(scene.data)
        self.actuator_model.apply(scene.data)
        self.overlay_animator.update(scene.data, active)
        self.parallel_physics.step(scene.data, nstep)
        self.actuator_model.restore(scene.data)
        self.step_wall_seconds += time.perf_counter() - started

    def step_physics_once(self) -> None:
        """Actuator/aero effects + visual bookkeeping + one mj_step."""
        started = time.perf_counter()
        scene = self.scene
        # "Active" means a command has actually been APPLIED to the physics —
        # under HIL network delay/loss a packet can be received long before it
        # reaches the plant, and recordings/overlays should not start early.
        active = self.command_dispatcher.applied_command_count > 0
        # Order matters: the dispatcher rewrites the command-owned body-wrench
        # rows first, then the aerodynamics model adds its force on top. With
        # aero active the rows must be rebuilt every step (its += needs a fresh
        # base); otherwise they are only touched while wrench commands are
        # active, so viewer drag perturbations on the vehicles work natively.
        self.command_dispatcher.refresh_body_wrenches(scene.data, force_rebuild=self._rebuild_wrench_rows())
        self.actuator_model.apply(scene.data)
        if self.plant is not None:
            self.plant.apply(scene.data)
        self.aerodynamics_model.apply(scene.data)
        self.overlay_animator.update(scene.data, active)
        self.frame_recorder.capture(scene.data, active)
        if self.parallel_physics is None:
            mujoco.mj_step(scene.model, scene.data)
        else:
            self.parallel_physics.step(scene.data, 1)
        if self.plant is not None:
            self.plant.restore(scene.data)
        self.actuator_model.restore(scene.data)
        if self.plant is not None:
            self.plant.record(scene.data)
        self.step_wall_seconds += time.perf_counter() - started

    def actuator_snapshot(self):
        """Commanded and applied rotor thrusts for the actuator log: the plant's rotor model when it has one."""
        if self.plant is not None and (pair := self.plant.requested_and_applied()) is not None:
            return ActuatorSnapshot(requested_ctrl=pair[0], applied_ctrl=pair[1])
        return self.actuator_model.snapshot()

    def publish_state(self, sock, realtime_factor: float, session_timing: SessionTimingTracker) -> None:
        started = time.perf_counter()
        timing = session_timing.snapshot(self.scene.data.time, realtime_factor)
        timing["step_wall_seconds"] = self.step_wall_seconds
        timing["publish_wall_seconds"] = self.publish_wall_seconds
        if self.command_wait_wall_seconds is not None:
            timing["command_wait_wall_seconds"] = self.command_wait_wall_seconds
        contact_reports = None
        if self.parallel_physics is not None:
            contact_reports = self.parallel_physics.contact_reports(
                self.scene.surface_evaluator, self.state_publisher.include_contact_details
            )
        self.state_publisher.send_state(
            sock,
            self.scene.request.state_target_ip,
            self.send_port,
            realtime_factor,
            self.actuator_snapshot(),
            aero_snapshot=self.aerodynamics_model.snapshot(),
            timing=timing,
            contact_reports=contact_reports,
        )
        self.step_wall_seconds = 0.0
        self.publish_wall_seconds = time.perf_counter() - started

    def close(self) -> None:
        self.frame_recorder.close()
        if self.parallel_physics is not None:
            self.parallel_physics.close()

    def republish_state(self, sock) -> None:
        self.state_publisher.resend_last(sock, self.scene.request.state_target_ip, self.send_port)

    def flush_pending_states(self, sock) -> float:
        """Send the state packets the HIL network model held back that are now due (see StatePayloadPublisher.flush_due)."""
        return self.state_publisher.flush_due(sock, self.scene.request.state_target_ip, self.send_port)

    def drain_pending_states(self, sock) -> None:
        """At the end of a run: send the held-back state packets when they fall due."""
        self.state_publisher.drain(sock, self.scene.request.state_target_ip, self.send_port)


def _run_stepping_loop(scene: SimulationScene, sock, viewer: mujoco.viewer.Handle | None) -> None:
    runtime = SimulationRuntime(scene)
    realtime_tracker = RealtimeTracker()
    lag_monitor = RealtimeLagMonitor()
    timestep = float(scene.model.opt.timestep)
    publish_every = runtime.timing_config.state_publish_every_n_steps
    viewer_sync_every = resolve_viewer_sync_every_n_steps(scene.params, timestep)
    duration_seconds = scene.request.duration_seconds
    use_realtime_pacing = runtime.timing_config.pacing_mode == "realtime"
    pacer = build_pacer(runtime.timing_config)
    step_count = 0
    previous_loop_time = time.perf_counter()
    session_timing = SessionTimingTracker.start(scene.data.time, timing=runtime.timing_config)
    timer_context = high_resolution_os_timer if use_realtime_pacing else nullcontext

    # Optionally freeze the vehicles at their spawn state (pose and initial
    # velocity) until the first control command arrives. Without this, an unactuated two-wheeled drone
    # tips over about its axle during the controller connection window and the
    # controller then starts from an unrecoverable attitude.
    #
    # Policy: automatic/scripted runs (hover, formation, batch studies) want
    # this ON for a clean, reproducible initial condition, so the default
    # vehicle_params.json enables it. A preset that starts physically resting
    # on the ground can disable it instead so the run begins from settled
    # rest. The CLI flag --hold-until-first-command /
    # --no-hold-until-first-command overrides the params file per run.
    if scene.request.hold_until_first_command is not None:
        hold_until_first_command = bool(scene.request.hold_until_first_command)
    else:
        hold_until_first_command = bool(scene.params.get("simulation", {}).get("hold_until_first_command", False))
    initial_qpos = scene.data.qpos.copy()
    initial_qvel = scene.data.qvel.copy()
    was_holding = False

    try:
        with timer_context():
            while (viewer is None or viewer.is_running()) and (duration_seconds is None or scene.data.time < duration_seconds):
                runtime.command_dispatcher.apply_next_command(sock, scene.data)
                # Release on the first APPLIED command (not merely received):
                # under HIL delay/loss the plant should stay frozen until a
                # command actually reaches it.
                holding = hold_until_first_command and runtime.command_dispatcher.applied_command_count == 0
                if was_holding and not holding and runtime.plant is not None:
                    # Released like a vehicle already flying: the rotors start at
                    # the first command's speed instead of spinning up from rest.
                    runtime.plant.settle(scene.data)
                was_holding = holding
                runtime.step_physics_once()
                if holding:
                    # Re-freeze AFTER the step so the published state is
                    # exactly the spawn state (pose and initial velocity); a
                    # pre-step reset would leave one integration step of
                    # gravity in the published velocity. data.time keeps
                    # advancing so --duration-seconds still bounds
                    # controller-less runs.
                    scene.data.qpos[:] = initial_qpos
                    scene.data.qvel[:] = initial_qvel
                    mujoco.mj_forward(scene.model, scene.data)
                    if runtime.parallel_physics is not None:
                        runtime.parallel_physics.load_state(scene.data)
                step_count += 1
                if step_count % publish_every == 0:
                    runtime.publish_state(sock, realtime_tracker.realtime_factor, session_timing)
                else:
                    # A state the HIL network model held back leaves when it is
                    # due, not with the next publish.
                    runtime.flush_pending_states(sock)
                if viewer is not None and step_count % viewer_sync_every == 0:
                    if runtime.parallel_physics is not None:
                        mujoco.mj_kinematics(scene.model, scene.data)  # poses for display
                    viewer.sync()

                pacer.pace()
                loop_time = time.perf_counter()
                realtime_tracker.update(loop_time - previous_loop_time, timestep)
                previous_loop_time = loop_time
                if use_realtime_pacing:
                    lag_monitor.check(realtime_tracker.realtime_factor)
            runtime.drain_pending_states(sock)
    finally:
        runtime.close()


def _run_lockstep_loop(scene: SimulationScene, sock, viewer: mujoco.viewer.Handle | None) -> None:
    """Deterministic co-simulation: publish state, block until a control
    command arrives, advance one control period, repeat.

    Removes the wall-clock coupling of the realtime loop (the physics only
    advances in response to controller commands), which makes runs exactly
    reproducible and lets slow controllers or large teams run without
    real-time pressure. While waiting, the current state packet is re-sent
    unchanged every second so a late-connecting controller can join; the copy
    carries the same sequence number, so controllers skip it as a duplicate.
    A command that arrives but is thrown away (HIL packet loss, stale) is
    answered with a freshly published state, which the controller answers.
    """
    runtime = SimulationRuntime(scene)
    publish_every = runtime.timing_config.state_publish_every_n_steps
    duration_seconds = scene.request.duration_seconds
    session_timing = SessionTimingTracker.start(scene.data.time, timing=runtime.timing_config)

    def publish_state() -> None:
        runtime.publish_state(sock, 1.0, session_timing)

    def republish_state() -> None:
        runtime.republish_state(sock)

    def send_due_states() -> float:
        return runtime.flush_pending_states(sock)

    runtime.command_wait_wall_seconds = 0.0  # the first state follows no command
    publish_state()
    try:
        while (viewer is None or viewer.is_running()) and (duration_seconds is None or scene.data.time < duration_seconds):
            wait_started = time.perf_counter()
            if not runtime.command_dispatcher.wait_for_command(
                sock, scene.data, republish=republish_state, request_again=publish_state, service=send_due_states
            ):
                print("lockstep: no control command received within the timeout; stopping simulation")
                return
            runtime.command_wait_wall_seconds = time.perf_counter() - wait_started
            runtime.advance(publish_every)
            publish_state()
            if viewer is not None:
                if runtime.parallel_physics is not None:
                    mujoco.mj_kinematics(scene.model, scene.data)  # poses for display
                viewer.sync()
        runtime.drain_pending_states(sock)
    finally:
        runtime.close()


def run_viewer_loop(scene: SimulationScene, sock) -> None:
    with open_passive_viewer(scene.model, scene.data) as viewer:
        configure_viewer(viewer, scene)
        if resolve_timing_config(scene).pacing_mode == "lockstep":
            _run_lockstep_loop(scene, sock, viewer)
        else:
            _run_stepping_loop(scene, sock, viewer)


def run_headless_loop(scene: SimulationScene, sock) -> None:
    if resolve_timing_config(scene).pacing_mode == "lockstep":
        _run_lockstep_loop(scene, sock, None)
    else:
        _run_stepping_loop(scene, sock, None)


def run_simulation(request: SimulationRequest, *, path_resolver: PathResolver | None = None) -> None:
    """Load the scene for `request` and run it until closed or duration end."""
    recv_port, _ = request.resolved_ports()
    scene = load_simulation_scene(request, path_resolver=path_resolver)
    sock = create_udp_socket(udp_ip=request.bind_ip, recv_port=recv_port)

    try:
        if request.headless:
            run_headless_loop(scene, sock)
        else:
            run_viewer_loop(scene, sock)
    finally:
        sock.close()
