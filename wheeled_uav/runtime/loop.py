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
from .fidelity import ActuatorModel
from .scene import SimulationRequest, SimulationScene, load_simulation_scene, resolve_timing_config
from .state_publisher import StatePayloadPublisher
from .visuals import FrameRecorder, WallOverlayAnimator, configure_viewer

__all__ = [
    "run_headless_loop",
    "run_simulation",
    "run_viewer_loop",
]


class _SimulationRuntime:
    """Bundles the per-run runtime components shared by both loop flavors."""

    def __init__(self, scene: SimulationScene):
        self.scene = scene
        self.state_publisher = StatePayloadPublisher(scene)
        self.command_dispatcher = ControlCommandDispatcher(scene.params, scene.fidelity, scene.request.num_uavs)
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
        self.overlay_animator = WallOverlayAnimator(scene)
        self.frame_recorder = FrameRecorder(scene, scene.request.record_path)
        self.timing_config = resolve_timing_config(scene)
        _, self.send_port = scene.request.resolved_ports()

    def step_physics_once(self) -> None:
        """Actuator/aero effects + visual bookkeeping + one mj_step."""
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
        self.command_dispatcher.refresh_body_wrenches(scene.data, force_rebuild=self.aerodynamics_model.active)
        self.actuator_model.apply(scene.data)
        self.aerodynamics_model.apply(scene.data)
        self.overlay_animator.update(scene.data, active)
        self.frame_recorder.capture(scene.data, active)
        mujoco.mj_step(scene.model, scene.data)

    def publish_state(self, sock, realtime_factor: float, session_timing: SessionTimingTracker) -> None:
        self.state_publisher.send_state(
            sock,
            self.scene.request.state_target_ip,
            self.send_port,
            realtime_factor,
            self.actuator_model.snapshot(),
            aero_snapshot=self.aerodynamics_model.snapshot(),
            timing=session_timing.snapshot(self.scene.data.time, realtime_factor),
        )


def _run_stepping_loop(scene: SimulationScene, sock, viewer: mujoco.viewer.Handle | None) -> None:
    runtime = _SimulationRuntime(scene)
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

    # Optionally freeze the vehicles at their spawn pose until the first
    # control command arrives. Without this, an unactuated two-wheeled drone
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

    try:
        with timer_context():
            while (viewer is None or viewer.is_running()) and (duration_seconds is None or scene.data.time < duration_seconds):
                runtime.command_dispatcher.apply_next_command(sock, scene.data)
                # Release on the first APPLIED command (not merely received):
                # under HIL delay/loss the plant should stay frozen until a
                # command actually reaches it.
                holding = hold_until_first_command and runtime.command_dispatcher.applied_command_count == 0
                runtime.step_physics_once()
                if holding:
                    # Re-freeze AFTER the step so the published state is
                    # exactly the spawn pose; a pre-step reset would leave one
                    # integration step of gravity in the published velocity.
                    # data.time keeps advancing so --duration-seconds still
                    # bounds controller-less runs.
                    scene.data.qpos[:] = initial_qpos
                    scene.data.qvel[:] = 0.0
                    mujoco.mj_forward(scene.model, scene.data)
                step_count += 1
                if step_count % publish_every == 0:
                    runtime.publish_state(sock, realtime_tracker.realtime_factor, session_timing)
                if viewer is not None and step_count % viewer_sync_every == 0:
                    viewer.sync()

                pacer.pace()
                loop_time = time.perf_counter()
                realtime_tracker.update(loop_time - previous_loop_time, timestep)
                previous_loop_time = loop_time
                if use_realtime_pacing:
                    lag_monitor.check(realtime_tracker.realtime_factor)
    finally:
        runtime.frame_recorder.close()


def _run_lockstep_loop(scene: SimulationScene, sock, viewer: mujoco.viewer.Handle | None) -> None:
    """Deterministic co-simulation: publish state, block until a control
    command arrives, advance one control period, repeat.

    Removes the wall-clock coupling of the realtime loop (the physics only
    advances in response to controller commands), which makes runs exactly
    reproducible and lets slow controllers or large teams run without
    real-time pressure. While waiting, the current state is republished every
    second so a late-connecting controller can join.
    """
    runtime = _SimulationRuntime(scene)
    publish_every = runtime.timing_config.state_publish_every_n_steps
    duration_seconds = scene.request.duration_seconds
    session_timing = SessionTimingTracker.start(scene.data.time, timing=runtime.timing_config)

    def publish_state() -> None:
        runtime.publish_state(sock, 1.0, session_timing)

    publish_state()
    try:
        while (viewer is None or viewer.is_running()) and (duration_seconds is None or scene.data.time < duration_seconds):
            if not runtime.command_dispatcher.wait_for_command(sock, scene.data, republish=publish_state):
                print("lockstep: no control command received within the timeout; stopping simulation")
                return
            for _ in range(publish_every):
                runtime.step_physics_once()
            publish_state()
            if viewer is not None:
                viewer.sync()
    finally:
        runtime.frame_recorder.close()


def run_viewer_loop(scene: SimulationScene, sock) -> None:
    with mujoco.viewer.launch_passive(scene.model, scene.data) as viewer:
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
