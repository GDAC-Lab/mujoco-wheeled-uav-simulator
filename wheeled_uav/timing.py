from __future__ import annotations

import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator, Literal

PacingMode = Literal["realtime", "accelerated", "lockstep"]
PACING_MODES: tuple[PacingMode, ...] = ("realtime", "accelerated", "lockstep")

__all__ = [
    "PACING_MODES",
    "PacingMode",
    "SimulationTimingConfig",
    "RealtimeTracker",
    "SessionTimingTracker",
    "StepPacer",
    "NullPacer",
    "RealtimeLagMonitor",
    "StateSampleTracker",
    "build_pacer",
    "build_state_sample_key",
    "compute_control_dt_seconds",
    "extract_sync_metrics",
    "high_resolution_os_timer",
    "parse_simulation_timing",
    "resolve_viewer_sync_every_n_steps",
    "sim_time_seconds",
]


@dataclass(frozen=True)
class SimulationTimingConfig:
    """Resolved timing knobs from vehicle_params.simulation."""

    physics_timestep_seconds: float
    state_publish_every_n_steps: int
    control_period_seconds: float
    viewer_fps: float
    pacing_mode: PacingMode = "realtime"

    @property
    def control_rate_hz(self) -> float:
        return 1.0 / self.control_period_seconds


def parse_simulation_timing(params: dict[str, Any]) -> SimulationTimingConfig:
    simulation = params["simulation"]
    physics_timestep_seconds = float(simulation["timestep"])
    if physics_timestep_seconds <= 0.0:
        raise ValueError("simulation.timestep must be positive")
    publish_every = int(simulation.get("state_publish_every_n_steps", 1))
    if publish_every < 1:
        raise ValueError("simulation.state_publish_every_n_steps must be >= 1")
    viewer_fps = float(simulation.get("viewer_fps", 60.0))
    if viewer_fps <= 0.0:
        raise ValueError("simulation.viewer_fps must be positive")
    pacing_mode = str(simulation.get("pacing_mode", "realtime")).strip().lower()
    if pacing_mode not in PACING_MODES:
        raise ValueError(f'simulation.pacing_mode must be one of {PACING_MODES}')
    return SimulationTimingConfig(
        physics_timestep_seconds=physics_timestep_seconds,
        state_publish_every_n_steps=publish_every,
        control_period_seconds=physics_timestep_seconds * publish_every,
        viewer_fps=viewer_fps,
        pacing_mode=pacing_mode,
    )


def resolve_viewer_sync_every_n_steps(params: dict[str, Any], timestep: float) -> int:
    timing = parse_simulation_timing(params)
    return max(1, round(1.0 / (timing.viewer_fps * timestep)))


def sim_time_seconds(state: dict[str, Any]) -> float | None:
    raw_time = state.get("time", state.get("sim_time"))
    if raw_time is None:
        return None
    return float(raw_time)


def build_state_sample_key(state: dict[str, Any]) -> str:
    # Composite key over (sequence, simulator time) — matches the MATLAB controller.
    t_s = sim_time_seconds(state)
    sequence = state.get("sequence")
    if sequence is not None:
        if t_s is not None:
            return f"seq={int(sequence)}|t={t_s:.17g}"
        return f"seq={int(sequence)}"
    if t_s is not None:
        return f"t={t_s:.17g}"
    return ""


@dataclass
class StateSampleTracker:
    """Drop duplicate UDP state reads; one control evaluation per simulator sample."""

    last_key: str = ""

    def is_new(self, state: dict[str, Any]) -> bool:
        key = build_state_sample_key(state)
        if not key or key == self.last_key:
            return False
        self.last_key = key
        return True


def compute_control_dt_seconds(
    last_sim_time: float | None,
    current_sim_time: float | None,
    timing: SimulationTimingConfig,
) -> float:
    # Integrators/differentiators must follow simulator time deltas, never wall clock.
    expected_dt = timing.control_period_seconds
    dt_min = min(timing.physics_timestep_seconds, expected_dt)
    dt_max = 3.0 * expected_dt
    if last_sim_time is None or current_sim_time is None:
        return expected_dt
    dt_raw = current_sim_time - last_sim_time
    if dt_raw <= 0.0:
        return expected_dt
    return max(dt_min, min(dt_max, dt_raw))


def extract_sync_metrics(state: dict[str, Any], *, receive_time_ns: int | None = None) -> dict[str, float]:
    metrics: dict[str, float] = {
        "sim_time_seconds": float("nan"),
        "realtime_factor": float("nan"),
        "packet_age_ms": float("nan"),
        "control_period_seconds": float("nan"),
        "sim_wall_skew_seconds": float("nan"),
        "session_wall_elapsed_seconds": float("nan"),
    }
    sim_t = sim_time_seconds(state)
    if sim_t is not None:
        metrics["sim_time_seconds"] = sim_t
    if "realtime_factor" in state:
        metrics["realtime_factor"] = float(state["realtime_factor"])
    wall_time_send_ns = state.get("wall_time_send_ns")
    if receive_time_ns is not None and wall_time_send_ns is not None:
        metrics["packet_age_ms"] = max(0.0, (int(receive_time_ns) - int(wall_time_send_ns)) / 1.0e6)
    timing = state.get("timing")
    if isinstance(timing, dict):
        for field_name in (
            "control_period_seconds",
            "sim_wall_skew_seconds",
            "session_wall_elapsed_seconds",
            "realtime_factor",
        ):
            if field_name in timing:
                metrics[field_name] = float(timing[field_name])
    return metrics


@dataclass
class RealtimeTracker:
    realtime_factor: float = 1.0
    _window_wall: float = 0.0
    _window_sim: float = 0.0

    def update(self, elapsed_wall: float, timestep: float) -> float:
        self._window_wall += elapsed_wall
        self._window_sim += timestep
        if self._window_wall >= 0.5:
            self.realtime_factor = self._window_sim / self._window_wall
            self._window_wall = 0.0
            self._window_sim = 0.0
        return self.realtime_factor


@dataclass
class SessionTimingTracker:
    """Tracks sim-time vs wall-clock skew embedded in every published state packet."""

    wall_origin_monotonic: float
    sim_origin: float
    control_period_seconds: float
    publish_every_n_steps: int
    pacing_mode: PacingMode = "realtime"

    @classmethod
    def start(
        cls,
        sim_time: float,
        timing: SimulationTimingConfig | None = None,
        *,
        control_period_seconds: float | None = None,
        publish_every_n_steps: int | None = None,
    ) -> SessionTimingTracker:
        if timing is None:
            if control_period_seconds is None or publish_every_n_steps is None:
                raise TypeError(
                    "SessionTimingTracker.start requires either timing=SimulationTimingConfig "
                    "or control_period_seconds and publish_every_n_steps"
                )
            resolved_control_period = float(control_period_seconds)
            resolved_publish_every = int(publish_every_n_steps)
        else:
            resolved_control_period = timing.control_period_seconds
            resolved_publish_every = timing.state_publish_every_n_steps
        resolved_pacing_mode: PacingMode = "realtime"
        if timing is not None:
            resolved_pacing_mode = timing.pacing_mode
        return cls(
            wall_origin_monotonic=time.monotonic(),
            sim_origin=float(sim_time),
            control_period_seconds=resolved_control_period,
            publish_every_n_steps=resolved_publish_every,
            pacing_mode=resolved_pacing_mode,
        )

    def snapshot(self, sim_time: float, realtime_factor: float) -> dict[str, float | str]:
        wall_elapsed = time.monotonic() - self.wall_origin_monotonic
        sim_elapsed = float(sim_time) - self.sim_origin
        return {
            "physics_timestep_seconds": self.control_period_seconds / float(self.publish_every_n_steps),
            "control_period_seconds": self.control_period_seconds,
            "publish_every_n_steps": float(self.publish_every_n_steps),
            "pacing_mode": self.pacing_mode,
            "session_wall_elapsed_seconds": wall_elapsed,
            "session_sim_elapsed_seconds": sim_elapsed,
            "sim_wall_skew_seconds": sim_elapsed - wall_elapsed,
            "realtime_factor": float(realtime_factor),
        }


@contextmanager
def high_resolution_os_timer() -> Iterator[None]:
    """Raise Windows timer resolution to 1 ms for real-time pacing."""
    if sys.platform != "win32":
        yield
        return
    import ctypes

    winmm = ctypes.WinDLL("winmm")
    winmm.timeBeginPeriod(1)
    try:
        yield
    finally:
        winmm.timeEndPeriod(1)


class StepPacer:
    """Absolute-deadline pacing: long-run sim rate tracks wall clock when CPU keeps up."""

    _SLEEP_MARGIN_SECONDS = 0.0015

    def __init__(self, timestep: float):
        self._timestep = timestep
        self._next_deadline = time.perf_counter() + timestep

    def pace(self) -> None:
        now = time.perf_counter()
        remaining = self._next_deadline - now
        if remaining > self._SLEEP_MARGIN_SECONDS:
            time.sleep(remaining - self._SLEEP_MARGIN_SECONDS)
        elif remaining < -0.1:
            self._next_deadline = now
        self._next_deadline += self._timestep


class NullPacer:
    """No-op pacer for accelerated runs (physics advances as fast as the host allows)."""

    def pace(self) -> None:
        return None


def build_pacer(timing: SimulationTimingConfig) -> StepPacer | NullPacer:
    if timing.pacing_mode == "realtime":
        return StepPacer(timing.physics_timestep_seconds)
    return NullPacer()


@dataclass
class RealtimeLagMonitor:
    warn_interval_seconds: float = 5.0
    rtf_warn_threshold: float = 0.9
    _next_warn_time: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        self._next_warn_time = time.monotonic() + self.warn_interval_seconds

    def check(self, realtime_factor: float) -> None:
        now = time.monotonic()
        if now < self._next_warn_time:
            return
        self._next_warn_time = now + self.warn_interval_seconds
        if realtime_factor < self.rtf_warn_threshold:
            print(
                f"[simulator] warning: running at {realtime_factor:.2f}x real time; "
                "increase simulation.state_publish_every_n_steps, disable "
                "logging_config.include_contact_details, or lower simulation.viewer_fps"
            )
