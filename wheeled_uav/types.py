from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

__all__ = [
    "ActuatorDynamicsConfig",
    "AerodynamicsConfig",
    "FidelityConfig",
    "InitialPoseSpec",
    "LoggingConfig",
    "NetworkFidelityConfig",
    "RotorSpec",
    "SensorFidelityConfig",
    "SensorLayout",
    "SensorNames",
    "SurfaceEvaluator",
    "SurfaceModelSpec",
    "UAVModelSpec",
    "WallEffectConfig",
]


@dataclass(frozen=True)
class NetworkFidelityConfig:
    enabled: bool = False
    state_tx_latency_ms: float = 0.0
    command_rx_latency_ms: float = 0.0
    packet_loss_percent: float = 0.0
    jitter_std_dev_ms: float = 0.0
    stale_command_threshold_ms: float | None = None
    stale_command_policy: Literal["hold-last-command", "zero-thrust", "hover-fallback"] = "hold-last-command"


@dataclass(frozen=True)
class ActuatorDynamicsConfig:
    motor_tau_ms: float = 0.0
    thrust_rate_limit_n_per_s: float | None = None
    omega_rate_limit_rad_per_s: float | None = None


@dataclass(frozen=True)
class SensorFidelityConfig:
    position_noise_std_m: float = 0.0
    velocity_noise_std_m_per_s: float = 0.0
    angular_velocity_noise_std_rad_per_s: float = 0.0
    attitude_noise_std_rad: float = 0.0


@dataclass(frozen=True)
class LoggingConfig:
    log_network_stats: bool = False
    log_actuator_stats: bool = False
    log_sensor_truth: bool = False
    # Per-contact detail records dominate the state payload size and the
    # per-step serialization cost; summaries are always included.
    include_contact_details: bool = True


@dataclass(frozen=True)
class FidelityConfig:
    mode: Literal["baseline", "hil"] = "baseline"
    network: NetworkFidelityConfig = NetworkFidelityConfig()
    actuator_dynamics: ActuatorDynamicsConfig = ActuatorDynamicsConfig()
    sensor_fidelity: SensorFidelityConfig = SensorFidelityConfig()
    logging: LoggingConfig = LoggingConfig()


@dataclass(frozen=True)
class WallEffectConfig:
    """Near-wall aerodynamic interaction along the environment wall normal.

    The force magnitude is a polynomial in the wall-tangential speed s,
        C(s) = coeff_const_n + coeff_linear_n_per_mps * s
               + coeff_quadratic_n_per_mps2 * s^2,
    scaled by exp(-max(0, clearance - reference_clearance_m) / decay_length_m)
    and clipped to [-max_force_n, max_force_n]. Positive values push the
    vehicle away from the wall (a pressing deficit); negative values model
    suction toward the wall.
    """

    enabled: bool = False
    coeff_const_n: float = 0.0
    coeff_linear_n_per_mps: float = 0.0
    coeff_quadratic_n_per_mps2: float = 0.0
    reference_clearance_m: float = 0.2
    decay_length_m: float = 0.15
    max_force_n: float = 0.0


@dataclass(frozen=True)
class AerodynamicsConfig:
    """Config-gated aerodynamic effect models (all disabled by default)."""

    enabled: bool = False
    wall_effect: WallEffectConfig = WallEffectConfig()


@dataclass(frozen=True)
class InitialPoseSpec:
    position: list[float]
    quaternion: tuple[float, float, float, float] | None = None


@dataclass(frozen=True)
class RotorSpec:
    suffix: str
    position: tuple[float, float, float]
    thrust_axis: tuple[float, float, float]
    yaw_moment_ratio: float
    spin_sign: float


@dataclass(frozen=True)
class SensorLayout:
    position: slice
    linear_velocity: slice
    angular_velocity: slice
    x_axis: slice
    y_axis: slice
    z_axis: slice


@dataclass(frozen=True)
class SensorNames:
    position: str
    linear_velocity: str
    angular_velocity: str
    x_axis: str
    y_axis: str
    z_axis: str


@dataclass(frozen=True)
class SurfaceEvaluator:
    kind: str
    parameters: dict[str, float]


@dataclass(frozen=True)
class SurfaceModelSpec:
    config: dict[str, Any]
    asset_block: str
    geom_block: str
    evaluator: SurfaceEvaluator | None


@dataclass(frozen=True)
class UAVModelSpec:
    name: str
    body_name: str
    actuator_names: tuple[str, str, str, str]
    sensor_names: SensorNames
    contact_prefix: str
