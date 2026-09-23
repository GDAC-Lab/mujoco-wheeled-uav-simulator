from __future__ import annotations

import copy
import json
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any

from .calibration import apply_calibration_file
from .paths import DEFAULT_PATH_RESOLVER, PathResolver
from .types import (
    ActuatorDynamicsConfig,
    AerodynamicsConfig,
    FidelityConfig,
    LoggingConfig,
    NetworkFidelityConfig,
    SensorFidelityConfig,
    WallEffectConfig,
)

__all__ = [
    "InertialReferenceWarning",
    "build_aerodynamics_config",
    "build_fidelity_config",
    "clear_vehicle_params_cache",
    "get_configured_drone_mass",
    "get_inertial_reference",
    "load_vehicle_params",
    "parse_total_mass",
]

INERTIAL_REFERENCES = ("body_only", "total_vehicle")


class InertialReferenceWarning(UserWarning):
    """drone.inertial_reference is missing, so the "body_only" default applies."""


def get_configured_drone_mass(drone_params: dict[str, Any]) -> float:
    # drone.mass; drone.body_box.mass is the legacy location, read only when
    # drone.mass is absent, so a config that sets drone.mass needs no body_box.mass.
    if "mass" in drone_params:
        return float(drone_params["mass"])
    return float(drone_params["body_box"]["mass"])


def get_inertial_reference(drone_params: dict[str, Any]) -> str:
    # "body_only": drone.mass / drone.inertia describe the central body and the two
    # wheels are added on top. "total_vehicle": they describe the whole vehicle and
    # the analytic wheel contributions are subtracted when the MuJoCo body is built.
    # A missing key keeps the "body_only" default but warns, because reading a
    # whole-vehicle measurement (a scale weight, a CAD assembly) as "body_only"
    # silently counts the wheel mass twice.
    raw_value = drone_params.get("inertial_reference")
    if raw_value is None:
        body_mass = get_configured_drone_mass(drone_params)
        wheel_mass = float(drone_params["wheels"]["mass"])
        # stacklevel=1 attributes the warning to this line, so Python's default
        # filter prints it once per process instead of once per call site
        # (the builder asks twice per UAV, the hover controller once more).
        warnings.warn(
            'drone.inertial_reference is not set; assuming "body_only": the model gets '
            f"drone.mass {body_mass:g} kg plus two wheels of {wheel_mass:g} kg = "
            f"{body_mass + 2.0 * wheel_mass:g} kg in total. If drone.mass and drone.inertia "
            'describe the whole vehicle, set "total_vehicle"; otherwise set "body_only" '
            "explicitly to silence this warning.",
            InertialReferenceWarning,
            stacklevel=1,
        )
        return "body_only"
    inertial_reference = str(raw_value).strip().lower()
    if inertial_reference not in INERTIAL_REFERENCES:
        raise ValueError('drone.inertial_reference must be "body_only" or "total_vehicle"')
    return inertial_reference


def parse_total_mass(params: dict[str, Any]) -> float:
    # drone.inertial_reference "total_vehicle": drone.mass already includes the wheels.
    # "body_only" (the default, with a warning, when the key is missing):
    # total = body mass + 2 * wheel mass.
    drone_params = params["drone"]
    inertial_reference = get_inertial_reference(drone_params)
    body_or_total_mass = get_configured_drone_mass(drone_params)
    if inertial_reference == "total_vehicle":
        return body_or_total_mass
    return body_or_total_mass + 2.0 * float(drone_params["wheels"]["mass"])


@lru_cache(maxsize=8)
def _load_vehicle_params_cached(resolved_params_path: str) -> dict[str, Any]:
    with Path(resolved_params_path).open("r", encoding="utf-8") as params_file:
        return json.load(params_file)


def clear_vehicle_params_cache() -> None:
    _load_vehicle_params_cached.cache_clear()


def load_vehicle_params(
    params_path: str | Path | None = None,
    *,
    path_resolver: PathResolver | None = None,
) -> dict[str, Any]:
    resolved_params_path = (path_resolver or DEFAULT_PATH_RESOLVER).get_params_path(params_path)
    # Deep copy so callers can never mutate the cached dict: every consumer in
    # the process aliases the same parse otherwise, which silently corrupts
    # later loads (and cross-contaminates tests).
    params = copy.deepcopy(_load_vehicle_params_cached(str(resolved_params_path)))
    # Overlay measured propulsion calibration (actuation.calibration_file), so
    # every consumer of the params dict sees the bench-test values.
    return apply_calibration_file(params, Path(str(resolved_params_path)))


def _get_bool(config_section: dict[str, Any], field_name: str, default_value: bool) -> bool:
    raw_value = config_section.get(field_name, default_value)
    return bool(raw_value)


def _get_float(config_section: dict[str, Any], field_name: str, default_value: float) -> float:
    raw_value = config_section.get(field_name, default_value)
    return float(raw_value)


def _get_optional_float(config_section: dict[str, Any], field_name: str) -> float | None:
    raw_value = config_section.get(field_name)
    if raw_value is None:
        return None
    return float(raw_value)


def build_aerodynamics_config(params: dict[str, Any]) -> AerodynamicsConfig:
    """Parse the optional top-level "aerodynamics" section (disabled by default).

    Aerodynamic effects are part of the plant physics, so they apply in every
    fidelity mode; they are gated purely by this configuration section.
    """
    aero_section = params.get("aerodynamics", {})
    if not isinstance(aero_section, dict):
        aero_section = {}

    wall_section = aero_section.get("wall_effect", {})
    if not isinstance(wall_section, dict):
        wall_section = {}

    decay_length = _get_float(wall_section, "decay_length_m", 0.15)
    if decay_length <= 0.0:
        raise ValueError("aerodynamics.wall_effect.decay_length_m must be positive")

    return AerodynamicsConfig(
        enabled=_get_bool(aero_section, "enabled", False),
        wall_effect=WallEffectConfig(
            enabled=_get_bool(wall_section, "enabled", False),
            coeff_const_n=_get_float(wall_section, "coeff_const_n", 0.0),
            coeff_linear_n_per_mps=_get_float(wall_section, "coeff_linear_n_per_mps", 0.0),
            coeff_quadratic_n_per_mps2=_get_float(wall_section, "coeff_quadratic_n_per_mps2", 0.0),
            reference_clearance_m=_get_float(wall_section, "reference_clearance_m", 0.2),
            decay_length_m=decay_length,
            max_force_n=_get_float(wall_section, "max_force_n", 0.0),
        ),
    )


def build_fidelity_config(params: dict[str, Any]) -> FidelityConfig:
    raw_fidelity_mode = params.get("fidelity_mode", "baseline")
    if isinstance(raw_fidelity_mode, dict):
        fidelity_mode = str(raw_fidelity_mode.get("mode", raw_fidelity_mode.get("baseline", "baseline")))
    else:
        fidelity_mode = str(raw_fidelity_mode)
    if fidelity_mode not in {"baseline", "hil"}:
        raise ValueError(f"Unsupported fidelity mode: {fidelity_mode}")

    network_section = params.get("network_fidelity", {})
    if not isinstance(network_section, dict):
        network_section = {}

    actuator_section = params.get("actuator_dynamics", {})
    if not isinstance(actuator_section, dict):
        actuator_section = {}

    sensor_section = params.get("sensor_fidelity", {})
    if not isinstance(sensor_section, dict):
        sensor_section = {}

    logging_section = params.get("logging_config", {})
    if not isinstance(logging_section, dict):
        logging_section = {}

    stale_policy = str(network_section.get("stale_command_policy", "hold-last-command"))
    if stale_policy not in {"hold-last-command", "zero-thrust", "hover-fallback"}:
        raise ValueError(f"Unsupported stale command policy: {stale_policy}")

    return FidelityConfig(
        mode=fidelity_mode,
        network=NetworkFidelityConfig(
            enabled=_get_bool(network_section, "enabled", False),
            state_tx_latency_ms=_get_float(network_section, "state_tx_latency_ms", 0.0),
            command_rx_latency_ms=_get_float(network_section, "command_rx_latency_ms", 0.0),
            packet_loss_percent=_get_float(network_section, "packet_loss_percent", 0.0),
            jitter_std_dev_ms=_get_float(network_section, "jitter_std_dev_ms", 0.0),
            stale_command_threshold_ms=_get_optional_float(network_section, "stale_command_threshold_ms"),
            stale_command_policy=stale_policy,
        ),
        actuator_dynamics=ActuatorDynamicsConfig(
            motor_tau_ms=_get_float(actuator_section, "motor_tau_ms", 0.0),
            thrust_rate_limit_n_per_s=_get_optional_float(actuator_section, "thrust_rate_limit_n_per_s"),
            omega_rate_limit_rad_per_s=_get_optional_float(actuator_section, "omega_rate_limit_rad_per_s"),
        ),
        sensor_fidelity=SensorFidelityConfig(
            position_noise_std_m=_get_float(sensor_section, "position_noise_std_m", 0.0),
            velocity_noise_std_m_per_s=_get_float(sensor_section, "velocity_noise_std_m_per_s", 0.0),
            angular_velocity_noise_std_rad_per_s=_get_float(sensor_section, "angular_velocity_noise_std_rad_per_s", 0.0),
            attitude_noise_std_rad=_get_float(sensor_section, "attitude_noise_std_rad", 0.0),
        ),
        logging=LoggingConfig(
            log_network_stats=_get_bool(logging_section, "log_network_stats", False),
            log_actuator_stats=_get_bool(logging_section, "log_actuator_stats", False),
            log_sensor_truth=_get_bool(logging_section, "log_sensor_truth", False),
            include_contact_details=_get_bool(logging_section, "include_contact_details", True),
        ),
    )
