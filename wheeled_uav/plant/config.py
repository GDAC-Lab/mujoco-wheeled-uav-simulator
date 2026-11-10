"""The ``plant`` section of vehicle_params.json: parsed, validated, typed.

Every subsection is optional and off unless present; a run without a
``plant`` section behaves exactly as before the section existed. Unknown keys
are errors (a misspelt key would otherwise silently leave an effect at its
default); keys starting with an underscore are comments and are ignored, as
is ``"enabled": false``, which turns a subsection off without deleting it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "ActuationConfig",
    "BatteryConfig",
    "CorrectionConfig",
    "GroundEffectConfig",
    "PlantAerodynamicsConfig",
    "PlantConfig",
    "PlantSensorConfig",
    "parse_plant_config",
]


@dataclass(frozen=True)
class ActuationConfig:
    """Command delay, rotor-speed lag, thrust scale and saturation of every rotor.

    The thrust command of each rotor is delayed by ``delay_s`` (rounded to
    physics steps), scaled by ``thrust_scale`` (and ``rotor_thrust_scales``),
    limited to [``idle_thrust_n``, ``max_thrust_n``] (a positive command
    never gives less than idle; a zero command stops the rotor), and the
    rotor speed follows the resulting steady-state speed as a first-order
    lag with ``motor_time_constant_up_s`` when speeding up and
    ``motor_time_constant_down_s`` when slowing down. Thrust is
    ``k_f * omega**2``.
    """

    delay_s: float = 0.0
    motor_time_constant_up_s: float = 0.0
    motor_time_constant_down_s: float = 0.0
    thrust_scale: float = 1.0
    rotor_thrust_scales: tuple[float, ...] | None = None
    max_thrust_n: float | None = None
    idle_thrust_n: float = 0.0


@dataclass(frozen=True)
class BatteryConfig:
    """One battery pack per vehicle: open-circuit voltage, internal resistance, charge.

    Open-circuit voltage is linear in the state of charge between
    ``empty_voltage_v`` and ``full_voltage_v``. The electrical power is
    ``power_coefficient_w_per_n1p5 * sum(T_i ** 1.5) + auxiliary_power_w``
    (momentum theory; the coefficient folds in figure of merit and motor and
    ESC efficiency), and the terminal voltage solves ``P = V (V_oc - V) / R``.
    The maximum thrust scales with ``(V / reference_voltage_v) ** 2``; with
    ``voltage_compensation`` false the thrust command is read as calibrated at
    the reference voltage and the delivered thrust scales the same way (a
    flight controller that does not compensate for the battery voltage).
    """

    full_voltage_v: float = 16.8
    empty_voltage_v: float = 13.6
    initial_charge: float = 1.0
    capacity_mah: float = 1500.0
    internal_resistance_ohm: float = 0.03
    power_coefficient_w_per_n1p5: float = 12.0
    auxiliary_power_w: float = 0.0
    reference_voltage_v: float = 16.0
    voltage_compensation: bool = True


@dataclass(frozen=True)
class GroundEffectConfig:
    """Cheeseman-Bennett ground effect on each rotor: ``T / T_inf = 1 / (1 - (R / 4z)^2)``, capped at ``max_gain``."""

    rotor_radius_m: float | None = None
    max_gain: float = 1.5
    ground_height_m: float | None = None


@dataclass(frozen=True)
class PlantAerodynamicsConfig:
    """Drag on the vehicle body, relative to the air (``wind_m_per_s``).

    Rotor drag is linear in the body-frame airspeed,
    ``F_b = -diag(rotor_drag_coefficients_n_s_per_m) v_b`` (the rotor drag of
    Faessler et al., RA-L 2018); body drag is quadratic,
    ``F_b = -0.5 rho diag(body_drag_area_m2) |v_b| v_b`` per body axis.
    """

    air_density_kg_per_m3: float = 1.204
    wind_m_per_s: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotor_drag_coefficients_n_s_per_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    body_drag_area_m2: tuple[float, float, float] = (0.0, 0.0, 0.0)
    ground_effect: GroundEffectConfig | None = None


@dataclass(frozen=True)
class CorrectionConfig:
    """A data-fitted correction force/torque (see wheeled_uav.plant.correction), scaled by ``scale``."""

    path: Path
    scale: float = 1.0


@dataclass(frozen=True)
class PlantSensorConfig:
    """What the controller receives instead of the true state.

    Each channel is the true value from ``*_delay_s`` ago (rounded to physics
    steps), plus white noise. The gyro (angular velocity, body frame) also
    carries a bias: an initial draw plus a random walk. A pose dropout (the
    motion-capture link failing) starts with ``pose_dropout_probability`` at
    each published state and lasts an exponentially distributed time of mean
    ``pose_dropout_mean_duration_s``; during it the position, attitude and
    velocity hold their last delivered values.
    """

    pose_delay_s: float = 0.0
    velocity_delay_s: float = 0.0
    angular_velocity_delay_s: float = 0.0
    position_noise_std_m: float = 0.0
    attitude_noise_std_rad: float = 0.0
    velocity_noise_std_m_per_s: float = 0.0
    angular_velocity_noise_std_rad_per_s: float = 0.0
    angular_velocity_bias_std_rad_per_s: float = 0.0
    angular_velocity_bias_walk_rad_per_s_per_sqrt_s: float = 0.0
    pose_dropout_probability: float = 0.0
    pose_dropout_mean_duration_s: float = 0.05

    @property
    def max_delay_s(self) -> float:
        return max(self.pose_delay_s, self.velocity_delay_s, self.angular_velocity_delay_s)


@dataclass(frozen=True)
class PlantConfig:
    actuation: ActuationConfig | None = None
    battery: BatteryConfig | None = None
    aerodynamics: PlantAerodynamicsConfig | None = None
    correction: CorrectionConfig | None = None
    sensors: PlantSensorConfig | None = None

    @property
    def empty(self) -> bool:
        return all(part is None for part in (self.actuation, self.battery, self.aerodynamics, self.correction, self.sensors))


def _section(parent: dict[str, Any], key: str, where: str) -> dict[str, Any] | None:
    """The dict at parent[key] with comments dropped, or None when absent or disabled."""
    if key not in parent or parent[key] is None:
        return None
    section = parent[key]
    if not isinstance(section, dict):
        raise ValueError(f"{where}.{key} must be an object")
    enabled = section.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ValueError(f"{where}.{key}.enabled must be true or false")
    if not enabled:
        return None
    return {name: value for name, value in section.items() if not name.startswith("_") and name != "enabled"}


def _check_keys(section: dict[str, Any], allowed: tuple[str, ...], where: str) -> None:
    unknown = sorted(set(section) - set(allowed))
    if unknown:
        raise ValueError(f"unknown key(s) in {where}: {', '.join(unknown)}; allowed: {', '.join(allowed)}")


def _number(section: dict[str, Any], key: str, default: float, where: str, *, minimum: float | None = 0.0, strictly_positive: bool = False) -> float:
    raw = section.get(key, default)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(float(raw)):
        raise ValueError(f"{where}.{key} must be a finite number, got {raw!r}")
    value = float(raw)
    if strictly_positive and value <= 0.0:
        raise ValueError(f"{where}.{key} must be positive, got {value!r}")
    if minimum is not None and value < minimum:
        raise ValueError(f"{where}.{key} must be at least {minimum:g}, got {value!r}")
    return value


def _optional_number(section: dict[str, Any], key: str, where: str, **limits: Any) -> float | None:
    if section.get(key) is None:
        return None
    return _number(section, key, 0.0, where, **limits)


def _vector(section: dict[str, Any], key: str, default: tuple[float, ...], where: str, *, length: int, minimum: float | None = None) -> tuple[float, ...]:
    raw = section.get(key, default)
    if not isinstance(raw, (list, tuple)) or len(raw) != length:
        raise ValueError(f"{where}.{key} must be a list of {length} numbers, got {raw!r}")
    values = []
    for item in raw:
        if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(float(item)):
            raise ValueError(f"{where}.{key} must be a list of {length} finite numbers, got {raw!r}")
        if minimum is not None and float(item) < minimum:
            raise ValueError(f"{where}.{key} entries must be at least {minimum:g}, got {raw!r}")
        values.append(float(item))
    return tuple(values)


def _parse_actuation(section: dict[str, Any], rotors_per_uav: int) -> ActuationConfig:
    where = "plant.actuation"
    _check_keys(
        section,
        ("delay_s", "motor_time_constant_up_s", "motor_time_constant_down_s", "thrust_scale", "rotor_thrust_scales", "max_thrust_n", "idle_thrust_n"),
        where,
    )
    tau_up = _number(section, "motor_time_constant_up_s", 0.0, where)
    tau_down = _optional_number(section, "motor_time_constant_down_s", where)
    rotor_scales = None
    if section.get("rotor_thrust_scales") is not None:
        rotor_scales = _vector(section, "rotor_thrust_scales", (), where, length=rotors_per_uav, minimum=0.0)
    config = ActuationConfig(
        delay_s=_number(section, "delay_s", 0.0, where),
        motor_time_constant_up_s=tau_up,
        motor_time_constant_down_s=tau_up if tau_down is None else tau_down,
        thrust_scale=_number(section, "thrust_scale", 1.0, where, strictly_positive=True),
        rotor_thrust_scales=rotor_scales,
        max_thrust_n=_optional_number(section, "max_thrust_n", where, strictly_positive=True),
        idle_thrust_n=_number(section, "idle_thrust_n", 0.0, where),
    )
    if config.max_thrust_n is not None and config.idle_thrust_n > config.max_thrust_n:
        raise ValueError("plant.actuation.idle_thrust_n must not exceed max_thrust_n")
    return config


def _parse_battery(section: dict[str, Any]) -> BatteryConfig:
    where = "plant.battery"
    _check_keys(
        section,
        (
            "full_voltage_v", "empty_voltage_v", "initial_charge", "capacity_mah", "internal_resistance_ohm",
            "power_coefficient_w_per_n1p5", "auxiliary_power_w", "reference_voltage_v", "voltage_compensation",
        ),
        where,
    )
    compensation = section.get("voltage_compensation", True)
    if not isinstance(compensation, bool):
        raise ValueError(f"{where}.voltage_compensation must be true or false")
    defaults = BatteryConfig()
    config = BatteryConfig(
        full_voltage_v=_number(section, "full_voltage_v", defaults.full_voltage_v, where, strictly_positive=True),
        empty_voltage_v=_number(section, "empty_voltage_v", defaults.empty_voltage_v, where, strictly_positive=True),
        initial_charge=_number(section, "initial_charge", defaults.initial_charge, where),
        capacity_mah=_number(section, "capacity_mah", defaults.capacity_mah, where, strictly_positive=True),
        internal_resistance_ohm=_number(section, "internal_resistance_ohm", defaults.internal_resistance_ohm, where),
        power_coefficient_w_per_n1p5=_number(section, "power_coefficient_w_per_n1p5", defaults.power_coefficient_w_per_n1p5, where),
        auxiliary_power_w=_number(section, "auxiliary_power_w", defaults.auxiliary_power_w, where),
        reference_voltage_v=_number(section, "reference_voltage_v", defaults.reference_voltage_v, where, strictly_positive=True),
        voltage_compensation=compensation,
    )
    if config.empty_voltage_v >= config.full_voltage_v:
        raise ValueError("plant.battery.empty_voltage_v must be below full_voltage_v")
    if config.initial_charge > 1.0:
        raise ValueError("plant.battery.initial_charge is a state of charge in [0, 1]")
    return config


def _parse_aerodynamics(section: dict[str, Any]) -> PlantAerodynamicsConfig:
    where = "plant.aerodynamics"
    _check_keys(
        section,
        ("air_density_kg_per_m3", "wind_m_per_s", "rotor_drag_coefficients_n_s_per_m", "body_drag_area_m2", "ground_effect"),
        where,
    )
    ground_effect = None
    ground_section = _section(section, "ground_effect", where)
    if ground_section is not None:
        ground_where = f"{where}.ground_effect"
        _check_keys(ground_section, ("rotor_radius_m", "max_gain", "ground_height_m"), ground_where)
        ground_effect = GroundEffectConfig(
            rotor_radius_m=_optional_number(ground_section, "rotor_radius_m", ground_where, strictly_positive=True),
            max_gain=_number(ground_section, "max_gain", 1.5, ground_where, minimum=1.0),
            ground_height_m=_optional_number(ground_section, "ground_height_m", ground_where, minimum=None),
        )
    return PlantAerodynamicsConfig(
        air_density_kg_per_m3=_number(section, "air_density_kg_per_m3", 1.204, where),
        wind_m_per_s=_vector(section, "wind_m_per_s", (0.0, 0.0, 0.0), where, length=3),
        rotor_drag_coefficients_n_s_per_m=_vector(section, "rotor_drag_coefficients_n_s_per_m", (0.0, 0.0, 0.0), where, length=3, minimum=0.0),
        body_drag_area_m2=_vector(section, "body_drag_area_m2", (0.0, 0.0, 0.0), where, length=3, minimum=0.0),
        ground_effect=ground_effect,
    )


def _parse_correction(section: dict[str, Any], params_dir: Path | None) -> CorrectionConfig:
    where = "plant.correction"
    _check_keys(section, ("file", "scale"), where)
    raw_path = section.get("file")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"{where}.file must name a correction JSON file")
    path = Path(raw_path)
    if not path.is_absolute() and params_dir is not None:
        path = params_dir / path
    return CorrectionConfig(path=path, scale=_number(section, "scale", 1.0, where, minimum=None))


def _parse_sensors(section: dict[str, Any]) -> PlantSensorConfig:
    where = "plant.sensors"
    names = tuple(PlantSensorConfig.__dataclass_fields__)
    _check_keys(section, names, where)
    values = {name: _number(section, name, getattr(PlantSensorConfig(), name), where) for name in names}
    if values["pose_dropout_probability"] > 1.0:
        raise ValueError(f"{where}.pose_dropout_probability is a probability in [0, 1]")
    if values["pose_dropout_probability"] > 0.0 and values["pose_dropout_mean_duration_s"] <= 0.0:
        raise ValueError(f"{where}.pose_dropout_mean_duration_s must be positive when dropouts are on")
    return PlantSensorConfig(**values)


def parse_plant_config(params: dict[str, Any], *, params_dir: Path | None = None, rotors_per_uav: int = 4) -> PlantConfig | None:
    """The plant model of ``params``, or None when there is none (the section is absent, empty or disabled)."""
    root = _section(params, "plant", "vehicle_params")
    if root is None:
        return None
    _check_keys(root, ("actuation", "battery", "aerodynamics", "correction", "sensors"), "plant")
    actuation = _section(root, "actuation", "plant")
    battery = _section(root, "battery", "plant")
    aerodynamics = _section(root, "aerodynamics", "plant")
    correction = _section(root, "correction", "plant")
    sensors = _section(root, "sensors", "plant")
    config = PlantConfig(
        actuation=None if actuation is None else _parse_actuation(actuation, rotors_per_uav),
        battery=None if battery is None else _parse_battery(battery),
        aerodynamics=None if aerodynamics is None else _parse_aerodynamics(aerodynamics),
        correction=None if correction is None else _parse_correction(correction, params_dir),
        sensors=None if sensors is None else _parse_sensors(sensors),
    )
    return None if config.empty else config
