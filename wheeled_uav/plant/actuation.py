"""Rotor actuation (delay, rotor-speed lag, scale, saturation) and the battery."""

from __future__ import annotations

import math

import numpy as np

from .config import ActuationConfig, BatteryConfig

__all__ = ["BatteryPacks", "RotorActuation"]


class BatteryPacks:
    """One battery pack per vehicle; voltages and currents follow the rotor thrusts.

    The terminal voltage solves ``P = V I`` with ``V = V_oc - R I``:
    ``V = (V_oc + sqrt(V_oc^2 - 4 R P)) / 2``. Past the pack's maximum power
    (negative discriminant) it stays at ``V_oc / 2``, where that maximum is
    delivered, and ``overloaded`` reports it.
    """

    def __init__(self, config: BatteryConfig, num_uavs: int):
        self.config = config
        self.charge = np.full(num_uavs, float(config.initial_charge))
        self.voltage = self.open_circuit_voltage()
        self.current = np.zeros(num_uavs)
        self.power = np.zeros(num_uavs)
        self.overloaded = np.zeros(num_uavs, dtype=bool)

    def open_circuit_voltage(self) -> np.ndarray:
        config = self.config
        return config.empty_voltage_v + (config.full_voltage_v - config.empty_voltage_v) * self.charge

    def voltage_ratio(self) -> np.ndarray:
        """Terminal voltage over the reference voltage, per vehicle."""
        return self.voltage / self.config.reference_voltage_v

    def evaluate(self, rotor_thrusts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Power, open-circuit voltage, overload flag and terminal voltage under ``rotor_thrusts``
        (one row per vehicle) at the present charge; changes nothing."""
        config = self.config
        thrusts = np.maximum(np.asarray(rotor_thrusts, dtype=float), 0.0)
        power = config.power_coefficient_w_per_n1p5 * np.sum(thrusts**1.5, axis=1) + config.auxiliary_power_w
        open_circuit = self.open_circuit_voltage()
        discriminant = open_circuit**2 - 4.0 * config.internal_resistance_ohm * power
        return power, open_circuit, discriminant < 0.0, 0.5 * (open_circuit + np.sqrt(np.maximum(discriminant, 0.0)))

    def terminal_voltage(self, rotor_thrusts: np.ndarray) -> np.ndarray:
        """The terminal voltage under ``rotor_thrusts`` at the present charge; changes nothing."""
        return self.evaluate(rotor_thrusts)[3]

    def update(self, rotor_thrusts: np.ndarray, timestep: float, evaluated=None) -> None:
        """Advance one physics step; ``rotor_thrusts`` has one row per vehicle (``evaluated``: its evaluate(), if at hand)."""
        config = self.config
        self.power, open_circuit, self.overloaded, self.voltage = self.evaluate(rotor_thrusts) if evaluated is None else evaluated
        # Past its maximum power the pack delivers that maximum: I = V_oc / (2 R) at V = V_oc / 2.
        self.current = np.where(
            self.overloaded,
            (open_circuit - self.voltage) / max(config.internal_resistance_ohm, 1.0e-12),
            self.power / self.voltage,
        )
        # 1 mAh = 3.6 C.
        self.charge = np.maximum(0.0, self.charge - self.current * timestep / (config.capacity_mah * 3.6))


class RotorActuation:
    """The thrust each rotor delivers, from the commanded thrust.

    Works in units of sqrt(thrust): the rotor speed is proportional to it
    (``T = k_f omega^2``), so the first-order lag on the rotor speed is exact
    whatever ``k_f`` is; ``k_f`` is only needed to report speeds in rad/s.
    The lag uses the exact discretization ``s += (1 - exp(-dt / tau)) (s* - s)``.
    """

    def __init__(
        self,
        config: ActuationConfig,
        *,
        timestep: float,
        num_uavs: int,
        rotors_per_uav: int,
        ctrl_limit: float,
        thrust_coefficient: float,
        voltage_compensation: bool = True,
    ):
        self.config = config
        self._compensated = bool(voltage_compensation)
        self._timestep = float(timestep)
        self._num_uavs = num_uavs
        self._rotors_per_uav = rotors_per_uav
        size = num_uavs * rotors_per_uav
        self.delay_steps = int(round(config.delay_s / self._timestep))
        self._alpha_up = self._alpha(config.motor_time_constant_up_s)
        self._alpha_down = self._alpha(config.motor_time_constant_down_s)
        scales = np.full(rotors_per_uav, float(config.thrust_scale))
        if config.rotor_thrust_scales is not None:
            scales = scales * np.asarray(config.rotor_thrust_scales, dtype=float)
        self._scale = np.tile(scales, num_uavs)
        self.max_thrust = float(ctrl_limit if config.max_thrust_n is None else config.max_thrust_n)
        self._ctrl_limit = float(ctrl_limit)
        self._thrust_coefficient = float(thrust_coefficient)
        # Ring buffer of the last delay_steps + 1 commands (row `_head` is the newest).
        self._history = np.zeros((self.delay_steps + 1, size))
        self._head = 0
        self._sqrt_thrust = np.zeros(size)
        # The step in progress: the command leaving the delay line and the rotor speeds it started from.
        self._delayed = np.zeros(size)
        self._step_start = np.zeros(size)
        self.requested = np.zeros(size)
        self.delivered = np.zeros(size)
        self.saturated = np.zeros(size, dtype=bool)
        self.initialized = False

    def _alpha(self, time_constant: float) -> float:
        return 1.0 if time_constant <= 0.0 else 1.0 - math.exp(-self._timestep / time_constant)

    def _per_rotor(self, per_uav: np.ndarray | None) -> np.ndarray:
        if per_uav is None:
            return np.ones(self._num_uavs * self._rotors_per_uav)
        return np.repeat(np.asarray(per_uav, dtype=float), self._rotors_per_uav)

    def steady_thrust(self, command: np.ndarray, voltage_ratio: np.ndarray | None = None) -> np.ndarray:
        """The thrust each rotor settles at under ``command`` (no lag, no delay); also updates ``saturated``.

        ``voltage_ratio`` (one per vehicle) is the battery's terminal voltage
        over its reference voltage, or None without a battery.
        """
        ratio_squared = self._per_rotor(voltage_ratio) ** 2
        command = np.asarray(command, dtype=float)
        if voltage_ratio is not None and not self._compensated:
            command = command * ratio_squared
        target = self._scale * command
        maximum = self.max_thrust * ratio_squared
        self.saturated = target > maximum
        target = np.minimum(target, maximum)
        # A positive command never gives less than idle; zero stops the rotor (disarmed).
        target = np.where(command > 0.0, np.maximum(target, self.config.idle_thrust_n), 0.0)
        return np.clip(target, 0.0, self._ctrl_limit)

    def settle(self, command: np.ndarray, voltage_ratio: np.ndarray | None = None) -> None:
        """Start in the steady state of ``command``: the delay line full of it, the rotors at its speed."""
        command = np.asarray(command, dtype=float)
        self._history[:] = command
        self._sqrt_thrust = np.sqrt(self.steady_thrust(command, voltage_ratio))
        self._delayed = command.copy()
        self._step_start = self._sqrt_thrust.copy()
        self.requested = command.copy()
        self.delivered = self._sqrt_thrust**2
        self.initialized = True

    def step(self, command: np.ndarray, voltage_ratio: np.ndarray | None = None) -> np.ndarray:
        """Advance one physics step under ``command``; return the delivered thrust of every rotor."""
        command = np.asarray(command, dtype=float)
        if not self.initialized:
            self.settle(command, voltage_ratio)
        self.requested = command.copy()
        self._head = (self._head + 1) % self._history.shape[0]
        self._history[self._head] = command
        # The oldest row is the command from delay_steps steps ago.
        self._delayed = self._history[(self._head + 1) % self._history.shape[0]].copy()
        self._step_start = self._sqrt_thrust
        return self.deliver(voltage_ratio)

    def depends_on_voltage(self, voltage_ratio: np.ndarray) -> bool:
        """Whether the step in progress gives another thrust at another voltage: always without
        voltage compensation, and with it when a rotor is at its (voltage-dependent) limit."""
        if not self._compensated:
            return True
        return bool(np.any(self._scale * self._delayed > self.max_thrust * self._per_rotor(voltage_ratio) ** 2))

    def deliver(self, voltage_ratio: np.ndarray | None = None) -> np.ndarray:
        """The delivered thrust of the step in progress under ``voltage_ratio``.

        ``step`` calls it once; calling it again with another ratio redoes the
        same step (same delayed command, same starting speeds), which is how
        the plant model finds the voltage consistent with the thrust it allows.
        """
        target = np.sqrt(self.steady_thrust(self._delayed, voltage_ratio))
        alpha = np.where(target > self._step_start, self._alpha_up, self._alpha_down)
        self._sqrt_thrust = self._step_start + alpha * (target - self._step_start)
        self.delivered = self._sqrt_thrust**2
        return self.delivered

    def rotor_speeds(self) -> np.ndarray | None:
        """Rotor speeds [rad/s], or None without a positive actuation.thrust_coefficient."""
        if self._thrust_coefficient <= 0.0:
            return None
        return self._sqrt_thrust / math.sqrt(self._thrust_coefficient)
