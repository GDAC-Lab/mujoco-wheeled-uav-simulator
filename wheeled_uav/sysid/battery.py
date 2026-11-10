"""Identify the battery from logged voltage and current (and the rotor thrusts, for the power model).

``V = V_oc0 - R I - k Q`` with ``Q`` the charge drawn so far [Ah] (linear
least squares), and, given the rotor thrusts, ``V I = c sum(T_i^1.5) + P_aux``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from ..flightlog import FlightLog

__all__ = ["BatteryFit", "fit_battery"]


@dataclass(frozen=True)
class BatteryFit:
    internal_resistance_ohm: float
    open_circuit_voltage_start_v: float
    open_circuit_slope_v_per_ah: float
    power_coefficient_w_per_n1p5: float | None
    auxiliary_power_w: float | None
    rms_v: float
    samples: int

    def as_dict(self) -> dict:
        return asdict(self)

    def plant_battery(self, capacity_mah: float | None = None) -> dict[str, float]:
        """plant.battery keys this fit determines (the voltages need the capacity to place full and empty)."""
        keys: dict[str, float] = {"internal_resistance_ohm": round(self.internal_resistance_ohm, 6)}
        if self.power_coefficient_w_per_n1p5 is not None:
            keys["power_coefficient_w_per_n1p5"] = round(self.power_coefficient_w_per_n1p5, 4)
            keys["auxiliary_power_w"] = round(max(0.0, self.auxiliary_power_w or 0.0), 4)
        return keys


def fit_battery(log: FlightLog, *, rotor_thrusts: np.ndarray | None = None) -> BatteryFit:
    if not (log.has("battery_voltage") and log.has("battery_current")):
        raise ValueError("fit_battery needs the battery_voltage and battery_current channels")
    voltage = log["battery_voltage"][:, 0]
    current = log["battery_current"][:, 0]
    used = log.valid_mask() & np.isfinite(voltage) & np.isfinite(current)
    if np.count_nonzero(used) < 10:
        raise ValueError("fit_battery: fewer than 10 usable samples")
    # Charge drawn so far [Ah], by the trapezoidal rule.
    charge = np.concatenate([[0.0], np.cumsum(0.5 * (current[1:] + current[:-1]) * np.diff(log.time))]) / 3600.0
    design = np.column_stack([np.ones(np.count_nonzero(used)), -current[used], -charge[used]])
    solution, *_ = np.linalg.lstsq(design, voltage[used], rcond=None)
    residual = voltage[used] - design @ solution
    power_coefficient = auxiliary = None
    if rotor_thrusts is not None:
        thrusts = np.maximum(np.asarray(rotor_thrusts, dtype=float), 0.0)
        power = (voltage * current)[used]
        power_design = np.column_stack([np.sum(thrusts[used] ** 1.5, axis=1), np.ones(np.count_nonzero(used))])
        (power_coefficient, auxiliary), *_ = np.linalg.lstsq(power_design, power, rcond=None)
        power_coefficient, auxiliary = float(power_coefficient), float(auxiliary)
    return BatteryFit(
        internal_resistance_ohm=float(solution[1]),
        open_circuit_voltage_start_v=float(solution[0]),
        open_circuit_slope_v_per_ah=float(solution[2]),
        power_coefficient_w_per_n1p5=power_coefficient,
        auxiliary_power_w=auxiliary,
        rms_v=float(np.sqrt(np.mean(residual**2))),
        samples=int(np.count_nonzero(used)),
    )
