"""Run the estimators on a flight log and propose a plant section."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..config import parse_total_mass
from ..flightlog import FlightLog
from ..model.builder import build_rotor_specs
from .actuation import fit_actuation, simulate_rotor_thrusts
from .battery import fit_battery
from .correction import fit_correction
from .drag import fit_drag

__all__ = ["IDENTIFICATION_FORMAT", "identify"]

IDENTIFICATION_FORMAT = "wheeled_uav.identification/1"


def identify(
    log: FlightLog,
    params: dict[str, Any],
    *,
    max_delay_s: float = 0.04,
    smooth_s: float = 0.0,
    quadratic_drag: bool = False,
    correction: str | None = None,
    correction_path: str | Path | None = None,
) -> dict[str, Any]:
    """Fit what the log supports and return the fits and the plant section they imply.

    The actuation needs ``rotor_thrust_command``; the battery needs
    ``battery_voltage`` and ``battery_current``. ``correction`` ("knn" or
    "linear") also fits a correction to what remains and saves it at
    ``correction_path``.
    """
    mass = parse_total_mass(params)
    gravity = abs(float(params["simulation"]["gravity"][2]))
    rotor_axes = np.array([spec.thrust_axis for spec in build_rotor_specs(params)], dtype=float)
    result: dict[str, Any] = {
        "format": IDENTIFICATION_FORMAT,
        "log": {"samples": len(log), "duration_s": float(log.time[-1] - log.time[0]), "source": log.metadata.get("source")},
        "mass_kg": mass,
        "fits": {},
        "plant": {},
        "notes": [],
    }
    plant: dict[str, Any] = result["plant"]
    actuation = None
    if log.has("rotor_thrust_command"):
        actuation = fit_actuation(log, mass=mass, gravity=gravity, max_delay_s=max_delay_s, smooth_s=smooth_s)
        result["fits"]["actuation"] = actuation.as_dict()
        plant["actuation"] = actuation.plant_actuation()
        thrusts = simulate_rotor_thrusts(log, actuation)
        drag = fit_drag(log, mass=mass, gravity=gravity, quadratic=quadratic_drag, thrust_force_body=thrusts @ rotor_axes, smooth_s=smooth_s)
    else:
        thrusts = None
        result["notes"].append("no rotor_thrust_command: actuation not identified, and the drag fit assumes no thrust along body x and y")
        drag = fit_drag(log, mass=mass, gravity=gravity, quadratic=quadratic_drag, smooth_s=smooth_s)
    result["fits"]["drag"] = drag.as_dict()
    aerodynamics: dict[str, Any] = {
        "rotor_drag_coefficients_n_s_per_m": [
            round(drag.rotor_drag_coefficients_n_s_per_m[0], 6),
            round(drag.rotor_drag_coefficients_n_s_per_m[1], 6),
            round(max(0.0, actuation.vertical_drag_n_s_per_m) if actuation is not None else 0.0, 6),
        ]
    }
    if drag.body_drag_area_m2 is not None:
        aerodynamics["body_drag_area_m2"] = [round(drag.body_drag_area_m2[0], 8), round(drag.body_drag_area_m2[1], 8), 0.0]
    plant["aerodynamics"] = aerodynamics
    if log.has("battery_voltage") and log.has("battery_current"):
        battery = fit_battery(log, rotor_thrusts=thrusts)
        result["fits"]["battery"] = battery.as_dict()
        plant["battery"] = battery.plant_battery()
    if correction is not None:
        if actuation is None:
            raise ValueError("a correction needs the actuation fit (rotor_thrust_command)")
        if correction_path is None:
            raise ValueError("give correction_path to save the correction")
        model = fit_correction(
            log, mass=mass, gravity=gravity, actuation=actuation, aerodynamics=aerodynamics, rotor_axes=rotor_axes, kind=correction,
            provenance={"log_source": log.metadata.get("source")},
        )
        model.save(correction_path)
        plant["correction"] = {"file": str(correction_path)}
        result["fits"]["correction"] = model.provenance
    return result
