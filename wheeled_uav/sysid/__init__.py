"""System identification from flight logs (wheeled_uav.flightlog), for the plant model.

See docs/SYSTEM_IDENTIFICATION.md. Each estimator takes a FlightLog and the
vehicle's mass; ``identify`` runs what a log supports and proposes the
``plant`` section; ``prediction_error`` compares a simulator with a log;
``excitation_flight`` flies the simulator to make logs with known plants.
"""

from .actuation import ActuationFit, fit_actuation, simulate_rotor_thrusts
from .battery import BatteryFit, fit_battery
from .correction import DEFAULT_CORRECTION_FEATURES, correction_dataset, fit_correction
from .drag import DragFit, fit_drag
from .excitation import excitation_flight, record_flight
from .identify import IDENTIFICATION_FORMAT, identify
from .prediction import prediction_error
from .signals import estimate_delay

__all__ = [
    "DEFAULT_CORRECTION_FEATURES",
    "IDENTIFICATION_FORMAT",
    "ActuationFit",
    "BatteryFit",
    "DragFit",
    "correction_dataset",
    "estimate_delay",
    "excitation_flight",
    "fit_actuation",
    "fit_battery",
    "fit_correction",
    "fit_drag",
    "identify",
    "prediction_error",
    "record_flight",
    "simulate_rotor_thrusts",
]
