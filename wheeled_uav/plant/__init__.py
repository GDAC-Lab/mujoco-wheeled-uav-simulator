"""Plant model: actuation, battery, drag and ground effect, data-fitted correction, sensors, randomization.

See docs/PLANT_MODEL.md. Everything here is opt-in through the ``plant`` and
``randomization`` sections of vehicle_params.json.
"""

from .config import (
    ActuationConfig,
    BatteryConfig,
    CorrectionConfig,
    GroundEffectConfig,
    PlantAerodynamicsConfig,
    PlantConfig,
    PlantSensorConfig,
    parse_plant_config,
)
from .correction import FEATURE_NAMES, OUTPUT_NAMES, CorrectionModel, fit_knn_correction, fit_linear_correction
from .model import PlantModel, VehicleKinematics, plant_random_generator
from .randomization import apply_randomization, write_randomization_record
from .sensors import MeasuredState, SensorModel

__all__ = [
    "FEATURE_NAMES",
    "OUTPUT_NAMES",
    "ActuationConfig",
    "BatteryConfig",
    "CorrectionConfig",
    "CorrectionModel",
    "GroundEffectConfig",
    "MeasuredState",
    "PlantAerodynamicsConfig",
    "PlantConfig",
    "PlantModel",
    "PlantSensorConfig",
    "SensorModel",
    "VehicleKinematics",
    "apply_randomization",
    "fit_knn_correction",
    "fit_linear_correction",
    "parse_plant_config",
    "plant_random_generator",
    "write_randomization_record",
]
