"""MuJoCo model generation: rotor/vehicle XML, terrain, and initial poses.

This subpackage turns ``vehicle_params.json`` plus the XML template into a
concrete MuJoCo model file. It has no runtime or networking dependencies.
"""

from .builder import (
    build_allocation_matrix,
    build_rotor_specs,
    build_sensor_names,
    build_uav_model_specs,
    build_xml_replacements,
    render_model_xml,
)
from .poses import build_initial_poses, build_surface_model_spec
from .surface import (
    build_surface_blocks,
    build_surface_evaluator,
    evaluate_height_function,
    evaluate_surface_properties,
    get_surface_config,
)

__all__ = [
    "build_allocation_matrix",
    "build_initial_poses",
    "build_rotor_specs",
    "build_sensor_names",
    "build_surface_blocks",
    "build_surface_evaluator",
    "build_surface_model_spec",
    "build_uav_model_specs",
    "build_xml_replacements",
    "evaluate_height_function",
    "evaluate_surface_properties",
    "get_surface_config",
    "render_model_xml",
]
