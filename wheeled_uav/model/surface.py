"""Terrain surfaces: config normalization, height-function math, XML emission.

Each supported height function is implemented once, with numpy operations that
accept scalars and grids alike, and returns ``(height, dh_dx, dh_dy)``. The
same implementation therefore serves the analytic single-point queries used by
contact reporting and spawn-pose solving, and the vectorized grid sampling used
to build MuJoCo hfield assets.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from ..paths import SURFACE_GEOM_NAME, SURFACE_HFIELD_NAME
from ..types import SurfaceEvaluator
from .xml_format import build_xml_attributes, format_scalar, format_vector

__all__ = [
    "get_surface_config",
    "build_surface_evaluator",
    "surface_is_approximately_flat",
    "surface_can_use_plane_geom",
    "evaluate_height_function",
    "evaluate_height_gradient",
    "evaluate_surface_properties",
    "evaluate_surface_normal",
    "build_surface_blocks",
]

_SURFACE_MODE_ALIASES_PLANE = {"plane", "floor"}
_SURFACE_MODE_ALIASES_HEIGHT_FUNCTION = {"flat", "slope", "paraboloid", "sinusoidal", "gaussian", "composite"}


def _copy_surface_config(surface_config: dict[str, Any]) -> dict[str, Any]:
    copied_config = dict(surface_config)

    plane_config = copied_config.get("plane")
    if isinstance(plane_config, dict):
        copied_config["plane"] = dict(plane_config)

    contact_config = copied_config.get("contact")
    if isinstance(contact_config, dict):
        copied_config["contact"] = dict(contact_config)

    height_function_config = copied_config.get("height_function")
    if isinstance(height_function_config, dict):
        copied_height_function_config = dict(height_function_config)
        parameters = copied_height_function_config.get("parameters")
        if isinstance(parameters, dict):
            copied_height_function_config["parameters"] = dict(parameters)
        copied_config["height_function"] = copied_height_function_config

    return copied_config


def get_surface_config(environment: dict[str, Any]) -> dict[str, Any]:
    if "surface" in environment:
        surface_config = _copy_surface_config(environment["surface"])
        surface_mode = str(surface_config.get("mode", "")).strip().lower()
        if surface_mode:
            if surface_mode in _SURFACE_MODE_ALIASES_PLANE:
                surface_config["type"] = "plane"
            elif surface_mode in _SURFACE_MODE_ALIASES_HEIGHT_FUNCTION:
                surface_config["type"] = "height_function"
                surface_config.setdefault("height_function", {})["name"] = surface_mode
            elif surface_mode != "height_function":
                raise ValueError(f"Unsupported environment.surface.mode: {surface_mode}")
        return surface_config

    return {
        "type": "plane",
        "material": "floor_mat",
        "solref": environment["floor_solref"],
        "contact": environment["floor_contact"],
        "plane": {"size": environment["floor_size"]},
    }


def _build_surface_geom_attributes(surface_config: dict[str, Any]) -> dict[str, str | None]:
    contact = surface_config["contact"]
    attributes: dict[str, str | None] = {
        "name": SURFACE_GEOM_NAME,
        "solref": format_vector(surface_config["solref"]),
        "contype": format_scalar(contact["contype"]),
        "conaffinity": format_scalar(contact["conaffinity"]),
    }
    if "friction" in surface_config and surface_config["friction"] is not None:
        friction_values = [float(value) for value in surface_config["friction"]]
        if len(friction_values) != 3:
            raise ValueError("environment.surface.friction must contain exactly three MuJoCo friction coefficients")
        attributes["friction"] = format_vector(friction_values)
    if surface_config.get("solimp") is not None:
        attributes["solimp"] = format_vector(surface_config["solimp"])
    if surface_config.get("condim") is not None:
        attributes["condim"] = format_scalar(int(surface_config["condim"]))
    if "material" in surface_config:
        attributes["material"] = format_scalar(surface_config["material"])
    if "rgba" in surface_config:
        attributes["rgba"] = format_vector(surface_config["rgba"])
    return attributes


def _get_surface_function_parameters(surface_config: dict[str, Any]) -> dict[str, float]:
    function_config = surface_config["height_function"]
    raw_parameters = function_config.get("parameters", {})
    return {parameter_name: float(parameter_value) for parameter_name, parameter_value in raw_parameters.items()}


def build_surface_evaluator(surface_config: dict[str, Any]) -> SurfaceEvaluator | None:
    if surface_config["type"] != "height_function":
        return None

    function_config = surface_config["height_function"]
    return SurfaceEvaluator(
        kind=str(function_config["name"]),
        parameters=_get_surface_function_parameters(surface_config),
    )


# ---------------------------------------------------------------------------
# Height-function implementations
#
# Each function maps (parameters, x, y) -> (height, dh_dx, dh_dy) and must be
# written with numpy ufuncs so x/y can be python floats or meshgrid arrays.
# ---------------------------------------------------------------------------

def _get_surface_parameter(parameters: dict[str, float], parameter_name: str, default_value: float = 0.0) -> float:
    return float(parameters.get(parameter_name, default_value))


def _validate_gaussian_sigma(sigma_x: float, sigma_y: float) -> None:
    if sigma_x <= 0.0 or sigma_y <= 0.0:
        raise ValueError("gaussian sigma_x and sigma_y must be positive")


def _get_composite_gaussian_terms(parameters: dict[str, float]) -> list[tuple[float, float, float, float]]:
    """Numbered isotropic Gaussian bumps of a "composite" surface.

    Each term k >= 1 is defined by
        gauss<k>_amplitude * exp(-((x - gauss<k>_center_x)^2 + (y - gauss<k>_center_y)^2) / gauss<k>_denominator)
    The denominator convention (instead of sigma) matches height functions
    written as amplitude * exp(-r^2 / c), avoiding sigma-conversion rounding.
    """
    terms: list[tuple[float, float, float, float]] = []
    term_index = 1
    while f"gauss{term_index}_amplitude" in parameters:
        denominator = float(parameters.get(f"gauss{term_index}_denominator", 0.0))
        if denominator <= 0.0:
            raise ValueError(f"composite surface gauss{term_index}_denominator must be positive")
        terms.append(
            (
                float(parameters[f"gauss{term_index}_amplitude"]),
                float(parameters.get(f"gauss{term_index}_center_x", 0.0)),
                float(parameters.get(f"gauss{term_index}_center_y", 0.0)),
                denominator,
            )
        )
        term_index += 1
    return terms


def _flat_terms(parameters: dict[str, float], x: Any, y: Any) -> tuple[Any, Any, Any]:
    z_offset = _get_surface_parameter(parameters, "z_offset")
    zero = np.zeros_like(np.asarray(x, dtype=float))
    return z_offset + zero, zero, zero


def _slope_terms(parameters: dict[str, float], x: Any, y: Any) -> tuple[Any, Any, Any]:
    z_offset = _get_surface_parameter(parameters, "z_offset")
    slope_x = _get_surface_parameter(parameters, "slope_x")
    slope_y = _get_surface_parameter(parameters, "slope_y")
    zero = np.zeros_like(np.asarray(x, dtype=float))
    return z_offset + slope_x * x + slope_y * y, slope_x + zero, slope_y + zero


def _paraboloid_terms(parameters: dict[str, float], x: Any, y: Any) -> tuple[Any, Any, Any]:
    z_offset = _get_surface_parameter(parameters, "z_offset")
    curvature_x = _get_surface_parameter(parameters, "curvature_x")
    curvature_y = _get_surface_parameter(parameters, "curvature_y")
    height = z_offset + curvature_x * np.square(x) + curvature_y * np.square(y)
    return height, 2.0 * curvature_x * np.asarray(x, dtype=float), 2.0 * curvature_y * np.asarray(y, dtype=float)


def _sinusoidal_terms(parameters: dict[str, float], x: Any, y: Any) -> tuple[Any, Any, Any]:
    z_offset = _get_surface_parameter(parameters, "z_offset")
    amplitude = _get_surface_parameter(parameters, "amplitude")
    frequency_x = _get_surface_parameter(parameters, "frequency_x", 1.0)
    frequency_y = _get_surface_parameter(parameters, "frequency_y", 1.0)
    phase_x = _get_surface_parameter(parameters, "phase_x")
    phase_y = _get_surface_parameter(parameters, "phase_y")
    sin_x = np.sin(frequency_x * np.asarray(x, dtype=float) + phase_x)
    cos_x = np.cos(frequency_x * np.asarray(x, dtype=float) + phase_x)
    sin_y = np.sin(frequency_y * np.asarray(y, dtype=float) + phase_y)
    cos_y = np.cos(frequency_y * np.asarray(y, dtype=float) + phase_y)
    height = z_offset + amplitude * sin_x * sin_y
    return height, amplitude * frequency_x * cos_x * sin_y, amplitude * frequency_y * sin_x * cos_y


def _gaussian_terms(parameters: dict[str, float], x: Any, y: Any) -> tuple[Any, Any, Any]:
    z_offset = _get_surface_parameter(parameters, "z_offset")
    amplitude = _get_surface_parameter(parameters, "amplitude")
    center_x = _get_surface_parameter(parameters, "center_x")
    center_y = _get_surface_parameter(parameters, "center_y")
    sigma_x = _get_surface_parameter(parameters, "sigma_x", 1.0)
    sigma_y = _get_surface_parameter(parameters, "sigma_y", 1.0)
    _validate_gaussian_sigma(sigma_x, sigma_y)
    dx = np.asarray(x, dtype=float) - center_x
    dy = np.asarray(y, dtype=float) - center_y
    gaussian_value = amplitude * np.exp(-0.5 * (np.square(dx / sigma_x) + np.square(dy / sigma_y)))
    height = z_offset + gaussian_value
    return height, -gaussian_value * dx / (sigma_x**2), -gaussian_value * dy / (sigma_y**2)


def _composite_terms(parameters: dict[str, float], x: Any, y: Any) -> tuple[Any, Any, Any]:
    z_offset = _get_surface_parameter(parameters, "z_offset")
    slope_x = _get_surface_parameter(parameters, "slope_x")
    slope_y = _get_surface_parameter(parameters, "slope_y")
    zero = np.zeros_like(np.asarray(x, dtype=float))
    height = z_offset + slope_x * x + slope_y * y + zero
    dh_dx = slope_x + zero
    dh_dy = slope_y + zero
    for amplitude, center_x, center_y, denominator in _get_composite_gaussian_terms(parameters):
        dx = np.asarray(x, dtype=float) - center_x
        dy = np.asarray(y, dtype=float) - center_y
        gaussian_value = amplitude * np.exp(-(np.square(dx) + np.square(dy)) / denominator)
        height = height + gaussian_value
        dh_dx = dh_dx + gaussian_value * (-2.0 * dx / denominator)
        dh_dy = dh_dy + gaussian_value * (-2.0 * dy / denominator)
    return height, dh_dx, dh_dy


_HEIGHT_FUNCTIONS: dict[str, Callable[[dict[str, float], Any, Any], tuple[Any, Any, Any]]] = {
    "flat": _flat_terms,
    "slope": _slope_terms,
    "paraboloid": _paraboloid_terms,
    "sinusoidal": _sinusoidal_terms,
    "gaussian": _gaussian_terms,
    "composite": _composite_terms,
}


def _evaluate_surface_terms_raw(surface_evaluator: SurfaceEvaluator, x: Any, y: Any) -> tuple[Any, Any, Any]:
    height_function = _HEIGHT_FUNCTIONS.get(surface_evaluator.kind)
    if height_function is None:
        raise ValueError(f"Unsupported height_function name: {surface_evaluator.kind}")
    return height_function(surface_evaluator.parameters, x, y)


def _evaluate_surface_terms(surface_evaluator: SurfaceEvaluator, x_coord: float, y_coord: float) -> tuple[float, float, float]:
    height, dh_dx, dh_dy = _evaluate_surface_terms_raw(surface_evaluator, float(x_coord), float(y_coord))
    return float(height), float(dh_dx), float(dh_dy)


def _evaluate_height_function_grid(surface_evaluator: SurfaceEvaluator, x_values: np.ndarray, y_values: np.ndarray) -> np.ndarray:
    grid_x, grid_y = np.meshgrid(x_values, y_values, indexing="xy")
    heights, _, _ = _evaluate_surface_terms_raw(surface_evaluator, grid_x, grid_y)
    return np.broadcast_to(np.asarray(heights, dtype=float), grid_x.shape)


def surface_is_approximately_flat(surface_evaluator: SurfaceEvaluator, tolerance: float = 1.0e-12) -> bool:
    parameters = surface_evaluator.parameters
    if surface_evaluator.kind == "flat":
        return True
    if surface_evaluator.kind == "paraboloid":
        return (
            abs(_get_surface_parameter(parameters, "curvature_x")) <= tolerance
            and abs(_get_surface_parameter(parameters, "curvature_y")) <= tolerance
        )
    if surface_evaluator.kind == "sinusoidal":
        return abs(_get_surface_parameter(parameters, "amplitude")) <= tolerance
    if surface_evaluator.kind == "composite":
        gaussian_terms = _get_composite_gaussian_terms(surface_evaluator.parameters)
        return (
            abs(_get_surface_parameter(parameters, "slope_x")) <= tolerance
            and abs(_get_surface_parameter(parameters, "slope_y")) <= tolerance
            and all(abs(amplitude) <= tolerance for amplitude, _, _, _ in gaussian_terms)
        )
    return False


def surface_can_use_plane_geom(surface_evaluator: SurfaceEvaluator) -> bool:
    return surface_evaluator.kind == "slope" or surface_is_approximately_flat(surface_evaluator)


def evaluate_height_function(surface_evaluator: SurfaceEvaluator, x_coord: float, y_coord: float) -> float:
    height, _, _ = _evaluate_surface_terms(surface_evaluator, x_coord, y_coord)
    return height


def evaluate_height_gradient(surface_evaluator: SurfaceEvaluator, x_coord: float, y_coord: float) -> tuple[float, float]:
    _, dh_dx, dh_dy = _evaluate_surface_terms(surface_evaluator, x_coord, y_coord)
    return dh_dx, dh_dy


def evaluate_surface_properties(surface_evaluator: SurfaceEvaluator, x_coord: float, y_coord: float) -> tuple[float, np.ndarray]:
    height, dh_dx, dh_dy = _evaluate_surface_terms(surface_evaluator, x_coord, y_coord)
    normal = np.array([-dh_dx, -dh_dy, 1.0], dtype=float)
    return height, normal / np.linalg.norm(normal)


def evaluate_surface_normal(surface_evaluator: SurfaceEvaluator, x_coord: float, y_coord: float) -> np.ndarray:
    _, normal = evaluate_surface_properties(surface_evaluator, x_coord, y_coord)
    return normal


# ---------------------------------------------------------------------------
# XML emission
# ---------------------------------------------------------------------------

def _get_height_function_grid(surface_config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, SurfaceEvaluator]:
    function_config = surface_config["height_function"]
    surface_evaluator = build_surface_evaluator(surface_config)
    if surface_evaluator is None:
        raise ValueError("Surface evaluator is required for height_function surfaces")

    x_min, x_max = [float(value) for value in function_config["x_range"]]
    y_min, y_max = [float(value) for value in function_config["y_range"]]
    grid_x, grid_y = [int(value) for value in function_config["grid_resolution"]]
    if grid_x < 2 or grid_y < 2:
        raise ValueError("height_function grid_resolution must be at least [2, 2]")

    x_values = np.linspace(x_min, x_max, grid_x)
    y_values = np.linspace(y_min, y_max, grid_y)
    heights = _evaluate_height_function_grid(surface_evaluator, x_values, y_values)

    return x_values, y_values, heights, surface_evaluator


def _get_surface_plane_size(surface_config: dict[str, Any]) -> list[float]:
    plane_config = surface_config.get("plane")
    if isinstance(plane_config, dict) and "size" in plane_config:
        return [float(value) for value in plane_config["size"]]

    function_config = surface_config["height_function"]
    x_min, x_max = [float(value) for value in function_config["x_range"]]
    y_min, y_max = [float(value) for value in function_config["y_range"]]
    return [0.5 * (x_max - x_min), 0.5 * (y_max - y_min), 0.1]


def _build_plane_surface_blocks(
    surface_config: dict[str, Any],
    surface_geom_attributes: dict[str, str | None],
    surface_evaluator: SurfaceEvaluator | None,
) -> tuple[str, str, SurfaceEvaluator | None]:
    plane_size = _get_surface_plane_size(surface_config)
    geom_attributes = {
        **surface_geom_attributes,
        "type": "plane",
        "size": format_vector(plane_size),
    }
    if surface_evaluator is not None:
        origin_height = evaluate_height_function(surface_evaluator, 0.0, 0.0)
        geom_attributes["pos"] = format_vector([0.0, 0.0, origin_height])
        geom_attributes["zaxis"] = format_vector(evaluate_surface_normal(surface_evaluator, 0.0, 0.0).tolist())

    geom_block = f'    <geom {build_xml_attributes(geom_attributes)}/>'
    return "", geom_block, surface_evaluator


def _build_hfield_surface_blocks(
    surface_config: dict[str, Any],
    surface_geom_attributes: dict[str, str | None],
) -> tuple[str, str, SurfaceEvaluator]:
    x_values, y_values, heights, surface_evaluator = _get_height_function_grid(surface_config)
    min_height = float(np.min(heights))
    max_height = float(np.max(heights))
    height_span = max_height - min_height
    if height_span <= 1.0e-12:
        return _build_plane_surface_blocks(surface_config, surface_geom_attributes, surface_evaluator)

    normalized_heights = np.array((heights - min_height) / height_span, dtype=float)
    # MuJoCo's XML parser is sensitive to extreme scientific-notation values in
    # hfield elevation attributes, so collapse numerically insignificant tails.
    np.clip(normalized_heights, 0.0, 1.0, out=normalized_heights)
    normalized_heights[normalized_heights < 1.0e-12] = 0.0
    normalized_heights[normalized_heights > 1.0 - 1.0e-12] = 1.0
    # MuJoCo places the FIRST elevation row at the MAXIMUM y (image-style, top
    # row first; verified empirically with a 3x3 ramp hfield and mj_ray).
    # heights[i, j] = h(x_j, y_i) with y ascending, so flip the row order or
    # the built terrain is the y-mirror h(x, -y) of the intended surface.
    # (This mirrored every generated surface; it went unnoticed because the
    # early test surfaces were y-symmetric.)
    normalized_heights = np.flipud(normalized_heights)
    elevation_values = " ".join(f"{value:.9g}" for value in normalized_heights.reshape(-1))
    x_radius = 0.5 * float(x_values[-1] - x_values[0])
    y_radius = 0.5 * float(y_values[-1] - y_values[0])
    center_x = 0.5 * float(x_values[0] + x_values[-1])
    center_y = 0.5 * float(y_values[0] + y_values[-1])
    base_thickness = max(float(surface_config["height_function"].get("base_thickness", 0.1)), 1.0e-4)

    asset_attributes = {
        "name": SURFACE_HFIELD_NAME,
        "nrow": format_scalar(int(len(y_values))),
        "ncol": format_scalar(int(len(x_values))),
        "size": format_vector([x_radius, y_radius, height_span, base_thickness]),
        "elevation": elevation_values,
    }
    geom_attributes = {
        **surface_geom_attributes,
        "type": "hfield",
        "hfield": SURFACE_HFIELD_NAME,
        "pos": format_vector([center_x, center_y, min_height]),
    }
    asset_block = f'    <hfield {build_xml_attributes(asset_attributes)}/>'
    geom_block = f'    <geom {build_xml_attributes(geom_attributes)}/>'
    return asset_block, geom_block, surface_evaluator


def build_surface_blocks(surface_config: dict[str, Any]) -> tuple[str, str, SurfaceEvaluator | None]:
    surface_type = str(surface_config["type"])
    surface_geom_attributes = _build_surface_geom_attributes(surface_config)

    if surface_type == "plane":
        return _build_plane_surface_blocks(surface_config, surface_geom_attributes, None)

    if surface_type == "height_function":
        surface_evaluator = build_surface_evaluator(surface_config)
        if surface_evaluator is None:
            raise ValueError("Surface evaluator is required for height_function surfaces")
        if surface_can_use_plane_geom(surface_evaluator):
            return _build_plane_surface_blocks(surface_config, surface_geom_attributes, surface_evaluator)
        return _build_hfield_surface_blocks(surface_config, surface_geom_attributes)

    raise ValueError(f"Unsupported environment.surface.type: {surface_type}")
