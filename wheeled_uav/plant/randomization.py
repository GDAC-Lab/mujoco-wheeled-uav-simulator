"""Randomized parameters: draw parameter values before the model is built (Monte Carlo studies).

The ``randomization`` section of vehicle_params.json names parameters by
their dotted path in the params (list entries by index) and how to draw
them::

    "randomization": {
      "parameters": {
        "drone.mass":                            {"uniform": [0.85, 0.95]},
        "drone.inertia":                         {"scale_uniform": [0.9, 1.1]},
        "plant.actuation.motor_time_constant_up_s": {"uniform": [0.02, 0.04]},
        "plant.actuation.thrust_scale":          {"normal": [1.0, 0.03], "clip": [0.9, 1.1]},
        "plant.aerodynamics.wind_m_per_s":       {"uniform": [[-1, -1, 0], [1, 1, 0]]},
        "actuation.rotors.0.position_body.2":    {"choice": [0.010, 0.012]}
      }
    }

Distributions: ``uniform: [low, high]``, ``normal: [mean, std]``,
``choice: [value, ...]`` set the value; ``scale_uniform: [low, high]`` and
``scale_normal: std`` multiply the present value (one common factor for
every number in it, so a matrix keeps its proportions, unless
``"per_element": true``). ``clip: [low, high]`` bounds the result. For a
list-valued parameter ``uniform`` and ``normal`` draw each element (bounds
may be lists of the same length). Objects missing on the way to a ``plant``
parameter are created (the plant parser rejects misspelt keys); anywhere
else the parent must exist. The draws of a parameter depend only on
the run's seed and the parameter's path, so adding or removing another
parameter does not change them. The drawn values are written next to the
generated model XML (``<model>.randomization.json``).
"""

from __future__ import annotations

import copy
import json
import zlib
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["RANDOMIZATION_STREAM", "apply_randomization", "write_randomization_record"]

RANDOMIZATION_STREAM = 0x52414E44  # "RAND"
_DISTRIBUTIONS = ("uniform", "normal", "choice", "scale_uniform", "scale_normal")


def _split(path: str) -> list[str | int]:
    if not isinstance(path, str) or not path or path.startswith(".") or path.endswith(".") or ".." in path:
        raise ValueError(f"randomization: bad parameter path {path!r}")
    return [int(part) if part.isdigit() else part for part in path.split(".")]


def _parent(params: dict[str, Any], path: str) -> tuple[Any, str | int]:
    parts = _split(path)
    node: Any = params
    for depth, part in enumerate(parts[:-1]):
        if isinstance(node, dict) and part not in node and parts[0] == "plant" and isinstance(part, str):
            # Missing objects under plant are created (its parser rejects misspelt keys);
            # elsewhere a missing parent is an error, so a typo cannot pass unnoticed.
            node[part] = {}
        try:
            node = node[part]
        except (KeyError, IndexError, TypeError):
            raise ValueError(f"randomization: {'.'.join(map(str, parts[: depth + 1]))!r} does not exist (in {path!r})") from None
    if not isinstance(node, (dict, list)):
        raise ValueError(f"randomization: the parent of {path!r} is not an object or a list")
    return node, parts[-1]


def _numbers(value: Any, path: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        raise ValueError(f"randomization: {path!r} must hold numbers to be scaled or drawn per element") from None
    if not np.all(np.isfinite(array)):
        raise ValueError(f"randomization: {path!r} holds non-finite numbers")
    return array


def _to_json(array: np.ndarray) -> Any:
    return float(array) if array.ndim == 0 else array.tolist()


def _clip(array: np.ndarray, spec: dict[str, Any], path: str) -> np.ndarray:
    if "clip" not in spec:
        return array
    clip = spec["clip"]
    if not isinstance(clip, list) or len(clip) != 2:
        raise ValueError(f"randomization: clip of {path!r} must be [low, high]")
    return np.clip(array, _numbers(clip[0], path), _numbers(clip[1], path))


def _draw(spec: dict[str, Any], present: Any, has_present: bool, path: str, rng: np.random.Generator) -> Any:
    if not isinstance(spec, dict):
        raise ValueError(f"randomization: the entry of {path!r} must be an object")
    unknown = set(spec) - set(_DISTRIBUTIONS) - {"clip", "per_element"}
    if unknown:
        raise ValueError(f"randomization: unknown key(s) {sorted(unknown)} for {path!r}")
    kinds = [kind for kind in _DISTRIBUTIONS if kind in spec]
    if len(kinds) != 1:
        raise ValueError(f"randomization: give exactly one of {', '.join(_DISTRIBUTIONS)} for {path!r}")
    kind = kinds[0]
    argument = spec[kind]
    per_element = spec.get("per_element", False)
    if not isinstance(per_element, bool):
        raise ValueError(f"randomization: per_element of {path!r} must be true or false")
    if kind == "choice":
        if not isinstance(argument, list) or not argument:
            raise ValueError(f"randomization: choice of {path!r} must be a non-empty list")
        value = copy.deepcopy(argument[int(rng.integers(len(argument)))])
        return _to_json(_clip(_numbers(value, path), spec, path)) if "clip" in spec else value
    if kind in ("uniform", "normal"):
        if not isinstance(argument, list) or len(argument) != 2:
            raise ValueError(f"randomization: {kind} of {path!r} must be [{'low, high' if kind == 'uniform' else 'mean, std'}]")
        first, second = _numbers(argument[0], path), _numbers(argument[1], path)
        shape = np.broadcast_shapes(first.shape, second.shape, _numbers(present, path).shape if has_present else ())
        if kind == "uniform":
            if np.any(second < first):
                raise ValueError(f"randomization: uniform of {path!r} needs low <= high")
            array = rng.uniform(np.broadcast_to(first, shape), np.broadcast_to(second, shape))
        else:
            if np.any(second < 0.0):
                raise ValueError(f"randomization: normal of {path!r} needs std >= 0")
            array = rng.normal(np.broadcast_to(first, shape), np.broadcast_to(second, shape))
        return _to_json(_clip(np.asarray(array, dtype=float), spec, path))
    if not has_present:
        raise ValueError(f"randomization: {kind} scales the present value, but {path!r} is not set")
    base = _numbers(present, path)
    shape = base.shape if per_element else ()
    if kind == "scale_uniform":
        if not isinstance(argument, list) or len(argument) != 2 or float(argument[1]) < float(argument[0]):
            raise ValueError(f"randomization: scale_uniform of {path!r} must be [low, high] with low <= high")
        factor = rng.uniform(float(argument[0]), float(argument[1]), size=shape)
    else:
        if isinstance(argument, bool) or not isinstance(argument, (int, float)) or argument < 0:
            raise ValueError(f"randomization: scale_normal of {path!r} must be a standard deviation >= 0")
        factor = rng.normal(1.0, float(argument), size=shape)
    return _to_json(_clip(base * factor, spec, path))


def apply_randomization(params: dict[str, Any], *, seed: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return a copy of ``params`` with the randomized values drawn, and the list of draws (empty without a section)."""
    section = params.get("randomization")
    if section is None:
        return params, []
    if not isinstance(section, dict):
        raise ValueError("randomization must be an object")
    enabled = section.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ValueError("randomization.enabled must be true or false")
    unknown = {name for name in section if not name.startswith("_")} - {"enabled", "parameters"}
    if unknown:
        raise ValueError(f"unknown key(s) in randomization: {', '.join(sorted(unknown))}; allowed: enabled, parameters")
    parameters = section.get("parameters", {})
    if not isinstance(parameters, dict):
        raise ValueError("randomization.parameters must be an object of path: distribution")
    if not enabled or not parameters:
        return params, []
    randomized = copy.deepcopy(params)
    draws = []
    for path, spec in parameters.items():
        if path.startswith("randomization"):
            raise ValueError("randomization cannot randomize itself")
        parent, key = _parent(randomized, path)
        if isinstance(parent, list):
            if not isinstance(key, int) or not -len(parent) <= key < len(parent):
                raise ValueError(f"randomization: index {key!r} out of range in {path!r}")
            has_present, present = True, parent[key]
        else:
            if not isinstance(key, str):
                key = str(key)
            has_present, present = key in parent, parent.get(key)
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), RANDOMIZATION_STREAM, zlib.crc32(path.encode("utf-8"))]))
        value = _draw(spec, present, has_present and present is not None, path, rng)
        parent[key] = value
        draws.append({"path": path, "value": value, "distribution": {k: v for k, v in spec.items()}})
    return randomized, draws


def write_randomization_record(model_xml_path: Path, seed: int, draws: list[dict[str, Any]]) -> Path:
    """Write the draws next to the model XML; return the file's path."""
    record_path = model_xml_path.with_name(f"{model_xml_path.stem}.randomization.json")
    record_path.write_text(json.dumps({"seed": int(seed), "draws": draws}, indent=1) + "\n", encoding="utf-8")
    return record_path
