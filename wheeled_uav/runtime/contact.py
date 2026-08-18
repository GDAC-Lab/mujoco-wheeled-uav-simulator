from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

from ..paths import SURFACE_GEOM_NAME, WALL_GEOM_NAME
from ..model.surface import evaluate_surface_properties
from ..types import SurfaceEvaluator

__all__ = ["build_contact_report", "build_geom_name_lookup"]

CONTACT_SUMMARY_GROUPS = ("left_wheel", "right_wheel", "surface", "wall")


def _initialize_contact_summary() -> dict[str, float]:
    return {
        "count": 0.0,
        "total_force_magnitude": 0.0,
        "max_force_magnitude": 0.0,
        "total_normal_force": 0.0,
        "max_normal_force": 0.0,
    }


def _accumulate_contact_summary(summary: dict[str, float], force_magnitude: float, normal_force: float) -> None:
    summary["count"] += 1.0
    summary["total_force_magnitude"] += force_magnitude
    summary["max_force_magnitude"] = max(summary["max_force_magnitude"], force_magnitude)
    summary["total_normal_force"] += normal_force
    summary["max_normal_force"] = max(summary["max_normal_force"], normal_force)


def _build_contact_group_summaries() -> dict[str, dict[str, float]]:
    return {group_name: _initialize_contact_summary() for group_name in CONTACT_SUMMARY_GROUPS}


def _get_geom_name(model: mujoco.MjModel, geom_id: int) -> str:
    geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
    if geom_name is None:
        return f"geom_{geom_id}"
    return geom_name


def build_geom_name_lookup(model: mujoco.MjModel) -> tuple[str, ...]:
    return tuple(_get_geom_name(model, geom_id) for geom_id in range(model.ngeom))


def _resolve_geom_name(model: mujoco.MjModel, geom_id: int, geom_names: tuple[str, ...] | None) -> str:
    if geom_names is not None and 0 <= geom_id < len(geom_names):
        return geom_names[geom_id]
    return _get_geom_name(model, geom_id)


def _extract_surface_contact_details(
    geom1_name: str,
    geom2_name: str,
    contact_position: np.ndarray,
    surface_evaluator: SurfaceEvaluator | None,
) -> dict[str, object]:
    is_surface_contact = SURFACE_GEOM_NAME in (geom1_name, geom2_name)
    details: dict[str, object] = {"surface_contact": is_surface_contact}
    if not is_surface_contact:
        return details

    details["surface_name"] = SURFACE_GEOM_NAME
    if surface_evaluator is None:
        details["surface_normal"] = [0.0, 0.0, 1.0]
        details["surface_height"] = 0.0
        return details

    x_coord = float(contact_position[0])
    y_coord = float(contact_position[1])
    surface_height, surface_normal = evaluate_surface_properties(surface_evaluator, x_coord, y_coord)
    details["surface_height"] = surface_height
    details["surface_normal"] = surface_normal.tolist()
    return details


def build_contact_report(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    surface_evaluator: SurfaceEvaluator | None,
    contact_prefix: str = "",
    geom_names: tuple[str, ...] | None = None,
    include_details: bool = True,
) -> dict[str, Any]:
    contacts: list[dict[str, object]] = []
    overall_summary = _initialize_contact_summary()
    contact_group_summaries = _build_contact_group_summaries()
    left_wheel_geom_name = f"{contact_prefix}left_wheel_geom"
    right_wheel_geom_name = f"{contact_prefix}right_wheel_geom"
    contact_wrench = np.zeros(6, dtype=float)

    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        contact_wrench.fill(0.0)
        mujoco.mj_contactForce(model, data, contact_index, contact_wrench)

        force_vector = contact_wrench[:3]
        force_magnitude = float(np.linalg.norm(force_vector))
        normal_force = float(abs(contact_wrench[0]))
        geom1_name = _resolve_geom_name(model, int(contact.geom1), geom_names)
        geom2_name = _resolve_geom_name(model, int(contact.geom2), geom_names)

        _accumulate_contact_summary(overall_summary, force_magnitude, normal_force)
        if left_wheel_geom_name in (geom1_name, geom2_name):
            _accumulate_contact_summary(contact_group_summaries["left_wheel"], force_magnitude, normal_force)
        if right_wheel_geom_name in (geom1_name, geom2_name):
            _accumulate_contact_summary(contact_group_summaries["right_wheel"], force_magnitude, normal_force)
        if SURFACE_GEOM_NAME in (geom1_name, geom2_name):
            _accumulate_contact_summary(contact_group_summaries["surface"], force_magnitude, normal_force)
        if WALL_GEOM_NAME in (geom1_name, geom2_name):
            _accumulate_contact_summary(contact_group_summaries["wall"], force_magnitude, normal_force)

        if include_details:
            contact_position = np.array(contact.pos, copy=True)
            contacts.append(
                {
                    "index": contact_index,
                    "geom1": geom1_name,
                    "geom2": geom2_name,
                    "position": contact_position.tolist(),
                    "distance": float(contact.dist),
                    "force_contact_frame": force_vector.tolist(),
                    "torque_contact_frame": contact_wrench[3:].tolist(),
                    "normal_force": normal_force,
                    "force_magnitude": force_magnitude,
                    **_extract_surface_contact_details(geom1_name, geom2_name, contact_position, surface_evaluator),
                }
            )

    return {
        "count": int(overall_summary["count"]),
        "total_force_magnitude": overall_summary["total_force_magnitude"],
        "max_force_magnitude": overall_summary["max_force_magnitude"],
        "total_normal_force": overall_summary["total_normal_force"],
        "max_normal_force": overall_summary["max_normal_force"],
        "left_wheel": contact_group_summaries["left_wheel"],
        "right_wheel": contact_group_summaries["right_wheel"],
        "surface": contact_group_summaries["surface"],
        "wall": contact_group_summaries["wall"],
        "contacts": contacts,
    }
