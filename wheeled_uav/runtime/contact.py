"""Contact reports for the state packet.

``build_contact_reports`` walks ``data.contact`` once and files every contact
under the UAV (or UAVs) whose geoms take part in it, so each vehicle's report
covers its own contacts and the cost grows with the number of contacts rather
than with vehicles x contacts.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

from ..model.surface import evaluate_surface_properties
from ..paths import SURFACE_GEOM_NAME, WALL_GEOM_NAME
from ..types import SurfaceEvaluator, UAVModelSpec

__all__ = [
    "ContactAttribution",
    "build_contact_attribution",
    "build_contact_report",
    "build_contact_reports",
    "build_geom_name_lookup",
]

CONTACT_SUMMARY_GROUPS = ("left_wheel", "right_wheel", "surface", "wall")


@dataclass(frozen=True)
class ContactAttribution:
    """Geom ownership used to split the scene's contacts by UAV.

    ``geom_owner[g]`` is the index of the UAV whose body subtree holds geom
    ``g``, or -1 for scenery (terrain, wall, overlays). The wheel geom ids are
    per UAV and -1 where a vehicle has no such geom.
    """

    geom_owner: tuple[int, ...]
    left_wheel_geoms: tuple[int, ...]
    right_wheel_geoms: tuple[int, ...]
    surface_geom: int
    wall_geom: int


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


def _build_contact_record(
    contact_index: int,
    geom1_name: str,
    geom2_name: str,
    contact_position: np.ndarray,
    distance: float,
    contact_wrench: np.ndarray,
    force_magnitude: float,
    normal_force: float,
    surface_evaluator: SurfaceEvaluator | None,
) -> dict[str, object]:
    return {
        "index": contact_index,
        "geom1": geom1_name,
        "geom2": geom2_name,
        "position": contact_position.tolist(),
        "distance": distance,
        "force_contact_frame": contact_wrench[:3].tolist(),
        "torque_contact_frame": contact_wrench[3:].tolist(),
        "normal_force": normal_force,
        "force_magnitude": force_magnitude,
        **_extract_surface_contact_details(geom1_name, geom2_name, contact_position, surface_evaluator),
    }


def _finish_report(
    overall_summary: dict[str, float],
    group_summaries: dict[str, dict[str, float]],
    contacts: list[dict[str, object]],
) -> dict[str, Any]:
    return {
        "count": int(overall_summary["count"]),
        "total_force_magnitude": overall_summary["total_force_magnitude"],
        "max_force_magnitude": overall_summary["max_force_magnitude"],
        "total_normal_force": overall_summary["total_normal_force"],
        "max_normal_force": overall_summary["max_normal_force"],
        "left_wheel": group_summaries["left_wheel"],
        "right_wheel": group_summaries["right_wheel"],
        "surface": group_summaries["surface"],
        "wall": group_summaries["wall"],
        "contacts": contacts,
    }


def build_contact_attribution(model: mujoco.MjModel, uav_specs: Sequence[UAVModelSpec]) -> ContactAttribution:
    uav_of_body = [-1] * model.nbody
    for uav_index, uav_spec in enumerate(uav_specs):
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, uav_spec.body_name)
        if body_id < 0:
            raise ValueError(f"UAV body {uav_spec.body_name!r} not found in the compiled model")
        uav_of_body[body_id] = uav_index
    # Bodies are numbered parents-first, so a single forward sweep hands every
    # body the UAV of its nearest UAV ancestor (the wheels inherit their vehicle).
    for body_id in range(1, model.nbody):
        if uav_of_body[body_id] < 0:
            uav_of_body[body_id] = uav_of_body[int(model.body_parentid[body_id])]

    def geom_id(name: str) -> int:
        return int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name))

    return ContactAttribution(
        geom_owner=tuple(uav_of_body[int(body_id)] for body_id in model.geom_bodyid),
        left_wheel_geoms=tuple(geom_id(f"{spec.contact_prefix}left_wheel_geom") for spec in uav_specs),
        right_wheel_geoms=tuple(geom_id(f"{spec.contact_prefix}right_wheel_geom") for spec in uav_specs),
        surface_geom=geom_id(SURFACE_GEOM_NAME),
        wall_geom=geom_id(WALL_GEOM_NAME),
    )


def build_contact_reports(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    surface_evaluator: SurfaceEvaluator | None,
    uav_specs: Sequence[UAVModelSpec],
    *,
    geom_names: tuple[str, ...] | None = None,
    include_details: bool = True,
    attribution: ContactAttribution | None = None,
) -> list[dict[str, Any]]:
    """Contact report of every UAV, in ``uav_specs`` order, from one pass.

    A contact counts for each UAV that owns one of its two geoms, so a contact
    between two vehicles appears in both reports. The report layout is the one
    of :func:`build_contact_report`; the ``surface`` and ``wall`` groups and
    the overall totals cover the vehicle's own contacts only.
    """
    if attribution is None:
        attribution = build_contact_attribution(model, uav_specs)
    uav_count = len(uav_specs)
    overall_summaries = [_initialize_contact_summary() for _ in range(uav_count)]
    group_summaries = [_build_contact_group_summaries() for _ in range(uav_count)]
    contact_records: list[list[dict[str, object]]] = [[] for _ in range(uav_count)]
    owner = attribution.geom_owner
    contact_wrench = np.zeros(6, dtype=float)
    contacts = data.contact
    geom1_ids = contacts.geom1
    geom2_ids = contacts.geom2

    for contact_index in range(data.ncon):
        geom1 = int(geom1_ids[contact_index])
        geom2 = int(geom2_ids[contact_index])
        owner1 = owner[geom1]
        owner2 = owner[geom2]
        if owner1 < 0 and owner2 < 0:
            continue
        if owner2 < 0 or owner2 == owner1:
            owners: tuple[int, ...] = (owner1,)
        elif owner1 < 0:
            owners = (owner2,)
        else:
            owners = (min(owner1, owner2), max(owner1, owner2))

        contact_wrench.fill(0.0)
        mujoco.mj_contactForce(model, data, contact_index, contact_wrench)
        force_magnitude = float(np.linalg.norm(contact_wrench[:3]))
        normal_force = float(abs(contact_wrench[0]))
        touches_surface = attribution.surface_geom in (geom1, geom2)
        touches_wall = attribution.wall_geom in (geom1, geom2)
        record = None
        if include_details:
            record = _build_contact_record(
                contact_index,
                _resolve_geom_name(model, geom1, geom_names),
                _resolve_geom_name(model, geom2, geom_names),
                np.array(contacts.pos[contact_index], copy=True),
                float(contacts.dist[contact_index]),
                contact_wrench,
                force_magnitude,
                normal_force,
                surface_evaluator,
            )

        for uav_index in owners:
            groups = group_summaries[uav_index]
            _accumulate_contact_summary(overall_summaries[uav_index], force_magnitude, normal_force)
            if attribution.left_wheel_geoms[uav_index] in (geom1, geom2):
                _accumulate_contact_summary(groups["left_wheel"], force_magnitude, normal_force)
            if attribution.right_wheel_geoms[uav_index] in (geom1, geom2):
                _accumulate_contact_summary(groups["right_wheel"], force_magnitude, normal_force)
            if touches_surface:
                _accumulate_contact_summary(groups["surface"], force_magnitude, normal_force)
            if touches_wall:
                _accumulate_contact_summary(groups["wall"], force_magnitude, normal_force)
            if record is not None:
                contact_records[uav_index].append(record)

    return [
        _finish_report(overall_summaries[uav_index], group_summaries[uav_index], contact_records[uav_index])
        for uav_index in range(uav_count)
    ]


def build_contact_report(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    surface_evaluator: SurfaceEvaluator | None,
    contact_prefix: str = "",
    geom_names: tuple[str, ...] | None = None,
    include_details: bool = True,
) -> dict[str, Any]:
    """Scene-wide contact report with the wheel groups of one vehicle.

    Every contact in the scene enters the totals and the ``surface``/``wall``
    groups; only the wheel groups are filtered by ``contact_prefix``. Kept for
    existing callers; the simulator itself reports per vehicle through
    :func:`build_contact_reports`.
    """
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

        force_magnitude = float(np.linalg.norm(contact_wrench[:3]))
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
            contacts.append(
                _build_contact_record(
                    contact_index,
                    geom1_name,
                    geom2_name,
                    np.array(contact.pos, copy=True),
                    float(contact.dist),
                    contact_wrench,
                    force_magnitude,
                    normal_force,
                    surface_evaluator,
                )
            )

    return _finish_report(overall_summary, contact_group_summaries, contacts)
