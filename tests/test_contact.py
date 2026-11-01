from __future__ import annotations

import mujoco
import numpy as np

from wheeled_uav.config import load_vehicle_params
from wheeled_uav.model.builder import render_model_xml
from wheeled_uav.runtime.contact import (
    build_contact_attribution,
    build_contact_report,
    build_contact_reports,
    build_geom_name_lookup,
)


def _settled_scene(tmp_path, num_uavs: int, spawn_radius: float = 1.0):
    """Default vehicles dropped onto the default floor until their wheels rest on it."""
    params = load_vehicle_params()
    xml_path, surface_evaluator, uav_specs = render_model_xml(
        params, output_path=tmp_path / f"model_{num_uavs}.xml", num_uavs=num_uavs, spawn_radius=spawn_radius
    )
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    for _ in range(400):
        mujoco.mj_step(model, data)
    assert data.ncon > 0
    return model, data, surface_evaluator, uav_specs


def test_each_uav_report_holds_only_its_own_contacts(tmp_path):
    model, data, surface_evaluator, uav_specs = _settled_scene(tmp_path, num_uavs=3)
    geom_names = build_geom_name_lookup(model)

    reports = build_contact_reports(model, data, surface_evaluator, uav_specs, geom_names=geom_names)

    assert len(reports) == 3
    for uav_index, (report, uav_spec) in enumerate(zip(reports, uav_specs)):
        own = [record for record in report["contacts"] if uav_spec.contact_prefix in record["geom1"] + record["geom2"]]
        assert report["contacts"] == own
        assert report["count"] == len(own) > 0
        assert report["surface"]["count"] == report["count"]
        wheel_count = report["left_wheel"]["count"] + report["right_wheel"]["count"]
        assert wheel_count == report["count"]
    # Vehicles on the floor do not touch each other, so the reports partition the scene.
    assert sum(report["count"] for report in reports) == data.ncon


def test_single_uav_report_matches_the_scene_wide_report(tmp_path):
    model, data, surface_evaluator, uav_specs = _settled_scene(tmp_path, num_uavs=1)
    geom_names = build_geom_name_lookup(model)

    (report,) = build_contact_reports(model, data, surface_evaluator, uav_specs, geom_names=geom_names)
    legacy = build_contact_report(
        model, data, surface_evaluator, contact_prefix=uav_specs[0].contact_prefix, geom_names=geom_names
    )

    assert report == legacy


def test_contact_between_two_vehicles_appears_in_both_reports():
    xml = """
    <mujoco>
      <worldbody>
        <geom name="surface_geom" type="plane" size="5 5 0.1"/>
        <body name="uav_1" pos="0 0 0.2"><freejoint/><geom name="uav_1_drone_body_box" type="box" size="0.1 0.1 0.1"/></body>
        <body name="uav_2" pos="0.19 0 0.2"><freejoint/><geom name="uav_2_drone_body_box" type="box" size="0.1 0.1 0.1"/></body>
      </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    class _Spec:
        def __init__(self, name: str) -> None:
            self.body_name = name
            self.contact_prefix = f"{name}_"

    specs = [_Spec("uav_1"), _Spec("uav_2")]
    attribution = build_contact_attribution(model, specs)
    reports = build_contact_reports(model, data, None, specs, attribution=attribution)

    shared = [record for record in reports[0]["contacts"] if "uav_2" in record["geom1"] + record["geom2"]]
    assert shared, "the boxes overlap, so the scene must hold a vehicle-vehicle contact"
    assert shared == [record for record in reports[1]["contacts"] if "uav_1" in record["geom1"] + record["geom2"]]
    assert all(report["surface"]["count"] == 0 for report in reports)
    assert np.isclose(reports[0]["total_normal_force"], reports[1]["total_normal_force"])
