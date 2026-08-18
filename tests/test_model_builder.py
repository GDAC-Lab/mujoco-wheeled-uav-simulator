from __future__ import annotations

import json
import unittest

from wheeled_uav.config import load_vehicle_params
from wheeled_uav.model.builder import build_rotor_specs, build_uav_model_specs, build_xml_replacements
from wheeled_uav.model.surface import build_surface_blocks

class ModelBuilderTests(unittest.TestCase):
    def test_default_vehicle_params_use_vertical_rotor_axes(self) -> None:
        params = load_vehicle_params()

        rotor_specs = build_rotor_specs(params)

        self.assertEqual([rotor_spec.thrust_axis for rotor_spec in rotor_specs], [(0.0, 0.0, 1.0)] * 4)

    def test_build_uav_model_specs_names_uavs_for_multi_uav_world(self) -> None:
        params = load_vehicle_params()

        specs = build_uav_model_specs(params, num_uavs=3)

        self.assertEqual(len(specs), 3)
        self.assertEqual(specs[0].body_name, "uav_1")
        self.assertEqual(specs[1].sensor_names.position, "uav_2_position")
        self.assertEqual(specs[2].actuator_names[0], "uav_3_thrust_fr")

    def test_build_xml_replacements_includes_multi_uav_blocks(self) -> None:
        params = load_vehicle_params()

        replacements, surface_evaluator, uav_specs = build_xml_replacements(params, num_uavs=2, spawn_radius=1.2)

        self.assertEqual(len(uav_specs), 2)
        self.assertIsNone(surface_evaluator)
        self.assertIn('name="uav_1"', replacements["__DRONE_BODY_BLOCK__"])
        self.assertIn('name="uav_2"', replacements["__DRONE_BODY_BLOCK__"])
        self.assertIn('name="uav_1_thrust_fr"', replacements["__ACTUATOR_BLOCK__"])
        self.assertIn('name="uav_2_position"', replacements["__SENSOR_BLOCK__"])
        self.assertIn('friction="1.2 0.005 0.0001"', replacements["__DRONE_BODY_BLOCK__"])
        self.assertEqual(replacements["__WALL_FRICTION__"], '1.2 0.005 0.0001')
        self.assertIn("surface_geom", replacements["__SURFACE_GEOM_BLOCK__"])

    def test_plane_surface_emits_optional_friction_attributes(self) -> None:
        cfg = {
            "type": "plane",
            "material": "floor_mat",
            "solref": [0.005, 1.0],
            "contact": {"contype": 1, "conaffinity": 1},
            "friction": [1.35, 0.007, 0.00012],
            "plane": {"size": [3.0, 3.0, 0.1]},
            "rgba": [0.3, 0.3, 0.35, 1.0],
        }
        _, geom_block, evaluator = build_surface_blocks(cfg)
        self.assertIsNone(evaluator)
        self.assertIn('friction="1.35 0.007 0.00012"', geom_block)

    def test_simulation_solver_placeholders_match_mujoco_defaults(self) -> None:
        params = load_vehicle_params()

        replacements, _, _ = build_xml_replacements(params, num_uavs=1, spawn_radius=1.2)

        self.assertEqual(replacements["__OPTION_ITERATIONS__"], "100")
        self.assertEqual(replacements["__OPTION_NOSLIP_ITERATIONS__"], "0")

    def test_simulation_solver_placeholders_respect_json_overrides(self) -> None:
        params = json.loads(json.dumps(load_vehicle_params()))
        params["simulation"]["iterations"] = 150
        params["simulation"]["noslip_iterations"] = 8

        replacements, _, _ = build_xml_replacements(params, num_uavs=1, spawn_radius=1.2)

        self.assertEqual(replacements["__OPTION_ITERATIONS__"], "150")
        self.assertEqual(replacements["__OPTION_NOSLIP_ITERATIONS__"], "8")


if __name__ == "__main__":
    unittest.main()

# ---------------- ロータ回転方向の規約（PX4 Quad-X と一致させる） ----------------

def _rotor_specs_from_default_params():
    import json
    from pathlib import Path

    from wheeled_uav.model.builder import build_rotor_specs

    params = json.loads(Path("vehicle_params.json").read_text(encoding="utf-8"))
    return {spec.suffix: spec for spec in build_rotor_specs(params)}


def test_spin_signs_match_px4_quad_x_convention():
    """PX4 Quad-X 規約のモータミキサーと反トルク符号が一致すること。

    正本は飛行実績のある Quad-X ミキサー（PX4 の CA_ROTOR*_KM と同じ規約）:

        input = [tau_x; tau_y; tau_z; Thrust]
        alpha = [-1/(4*lx), -1/(4*ly), -1/(4*a1), 1/4;     % M1
                  1/(4*lx),  1/(4*ly), -1/(4*a1), 1/4;     % M2
                  1/(4*lx), -1/(4*ly),  1/(4*a1), 1/4;     % M3
                 -1/(4*lx),  1/(4*ly),  1/(4*a1), 1/4];    % M4

    これを逆にすると、各ロータ推力 -> レンチ の割当行列（NWU, z上）が得られる。
    lx は **y方向**、ly は **x方向** の半スパン（名前と軸が逆なので注意）。
    既定パラメータ（アーム x=0.08, y=0.10）では:

        M1 (+0.08, -0.10) 前右  ヨー係数 -a1
        M2 (-0.08, +0.10) 後左  ヨー係数 -a1
        M3 (+0.08, +0.10) 前左  ヨー係数 +a1
        M4 (-0.08, -0.10) 後右  ヨー係数 +a1

    この符号を反転させると閉ループでヨーが正帰還になり発散する。
    """
    real_yaw_sign_nwu = {"fr": -1.0, "bl": -1.0, "fl": +1.0, "br": +1.0}
    specs = _rotor_specs_from_default_params()
    for name, sign in real_yaw_sign_nwu.items():
        model = specs[name].spin_sign * specs[name].yaw_moment_ratio
        assert (model > 0) == (sign > 0), (
            f"{name}: モデル={model:+.3f} vs 規約上のミキサー係数の符号={sign:+.0f}。"
            "反転するとヨーが正帰還になり閉ループで発散する"
        )


# ---------------------------------------------------------------------------
# Optional mesh: params without drone.mesh must build, render, and compile,
# and relative mesh paths must prefer the params-file directory.
# ---------------------------------------------------------------------------

def _write_tetrahedron_stl(path):
    """Minimal binary STL (a small tetrahedron) for mesh-resolution tests."""
    import struct

    vertices = [
        (0.0, 0.0, 0.0),
        (0.05, 0.0, 0.0),
        (0.0, 0.05, 0.0),
        (0.0, 0.0, 0.05),
    ]
    faces = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
    with open(path, "wb") as stl_file:
        stl_file.write(b"\0" * 80)
        stl_file.write(struct.pack("<I", len(faces)))
        for a, b, c in faces:
            stl_file.write(struct.pack("<3f", 0.0, 0.0, 0.0))
            for index in (a, b, c):
                stl_file.write(struct.pack("<3f", *vertices[index]))
            stl_file.write(struct.pack("<H", 0))


def _fixture_mesh_block(file_name):
    return {
        "file": file_name,
        "scale": [1.0, 1.0, 1.0],
        "contact": {"contype": 0, "conaffinity": 0, "group": 1},
    }


def test_meshless_params_omit_mesh_asset_and_geom():
    params = load_vehicle_params()
    params["drone"].pop("mesh", None)

    replacements, _, _ = build_xml_replacements(params, num_uavs=1, spawn_radius=1.2)

    assert replacements["__MESH_ASSET_BLOCK__"] == ""
    assert "drone_cad" not in replacements["__DRONE_BODY_BLOCK__"]
    assert "__MESH_FILE__" not in replacements


def test_meshless_render_compiles(tmp_path):
    import mujoco

    from wheeled_uav.model.builder import render_model_xml

    params = load_vehicle_params()
    params["drone"].pop("mesh", None)

    xml_path, _, _ = render_model_xml(params, output_path=tmp_path / "model.xml")

    rendered = xml_path.read_text(encoding="utf-8")
    assert "<mesh" not in rendered
    assert "__MESH_ASSET_BLOCK__" not in rendered
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    assert model.nmesh == 0


def test_mesh_resolves_against_params_dir_first(tmp_path):
    import mujoco

    from wheeled_uav.model.builder import render_model_xml

    _write_tetrahedron_stl(tmp_path / "visual.stl")
    params = load_vehicle_params()
    params["drone"]["mesh"] = _fixture_mesh_block("visual.stl")

    xml_path, _, _ = render_model_xml(
        params, output_path=tmp_path / "generated" / "model.xml", params_dir=tmp_path
    )

    rendered = xml_path.read_text(encoding="utf-8")
    assert 'file="../visual.stl"' in rendered
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    assert model.nmesh == 1


def test_mesh_falls_back_to_template_directory(tmp_path):
    import mujoco

    from wheeled_uav.model.builder import render_model_xml
    from wheeled_uav.paths import DEFAULT_PATH_RESOLVER

    template_dir = tmp_path / "template_home"
    template_dir.mkdir()
    template_copy = template_dir / "template.xml"
    template_copy.write_text(
        DEFAULT_PATH_RESOLVER.get_xml_template_path(None).read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    _write_tetrahedron_stl(template_dir / "fallback.stl")
    empty_params_dir = tmp_path / "params_home"
    empty_params_dir.mkdir()

    params = load_vehicle_params()
    params["drone"]["mesh"] = _fixture_mesh_block("fallback.stl")

    xml_path, _, _ = render_model_xml(
        params,
        output_path=tmp_path / "generated" / "model.xml",
        template_path=template_copy,
        params_dir=empty_params_dir,
    )

    rendered = xml_path.read_text(encoding="utf-8")
    assert 'file="../template_home/fallback.stl"' in rendered
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    assert model.nmesh == 1


def test_legacy_template_requires_mesh_when_it_hardcodes_the_asset(tmp_path):
    import pytest

    from wheeled_uav.model.builder import render_model_xml

    legacy_template = tmp_path / "legacy.xml"
    legacy_template.write_text(
        '<mesh name="drone_cad" file="__MESH_FILE__" scale="__MESH_SCALE__"/>',
        encoding="utf-8",
    )
    params = load_vehicle_params()
    params["drone"].pop("mesh", None)

    with pytest.raises(ValueError, match="__MESH_ASSET_BLOCK__"):
        render_model_xml(params, output_path=tmp_path / "model.xml", template_path=legacy_template)
