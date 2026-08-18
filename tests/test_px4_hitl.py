"""PX4 HIL bridge unit tests (no hardware, no pymavlink: FakeLink injection)."""

from __future__ import annotations

import math

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from wheeled_uav.px4_hitl.bridge import (  # noqa: E402
    AutoSequence,
    HilConfig,
    HilStats,
    _map_controls_to_ctrl,
    run_bridge,
)
from wheeled_uav.px4_hitl.sensors import (  # noqa: E402
    AXIS_FLIP,
    HilSensorExtractor,
    gps_from_ned,
    rotation_to_quaternion,
)
from wheeled_uav.runtime.scene import SimulationRequest, load_simulation_scene  # noqa: E402


@pytest.fixture(scope="module")
def scene():
    return load_simulation_scene(SimulationRequest(headless=True, hold_until_first_command=False))


def make_extractor(scene):
    return HilSensorExtractor(scene.model, scene.uav_specs[0].sensor_names, origin_alt_m=488.0)


class FakeLink:
    """Px4HilLink と同じ面を持つ試験用ダミー。"""

    def __init__(self, controls=None, armed=True, arm_after_requests=1):
        self._controls = controls
        self.latest_controls = None
        self.latest_controls_wall_time = 0.0
        self.controls_count = 0
        self.last_heartbeat_wall_time = 0.0
        self.armed = armed
        self.custom_mode = 0
        self.mode_text = "0.0"
        self.statustexts = []
        self.sensor_samples = []
        self.gps_msgs = []
        self.odom_msgs = []
        self.arm_requests = 0
        self.disarm_requests = 0
        self.takeoff_alts = []
        self._arm_after_requests = arm_after_requests
        self._acks: dict[int, int] = {}

    def arm(self, force=False):
        self.arm_requests += 1
        if self.arm_requests >= self._arm_after_requests:
            self.armed = True
        self._acks[400] = 0 if self.armed else 1      # ACCEPTED / TEMPORARILY_REJECTED

    def disarm(self, force=False):
        self.disarm_requests += 1
        self.armed = False

    def takeoff(self, altitude_amsl_m):
        self.takeoff_alts.append(float(altitude_amsl_m))
        self._acks[22] = 0                            # MAV_CMD_NAV_TAKEOFF -> ACCEPTED

    def take_ack(self, command):
        return self._acks.pop(command, None)

    def wait_heartbeat(self, timeout_s: float = 10.0):
        import time

        self.last_heartbeat_wall_time = time.monotonic()
        return {"system": 1, "component": 1, "autopilot": 12, "base_mode": 128}

    def pump(self):
        import time

        if self._controls is not None:
            self.latest_controls = list(self._controls)
            self.latest_controls_wall_time = time.monotonic()
            self.controls_count += 1

    def send_hil_sensor(self, sample):
        self.sensor_samples.append(sample)

    def send_hil_gps(self, time_usec, gps):
        self.gps_msgs.append((time_usec, gps))

    def send_odometry(self, sample):
        self.odom_msgs.append(sample)

    def close(self):
        pass


def test_static_sample_matches_gravity_and_frames(scene):
    model, data = scene.model, scene.data
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    extractor = make_extractor(scene)
    sample = extractor.sample(data)

    # 静止: 比力は FRD で (0,0,-g)、角速度ゼロ、クォータニオンは単位ノルム
    assert sample.accel_frd == pytest.approx([0.0, 0.0, -9.80665], abs=0.05)
    assert np.linalg.norm(sample.gyro_frd) < 1e-6
    assert np.linalg.norm(sample.quat_ned) == pytest.approx(1.0, abs=1e-9)
    # NWU -> NED: z_up -> -z_down
    assert sample.pos_ned[2] == pytest.approx(-float(data.sensordata[2]), abs=1e-9)
    # 磁気ベクトルの大きさは設定値と同じ（回転で不変）
    assert np.linalg.norm(sample.mag_frd) == pytest.approx(np.linalg.norm([0.21523, 0.00771, 0.42741]), rel=1e-6)
    # 気圧: 488m で ~955hPa 付近
    assert 900.0 < sample.abs_pressure_hpa < 1013.0


def test_yaw_sign_flips_between_nwu_and_ned(scene):
    model, data = scene.model, scene.data
    mujoco.mj_resetData(model, data)
    body = model.body(scene.uav_specs[0].body_name)
    joint_id = int(body.jntadr[0])
    qpos_adr = int(model.jnt_qposadr[joint_id])
    yaw = math.radians(30.0)
    data.qpos[qpos_adr + 3:qpos_adr + 7] = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
    mujoco.mj_forward(model, data)

    sample = make_extractor(scene).sample(data)
    w, x, y, z = sample.quat_ned
    yaw_ned = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    # NWU の +30° ヨーは NED では -30°（NWU→NED で y・z 軸が反転するため）
    assert yaw_ned == pytest.approx(-yaw, abs=1e-6)
    data.qpos[qpos_adr + 3:qpos_adr + 7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)


def test_rotation_to_quaternion_roundtrip():
    rng = np.random.default_rng(7)
    for _ in range(20):
        v = rng.normal(size=3)
        angle = rng.uniform(0, math.pi)
        axis = v / np.linalg.norm(v)
        w = math.cos(angle / 2)
        xyz = axis * math.sin(angle / 2)
        q = np.array([w, *xyz])
        if q[0] < 0:
            q = -q
        # quat -> R -> quat
        ww, xx, yy, zz = q
        R = np.array([
            [1 - 2 * (yy**2 + zz**2), 2 * (xx * yy - ww * zz), 2 * (xx * zz + ww * yy)],
            [2 * (xx * yy + ww * zz), 1 - 2 * (xx**2 + zz**2), 2 * (yy * zz - ww * xx)],
            [2 * (xx * zz - ww * yy), 2 * (yy * zz + ww * xx), 1 - 2 * (xx**2 + yy**2)],
        ])
        q2 = rotation_to_quaternion(R)
        assert q2 == pytest.approx(q, abs=1e-8)


def test_gps_from_ned_flat_earth():
    gps = gps_from_ned(np.array([111.0, 0.0, -5.0]), np.array([1.0, 0.0, 0.0]), 47.0, 8.0, 100.0)
    # 北へ111m ≈ 緯度 +0.000998°
    assert (gps["lat"] - 470000000) == pytest.approx(9980, abs=30)
    assert gps["lon"] == 80000000
    assert gps["alt_mm"] == 105000
    assert gps["vn_cm"] == 100 and gps["vel_cm"] == 100


def test_map_controls_quadratic_and_order():
    ctrl = _map_controls_to_ctrl([0.1, 0.2, 0.3, 0.4] + [0.0] * 12, (0, 2, 3, 1), np.full(4, 20.0), 2.0)
    # ctrl[fr,fl,br,bl] <- controls[M1=0.1, M3=0.3, M4=0.4, M2=0.2] を2乗スケール
    assert ctrl == pytest.approx([0.01 * 20, 0.09 * 20, 0.16 * 20, 0.04 * 20])
    # 範囲外はクリップ
    assert _map_controls_to_ctrl([-1.0, 2.0, 0.0, 0.0], (0, 1, 2, 3), np.full(4, 20.0), 2.0)[1] == 20.0


def test_bridge_closed_loop_with_fake_link():
    request = SimulationRequest(headless=True, duration_seconds=0.3, hold_until_first_command=False)
    scene_probe = load_simulation_scene(request)
    mass = float(scene_probe.params["drone"]["mass"])
    hover_thrust_per_rotor = mass * 9.80665 / 4.0
    u_hover = math.sqrt(hover_thrust_per_rotor / 20.0)

    fake = FakeLink(controls=[u_hover] * 4 + [0.0] * 12)
    config = HilConfig(device="FAKE", mode="gps")
    stats = run_bridge(config, request, link=fake)

    assert isinstance(stats, HilStats)
    assert stats.heartbeat_ok
    assert stats.steps >= 100                     # 0.3s / 2ms = 150 歩（実時間ペーシングの揺らぎ許容）
    assert stats.sensor_sent >= stats.steps // 2 - 2
    assert stats.gps_sent >= 2                    # 10Hz * 0.3s
    assert stats.controls_applied_steps > 0
    assert len(fake.sensor_samples) == stats.sensor_sent
    # 送信サンプルの妥当性: 時刻が単調、姿勢ノルム1
    times = [s.time_usec for s in fake.sensor_samples]
    assert all(b > a for a, b in zip(times, times[1:]))
    assert np.linalg.norm(fake.sensor_samples[-1].quat_ned) == pytest.approx(1.0, abs=1e-6)


def test_bridge_ev_mode_sends_normalized_odometry():
    request = SimulationRequest(headless=True, duration_seconds=0.2, hold_until_first_command=False)
    fake = FakeLink(controls=[0.3] * 4 + [0.0] * 12)
    stats = run_bridge(HilConfig(device="FAKE", mode="ev"), request, link=fake)
    assert stats.odom_sent > 0 and stats.gps_sent == 0
    assert len(fake.odom_msgs) == stats.odom_sent


def test_axis_flip_is_involution():
    assert np.allclose(AXIS_FLIP @ AXIS_FLIP, np.eye(3))


# ---------------- RC 無しの自動 arm / 離陸手順 ----------------

def run_sequence(config, link, sim_times, wall_step=1.0):
    """AutoSequence を疑似時間で回して stats を返す。"""
    stats = HilStats()
    seq = AutoSequence(config, stats)
    now = 0.0
    for t in sim_times:
        seq.update(link, t, now)
        now += wall_step
    return stats, seq


def test_auto_sequence_disabled_by_default():
    link = FakeLink(armed=False)
    stats, seq = run_sequence(HilConfig(device="FAKE"), link, [0.0, 10.0, 20.0])
    assert seq.state == "off"
    assert link.arm_requests == 0 and stats.arm_attempts == 0


def test_auto_sequence_waits_then_arms_then_takes_off():
    link = FakeLink(armed=False, arm_after_requests=2)
    config = HilConfig(device="FAKE", auto_arm=True, takeoff_alt_m=2.5,
                       arm_after_s=5.0, takeoff_after_arm_s=2.0,
                       command_retry_s=1.0, origin_alt_m=488.0)
    stats, seq = run_sequence(config, link, [0.0, 1.0, 4.9, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])

    # arm_after_s より前には要求しない
    assert link.arm_requests == 2                    # 2 回目で armed になる設定
    assert stats.armed_reached and link.armed
    # 離陸高度は AMSL（原点高度 + 相対高度）。gps モードは origin_alt_m 基準
    assert link.takeoff_alts == [pytest.approx(490.5)]
    assert stats.takeoff_commanded and seq.state == "done"
    assert stats.takeoff_result == 0                 # ACCEPTED


def test_takeoff_altitude_uses_zero_reference_in_ev_mode():
    """ev モードは高度基準 0 を使う（488 を渡すと 490m 上昇を命じてしまう）。"""
    link = FakeLink(armed=False, arm_after_requests=1)
    config = HilConfig(device="FAKE", mode="ev", auto_arm=True, takeoff_alt_m=2.5,
                       arm_after_s=0.0, takeoff_after_arm_s=0.0, origin_alt_m=488.0)
    run_sequence(config, link, [0.0] * 6)
    assert link.takeoff_alts == [pytest.approx(2.5)]


def test_auto_sequence_reports_unexpected_disarm(capsys):
    link = FakeLink(armed=False, arm_after_requests=1)
    config = HilConfig(device="FAKE", auto_arm=True, takeoff_alt_m=None, arm_after_s=0.0)
    stats = HilStats()
    seq = AutoSequence(config, stats)
    for now in (0.0, 1.0, 2.0):
        seq.update(link, 0.0, now)
    assert stats.armed_reached
    link.armed = False                      # PX4 側の自動ディスアーム相当
    seq.update(link, 12.0, 3.0)
    assert "ディスアーム" in capsys.readouterr().out


def test_auto_sequence_gives_up_after_max_attempts():
    link = FakeLink(armed=False, arm_after_requests=10**6)   # 決して arm しない
    config = HilConfig(device="FAKE", auto_arm=True, arm_after_s=0.0,
                       command_retry_s=1.0, arm_max_attempts=3)
    stats, seq = run_sequence(config, link, [0.0] * 12)
    assert stats.arm_attempts == 3 and not stats.armed_reached
    assert link.takeoff_alts == [] and seq.state == "done"


def test_auto_sequence_arm_only_does_not_take_off():
    link = FakeLink(armed=False, arm_after_requests=1)
    config = HilConfig(device="FAKE", auto_arm=True, takeoff_alt_m=None,
                       arm_after_s=0.0, takeoff_after_arm_s=0.0)
    stats, seq = run_sequence(config, link, [0.0, 1.0, 2.0, 3.0])
    assert stats.armed_reached and link.takeoff_alts == []
    assert seq.state == "done"


def test_auto_sequence_retries_if_arm_drops():
    link = FakeLink(armed=False, arm_after_requests=1)
    config = HilConfig(device="FAKE", auto_arm=True, takeoff_alt_m=None, arm_after_s=0.0)
    stats = HilStats()
    seq = AutoSequence(config, stats)
    seq.update(link, 0.0, 0.0)          # wait -> arming
    seq.update(link, 0.0, 1.0)          # arm 要求 -> armed
    seq.update(link, 0.0, 2.0)          # arming -> armed 状態へ
    assert seq.state in {"armed", "done"}
    link.armed = False                  # 機体側でディスアームされた
    seq.state = "armed"
    seq.update(link, 0.0, 3.0)
    assert seq.state == "arming"


def test_bridge_disarms_on_exit_when_auto_armed():
    request = SimulationRequest(headless=True, duration_seconds=0.1, hold_until_first_command=False)
    link = FakeLink(controls=[0.2] * 4 + [0.0] * 12, armed=False, arm_after_requests=1)
    config = HilConfig(device="FAKE", auto_arm=True, arm_after_s=0.0, takeoff_alt_m=None)
    run_bridge(config, request, link=link)
    assert link.disarm_requests >= 1 and not link.armed
