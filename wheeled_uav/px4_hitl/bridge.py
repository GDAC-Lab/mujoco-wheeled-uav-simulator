"""HIL main loops: the closed-loop bridge and the staged link check.

Loop shape (run_bridge):
    receive HIL_ACTUATOR_CONTROLS -> map to rotor thrusts -> step physics
    -> extract sensors -> send HIL_SENSOR (+ HIL_GPS or ODOMETRY) -> pace.

The plant stays frozen at the spawn pose until the first controls message is
applied (same rationale as the UDP loop: a wheeled drone tips over during the
connection window otherwise).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import mujoco
import numpy as np

from ..config import build_aerodynamics_config
from ..timing import StepPacer, high_resolution_os_timer
from ..runtime.aerodynamics import AerodynamicsModel
from ..runtime.fidelity import ActuatorModel
from ..runtime.scene import SimulationRequest, load_simulation_scene
from .link import MAV_CMD_COMPONENT_ARM_DISARM, MAV_CMD_NAV_TAKEOFF, MAV_RESULT_TEXT
from .sensors import HilSensorExtractor, gps_from_ned

DEFAULT_ORIGIN_LAT_DEG = 47.397742   # PX4 SITL 既定原点（ツール互換のため踏襲）
DEFAULT_ORIGIN_LON_DEG = 8.545594
DEFAULT_ORIGIN_ALT_M = 488.0


@dataclass
class HilConfig:
    device: str
    baud: int = 921600
    mode: str = "gps"                    # "gps": HIL_GPS 融合 / "ev": ODOMETRY(External Vision) 融合
    sensor_every_n_steps: int = 2        # 物理 2ms なら 250Hz で HIL_SENSOR 送信
    gps_period_s: float = 0.1            # 10Hz
    odom_period_s: float = 0.005         # 200Hz（外部ビジョン入力の典型レート）
    origin_lat_deg: float = DEFAULT_ORIGIN_LAT_DEG
    origin_lon_deg: float = DEFAULT_ORIGIN_LON_DEG
    origin_alt_m: float = DEFAULT_ORIGIN_ALT_M
    mag_ned_gauss: tuple[float, float, float] = (0.21523, 0.00771, 0.42741)
    # PX4 の HIL_ACTUATOR_CONTROLS は Motor1..4 順。シミュレータの ctrl は
    # [fr, fl, br, bl] 順なので、PX4 Quad X 規約 (M1=FR, M2=BL, M3=FL, M4=BR) では
    # ctrl[i] <- controls[motor_map[i]]。
    motor_map: tuple[int, int, int, int] = (0, 2, 3, 1)
    control_exponent: float = 2.0        # 正規化出力 u -> 推力 T = u^exp * ctrl上限（T∝Ω^2, Ω∝u の一次近似）
    control_timeout_s: float = 0.5       # controls 途絶時は全モータ 0（安全側）
    status_period_s: float = 1.0
    forward: str | None = None           # 例 "udpout:127.0.0.1:14550"（QGC 併用）
    # GPS を使わない ev モードでは EKF に全球原点が無く、ホーム位置が確定しない。
    # ホームを要求するモード（AUTO.LOITER 等）では arm が拒否されるため、
    # 起動時に SET_GPS_GLOBAL_ORIGIN と DO_SET_HOME を送る。
    set_origin: bool = True
    origin_period_s: float = 2.0         # ホームが立つまで再送する間隔
    # SET_GPS_GLOBAL_ORIGIN に渡す高度。PX4 は「いまの高度推定値が渡した高度に
    # なる」ようにローカル原点を置き直すため、ev モード（高度基準は ODOMETRY の
    # z で、地面がほぼ 0）で origin_alt_m=488 を渡すと機体が原点の 488m 下にいる
    # ことになり、AUTO.TAKEOFF が 490m 上昇を命じてしまう。
    # None なら ev モードは 0.0、gps モードは origin_alt_m（HIL_GPS と揃える）。
    ekf_origin_alt_m: float | None = None
    # --- RC を使わない自動手順（既定は無効。--arm / --takeoff で有効化）---
    auto_arm: bool = False
    takeoff_alt_m: float | None = None   # None なら arm のみで離陸しない（相対高度 [m]）
    arm_after_s: float = 5.0             # センサ送信開始から arm 要求まで（EKF 収束待ち）
    takeoff_after_arm_s: float = 2.0
    command_retry_s: float = 1.0
    arm_max_attempts: int = 15


@dataclass
class HilStats:
    steps: int = 0
    sensor_sent: int = 0
    gps_sent: int = 0
    odom_sent: int = 0
    controls_received: int = 0
    controls_applied_steps: int = 0
    heartbeat_ok: bool = False
    last_controls: list[float] = field(default_factory=list)
    armed_reached: bool = False
    takeoff_commanded: bool = False
    arm_attempts: int = 0
    last_arm_result: int | None = None
    takeoff_result: int | None = None


def _unhealthy(link) -> list[str]:
    fn = getattr(link, "unhealthy_sensors", None)
    return fn() if callable(fn) else []


def ekf_origin_altitude(config: HilConfig) -> float:
    """SET_GPS_GLOBAL_ORIGIN と AUTO.TAKEOFF が共有する高度基準 [m AMSL]。

    離陸高度は AMSL 指定なので、原点高度と同じ基準で計算しないと
    とんでもない高度を命じることになる。
    """
    if config.ekf_origin_alt_m is not None:
        return config.ekf_origin_alt_m
    return 0.0 if config.mode == "ev" else config.origin_alt_m


class OriginSetter:
    """ホームが確定するまで SET_GPS_GLOBAL_ORIGIN と DO_SET_HOME を送り続ける。

    ev モード（GPS 無し）では EKF に全球原点が無く、ホーム位置が確定しない。
    PX4 の modeCheck は
        home_position_invalid && mode_req_home_position != 0
    で arm を拒否するため、ホームを要求するモード（AUTO.LOITER など）では
    これが無いと永久に arm できない。
    """

    def __init__(self, config: HilConfig):
        self.config = config
        self.next_wall = 0.0
        self.attempts = 0
        self.done = False

    @property
    def altitude_m(self) -> float:
        return ekf_origin_altitude(self.config)

    def update(self, link, now: float) -> None:
        if self.done or not self.config.set_origin:
            return
        # 前回実行のホームが残っていることがあるが、そのときの原点高度は今回と
        # 違うかもしれない。必ず 1 回は送ってから完了判定する。
        if self.attempts > 0 and getattr(link, "home_position", None) is not None:
            self.done = True
            print(f"  [origin] ホーム確定 ({self.attempts} 回送信、原点高度 {self.altitude_m:.1f}m)")
            return
        if now < self.next_wall or not hasattr(link, "set_gps_global_origin"):
            return
        self.attempts += 1
        link.set_gps_global_origin(
            self.config.origin_lat_deg, self.config.origin_lon_deg, self.altitude_m
        )
        link.set_home_here()
        self.next_wall = now + self.config.origin_period_s


def _map_controls_to_ctrl(controls, motor_map, ctrl_hi, exponent) -> np.ndarray:
    ctrl = np.zeros(len(motor_map), dtype=float)
    for i, src in enumerate(motor_map):
        u = min(max(float(controls[src]), 0.0), 1.0)
        ctrl[i] = (u ** exponent) * float(ctrl_hi[i])
    return ctrl


class AutoSequence:
    """RC 無しで arm → 離陸まで進める非ブロッキングの手順実行。

    HIL_SENSOR の送信を止めると EKF2 が落ちるため、コマンドの応答をブロッキングで
    待たない。ループから毎ステップ ``update`` を呼び、未達なら一定間隔で再送する。
    拒否理由は PX4 の STATUSTEXT に出るので、呼び出し側がそれを表示する。
    """

    def __init__(self, config: HilConfig, stats: HilStats):
        self.config = config
        self.stats = stats
        self.state = "wait" if config.auto_arm else "off"
        self.next_attempt_wall = 0.0
        self.armed_wall = 0.0
        self._reported_disarm = False

    def update(self, link, sim_time: float, now: float) -> None:
        cfg = self.config
        if self.state == "off":
            return

        # 一度 arm した後に落ちたら理由の心当たりごと 1 回だけ報告する
        if self.stats.armed_reached and not link.armed and not self._reported_disarm:
            self._reported_disarm = True
            print(f"  [auto] t={sim_time:.1f}s ディスアームされました。"
                  "離陸しないまま COM_DISARM_PRFLT 秒（既定 10s）経つと "
                  "PX4 が自動でディスアームします（正常動作）。")

        if self.state == "takeoff_sent":
            ack = link.take_ack(MAV_CMD_NAV_TAKEOFF) if hasattr(link, "take_ack") else None
            if ack is not None:
                self.stats.takeoff_result = ack
                print(f"  [auto] 離陸指令の結果: {MAV_RESULT_TEXT.get(ack, ack)}")
                self.state = "done"
            return

        if self.state == "done":
            return

        if self.state == "wait":
            if sim_time >= cfg.arm_after_s:
                self.state = "arming"
                self.next_attempt_wall = 0.0
            return

        if self.state == "arming":
            if link.armed:
                self.state = "armed"
                self.armed_wall = now
                self.stats.armed_reached = True
                print(f"  [auto] arm 成功 (t={sim_time:.1f}s, 要求 {self.stats.arm_attempts} 回)")
                return
            ack = link.take_ack(MAV_CMD_COMPONENT_ARM_DISARM) if hasattr(link, "take_ack") else None
            if ack is not None and ack != 0:
                self.stats.last_arm_result = ack
                print(f"  [auto] arm 拒否: {MAV_RESULT_TEXT.get(ack, ack)}"
                      + (f" / 不健全: {', '.join(bad)}" if (bad := _unhealthy(link)) else ""))
            if self.stats.arm_attempts >= cfg.arm_max_attempts:
                print(f"  [auto] arm できませんでした（{cfg.arm_max_attempts} 回）。"
                      "上の PX4: メッセージに拒否理由が出ています。")
                self.state = "done"
                return
            if now >= self.next_attempt_wall:
                self.stats.arm_attempts += 1
                link.arm()
                self.next_attempt_wall = now + cfg.command_retry_s
            return

        if self.state == "armed":
            if not link.armed:                      # 失敗して落ちたら arm からやり直す
                self.state = "arming"
                return
            if cfg.takeoff_alt_m is None:
                self.state = "done"
                return
            if now - self.armed_wall >= cfg.takeoff_after_arm_s:
                target_amsl = ekf_origin_altitude(cfg) + cfg.takeoff_alt_m
                link.takeoff(target_amsl)
                self.stats.takeoff_commanded = True
                print(f"  [auto] 離陸指令 (相対 {cfg.takeoff_alt_m:.1f}m / "
                      f"AMSL {target_amsl:.1f}m)")
                self.state = "takeoff_sent"


def run_bridge(
    config: HilConfig,
    request: SimulationRequest,
    link=None,
    viewer_handle_factory=None,
) -> HilStats:
    """Run the closed HIL loop until the requested duration (or viewer close)."""
    scene = load_simulation_scene(request)
    if scene.request.num_uavs != 1:
        raise ValueError("HIL bridge は num_uavs=1 のみ対応です")
    model, data = scene.model, scene.data

    actuator_model = ActuatorModel(model, scene.fidelity, float(scene.params["actuation"]["thrust_coefficient"]))
    aerodynamics_model = AerodynamicsModel(
        model, scene.params, build_aerodynamics_config(scene.params), scene.uav_specs, scene.sensor_layouts
    )
    extractor = HilSensorExtractor(
        model,
        scene.uav_specs[0].sensor_names,
        mag_ned_gauss=config.mag_ned_gauss,
        origin_alt_m=config.origin_alt_m,
    )
    ctrl_hi = np.array(model.actuator_ctrlrange[:, 1], dtype=float)

    own_link = link is None
    if own_link:
        from .link import Px4HilLink

        link = Px4HilLink(config.device, baud=config.baud)
        if config.forward:
            link.enable_forward(config.forward)
    stats = HilStats()

    try:
        heartbeat = link.wait_heartbeat()
        stats.heartbeat_ok = True
        print(f"HEARTBEAT: system={heartbeat['system']} component={heartbeat['component']} "
              f"armed={link.armed}")
        print(f"mode={config.mode} sensor_rate={1.0 / (model.opt.timestep * config.sensor_every_n_steps):.0f}Hz "
              f"motor_map={list(config.motor_map)}")

        timestep = float(model.opt.timestep)
        duration = request.duration_seconds
        initial_qpos = data.qpos.copy()
        next_gps_time = 0.0
        next_odom_time = 0.0
        next_status_wall = time.monotonic() + config.status_period_s
        viewer = viewer_handle_factory(model, data) if viewer_handle_factory is not None else None
        sequence = AutoSequence(config, stats)
        origin_setter = OriginSetter(config)
        if config.auto_arm:
            print(f"auto: {config.arm_after_s:.0f}s 後に arm 要求"
                  + (f" → {config.takeoff_after_arm_s:.0f}s 後に離陸 (相対 {config.takeoff_alt_m:.1f}m)"
                     if config.takeoff_alt_m is not None else "（離陸はしない）"))

        with high_resolution_os_timer():
            pacer = StepPacer(timestep)
            while duration is None or data.time < duration:
                if viewer is not None and not viewer.is_running():
                    break
                link.pump()

                fresh = (
                    link.latest_controls is not None
                    and (time.monotonic() - link.latest_controls_wall_time) <= config.control_timeout_s
                )
                if fresh:
                    data.ctrl[:] = _map_controls_to_ctrl(
                        link.latest_controls, config.motor_map, ctrl_hi, config.control_exponent
                    )
                    stats.controls_applied_steps += 1
                else:
                    data.ctrl[:] = 0.0

                actuator_model.apply(data)
                aerodynamics_model.apply(data)
                mujoco.mj_step(model, data)
                if not fresh and stats.controls_applied_steps == 0:
                    # 最初の指令が届くまでスポーン姿勢で保持（UDP ループと同じ方針）
                    data.qpos[:] = initial_qpos
                    data.qvel[:] = 0.0
                    mujoco.mj_forward(model, data)
                    extractor.reset()
                stats.steps += 1

                if stats.steps % config.sensor_every_n_steps == 0:
                    sample = extractor.sample(data)
                    link.send_hil_sensor(sample)
                    stats.sensor_sent += 1

                    if config.mode == "gps" and data.time >= next_gps_time:
                        gps = gps_from_ned(
                            sample.pos_ned, sample.vel_ned,
                            config.origin_lat_deg, config.origin_lon_deg, config.origin_alt_m,
                        )
                        link.send_hil_gps(sample.time_usec, gps)
                        stats.gps_sent += 1
                        next_gps_time = data.time + config.gps_period_s
                    elif config.mode == "ev" and data.time >= next_odom_time:
                        link.send_odometry(sample)
                        stats.odom_sent += 1
                        next_odom_time = data.time + config.odom_period_s

                if viewer is not None and stats.steps % 20 == 0:
                    viewer.sync()

                now = time.monotonic()
                origin_setter.update(link, now)
                sequence.update(link, data.time, now)
                if now >= next_status_wall:
                    hb_age = now - link.last_heartbeat_wall_time if link.last_heartbeat_wall_time else float("inf")
                    alt = -float(data.qpos[2]) if len(data.qpos) > 2 else float("nan")
                    # PX4 が出しているモータ指令（0..1）と、それを推力へ変換した結果。
                    # 「浮かない」ときに PX4 側が出していないのか、変換が弱いのかを
                    # ここだけで切り分けられる。
                    u = link.latest_controls[:4] if link.latest_controls else [float("nan")] * 4
                    print(
                        f"t={data.time:7.2f}s armed={int(link.armed)} mode={getattr(link, 'mode_text', '-')} "
                        f"z_ned={alt:6.2f}m hb_age={hb_age:4.1f}s "
                        f"u=[{' '.join(f'{v:.2f}' for v in u)}] "
                        f"T={float(np.sum(data.ctrl)):5.1f}N "
                        f"ctrl_msgs={link.controls_count:6d}"
                    )
                    for text in link.statustexts[-3:]:
                        print(f"  PX4: {text}")
                    link.statustexts.clear()
                    next_status_wall = now + config.status_period_s

                pacer.pace()
    finally:
        stats.controls_received = getattr(link, "controls_count", 0)
        if getattr(link, "latest_controls", None):
            stats.last_controls = list(link.latest_controls[:8])
        # 自動 arm した場合は必ずディスアームして終わる（次回起動時に armed で
        # 始まってしまうのを防ぐ。HITL なので実モータは元から止まっている）
        if config.auto_arm and getattr(link, "armed", False):
            try:
                link.disarm(force=True)
                for _ in range(20):
                    link.pump()
                    if not link.armed:
                        break
                    time.sleep(0.05)
                print(f"終了時ディスアーム: armed={int(link.armed)}")
            except Exception as exc:      # 通信が切れている場合など
                print(f"終了時ディスアームに失敗: {exc}")
        if own_link:
            link.close()

    return stats


def run_diagnose(config: HilConfig, request: SimulationRequest, link=None, seconds: float = 15.0) -> HilStats:
    """arm できない原因を PX4 自身に報告させる。

    静止センサを送り続けながら、SYS_STATUS（センサ健全性）・ESTIMATOR_STATUS
    （EKF2 がどこまで有効か）を集め、最後に arm を 1 回要求して COMMAND_ACK の
    result を表示する。PX4 v1.15 は arm 拒否理由を STATUSTEXT ではなく EVENT で
    送るため、STATUSTEXT だけ見ていても理由が分からないことへの対処。
    """
    scene = load_simulation_scene(request)
    model, data = scene.model, scene.data
    mujoco.mj_forward(model, data)
    extractor = HilSensorExtractor(
        model, scene.uav_specs[0].sensor_names,
        mag_ned_gauss=config.mag_ned_gauss, origin_alt_m=config.origin_alt_m,
    )

    own_link = link is None
    if own_link:
        from .link import Px4HilLink

        link = Px4HilLink(config.device, baud=config.baud)
        if config.forward:
            link.enable_forward(config.forward)
    stats = HilStats()

    try:
        link.wait_heartbeat()
        stats.heartbeat_ok = True
        sample = extractor.sample(data)
        period = model.opt.timestep * config.sensor_every_n_steps
        print(f"{seconds:.0f} 秒センサを送りながら PX4 の状態を集めます...")
        # 見たい telemetry を明示的に要求する（既定ストリームに無い場合の保険）
        for msg_id, hz in ((24, 2.0), (32, 2.0), (230, 2.0), (1, 2.0)):   # GPS_RAW_INT / LOCAL_POSITION_NED / ESTIMATOR_STATUS / SYS_STATUS
            if hasattr(link, "request_message_interval"):
                link.request_message_interval(msg_id, hz)

        start = time.monotonic()
        arm_sent_at = None
        next_report = start + 5.0
        origin_setter = OriginSetter(config)
        with high_resolution_os_timer():
            while (elapsed := time.monotonic() - start) < seconds:
                link.pump()
                origin_setter.update(link, time.monotonic())
                now_sample = type(sample)(
                    time_usec=int(elapsed * 1e6),
                    accel_frd=sample.accel_frd, gyro_frd=sample.gyro_frd, mag_frd=sample.mag_frd,
                    abs_pressure_hpa=sample.abs_pressure_hpa, pressure_alt_m=sample.pressure_alt_m,
                    pos_ned=sample.pos_ned, vel_ned=sample.vel_ned, quat_ned=sample.quat_ned,
                )
                link.send_hil_sensor(now_sample)
                stats.sensor_sent += 1
                if config.mode == "gps" and stats.sensor_sent % 50 == 0:
                    link.send_hil_gps(now_sample.time_usec, gps_from_ned(
                        sample.pos_ned, sample.vel_ned,
                        config.origin_lat_deg, config.origin_lon_deg, config.origin_alt_m))
                    stats.gps_sent += 1
                elif config.mode == "ev" and stats.sensor_sent % 2 == 0:   # 250Hz
                    link.send_odometry(now_sample)
                    stats.odom_sent += 1
                if arm_sent_at is None and elapsed >= seconds - 8.0:
                    link.arm()
                    stats.arm_attempts += 1
                    arm_sent_at = elapsed
                    print(f"[{elapsed:5.1f}s] 通常 arm を要求")
                elif arm_sent_at is not None and elapsed >= arm_sent_at + 3.0 and stats.arm_attempts == 1:
                    normal = link.take_ack(MAV_CMD_COMPONENT_ARM_DISARM)
                    print(f"[{elapsed:5.1f}s] 通常 arm の結果: "
                          f"{MAV_RESULT_TEXT.get(normal, normal) if normal is not None else '応答なし'}"
                          " -> 強制 arm を要求（HITL は実出力が無いので安全）")
                    link.arm(force=True)
                    stats.arm_attempts += 1
                if time.monotonic() >= next_report:
                    _print_health(link)
                    next_report = time.monotonic() + 5.0
                time.sleep(period)

        print("\n===== 診断結果 =====")
        print(f"HEARTBEAT   : armed={int(link.armed)} mode={getattr(link, 'mode_text', '-')}")
        print(f"controls    : {link.controls_count} 通受信")
        _print_health(link, prefix="")
        result = link.take_ack(MAV_CMD_COMPONENT_ARM_DISARM)
        stats.last_arm_result = result
        stats.armed_reached = bool(link.armed)
        if result is None:
            print("arm ACK     : 返ってこず（コマンドが届いていない可能性）")
        else:
            print(f"arm ACK     : {MAV_RESULT_TEXT.get(result, result)}")
        if getattr(link, "events", None):
            print(f"EVENT       : id={sorted(set(link.events))[:20]}")
        counts = getattr(link, "msg_counts", {})
        if counts:
            top = sorted(counts.items(), key=lambda kv: -kv[1])[:12]
            print("受信メッセージ: " + ", ".join(f"{k}={v}" for k, v in top))
        for text in link.statustexts[-10:]:
            print(f"  PX4: {text}")
    finally:
        # 診断で arm できてしまった場合は必ず戻す（次回 armed で始まらないように）
        if getattr(link, "armed", False):
            try:
                link.disarm(force=True)
                for _ in range(20):
                    link.pump()
                    if not link.armed:
                        break
                    time.sleep(0.05)
                print(f"終了時ディスアーム: armed={int(link.armed)}")
            except Exception as exc:
                print(f"終了時ディスアームに失敗: {exc}")
        if own_link:
            link.close()
    return stats


def _print_health(link, prefix: str = "  ") -> None:
    status = getattr(link, "sys_status", None)
    if status is None:
        print(f"{prefix}SYS_STATUS  : 未受信")
    else:
        bad = link.unhealthy_sensors()
        print(f"{prefix}不健全      : {', '.join(bad) if bad else '(なし)'}")
    flags = getattr(link, "estimator_flags", None)
    if flags is None:
        print(f"{prefix}EKF2        : ESTIMATOR_STATUS 未受信")
    else:
        print(f"{prefix}EKF2 有効   : {', '.join(link.estimator_ok()) or '(なし)'}")
    gps = getattr(link, "gps_raw", None)
    if gps is None:
        print(f"{prefix}GPS_RAW_INT : 未受信（PX4 が GPS を持っていない）")
    else:
        print(f"{prefix}GPS_RAW_INT : fix={gps['fix_type']} sats={gps['sats']} "
              f"eph={gps['eph_cm'] / 100:.1f}m lat={gps['lat'] * 1e-7:.6f} lon={gps['lon'] * 1e-7:.6f}")
    pos = getattr(link, "local_pos", None)
    if pos is not None:
        print(f"{prefix}LOCAL_POS   : x={pos[0]:.2f} y={pos[1]:.2f} z={pos[2]:.2f}")
    origin = getattr(link, "global_origin", None)
    home = getattr(link, "home_position", None)
    print(f"{prefix}全球原点    : "
          + (f"{origin[0] * 1e-7:.6f}, {origin[1] * 1e-7:.6f}" if origin else "未設定"))
    print(f"{prefix}ホーム位置  : "
          + (f"{home[0] * 1e-7:.6f}, {home[1] * 1e-7:.6f}  -> arm 可" if home
             else "**未確定**（ホームを要求するモードでは arm できない）"))


def run_check_link(config: HilConfig, request: SimulationRequest, link=None, seconds: float = 5.0) -> HilStats:
    """Stage 1+2 の疎通確認: HEARTBEAT 受信 → 静止センサ送信 → controls 受信数を報告。

    物理は進めない（スポーン姿勢の静止値を送るだけ）。合格基準:
    - HEARTBEAT が取れる
    - HIL_ACTUATOR_CONTROLS が 1 通以上返る（SYS_HITL=1 でセンサ受信が始まると
      PX4 が送出する。disarmed では値はアイドル/0 で正常）
    """
    scene = load_simulation_scene(request)
    model, data = scene.model, scene.data
    mujoco.mj_forward(model, data)
    extractor = HilSensorExtractor(
        model, scene.uav_specs[0].sensor_names,
        mag_ned_gauss=config.mag_ned_gauss, origin_alt_m=config.origin_alt_m,
    )

    own_link = link is None
    if own_link:
        from .link import Px4HilLink

        link = Px4HilLink(config.device, baud=config.baud)
        if config.forward:
            link.enable_forward(config.forward)
    stats = HilStats()

    try:
        heartbeat = link.wait_heartbeat()
        stats.heartbeat_ok = True
        print(f"[1/3] HEARTBEAT OK: system={heartbeat['system']} component={heartbeat['component']}")

        sample = extractor.sample(data)   # 静止値（accel_frd ≈ (0,0,-9.81)）
        print(f"[2/3] 静止センサ値: accel_frd={np.round(sample.accel_frd, 2).tolist()} "
              f"abs_p={sample.abs_pressure_hpa:.1f}hPa  ({seconds:.0f}s 送信します...)")
        start = time.monotonic()
        period = model.opt.timestep * config.sensor_every_n_steps
        next_gps = 0.0
        with high_resolution_os_timer():
            while (elapsed := time.monotonic() - start) < seconds:
                link.pump()
                sample_now = type(sample)(
                    time_usec=int(elapsed * 1e6),
                    accel_frd=sample.accel_frd, gyro_frd=sample.gyro_frd, mag_frd=sample.mag_frd,
                    abs_pressure_hpa=sample.abs_pressure_hpa, pressure_alt_m=sample.pressure_alt_m,
                    pos_ned=sample.pos_ned, vel_ned=sample.vel_ned, quat_ned=sample.quat_ned,
                )
                link.send_hil_sensor(sample_now)
                stats.sensor_sent += 1
                if config.mode == "gps" and elapsed >= next_gps:
                    gps = gps_from_ned(sample.pos_ned, sample.vel_ned,
                                       config.origin_lat_deg, config.origin_lon_deg, config.origin_alt_m)
                    link.send_hil_gps(sample_now.time_usec, gps)
                    stats.gps_sent += 1
                    next_gps = elapsed + config.gps_period_s
                time.sleep(period)

        stats.controls_received = link.controls_count
        ok = stats.controls_received > 0
        print(f"[3/3] HIL_ACTUATOR_CONTROLS 受信: {stats.controls_received} 通 -> {'OK' if ok else 'NG'}")
        if link.latest_controls is not None:
            stats.last_controls = list(link.latest_controls[:8])
            print(f"      最新 controls[0:4]={[round(v, 3) for v in link.latest_controls[:4]]} (disarmed なら 0/アイドル)")
        if not ok:
            print("      切り分け順:")
            print("        1. HIL_ACT_FUNC1 が読めるか（読めなければ pwm_out_sim を含まないファーム。")
            print("           HIL 対応ビルドのファームウェアを焼き直す。これが最頻の原因）")
            print("        2. SYS_HITL=1 かつ再起動済みか（SYS_HITL は reboot_required）")
            print("        3. PX4 の NSH で `mavlink status` → compid:51 からの受信が増えているか")
            print("        4. QGC が同じ COM ポートを掴んでいないか")
        for text in link.statustexts[-5:]:
            print(f"  PX4: {text}")
    finally:
        if own_link:
            link.close()
    return stats
