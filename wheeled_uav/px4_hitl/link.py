"""MAVLink link to a Pixhawk in HITL mode (pymavlink, lazy import).

Sends HIL_SENSOR / HIL_GPS / ODOMETRY, receives HIL_ACTUATOR_CONTROLS and
HEARTBEAT. pymavlink is an optional dependency (``uv sync --extra hil``); the
import happens on connection so the rest of the simulator never needs it.
"""

from __future__ import annotations

import os
import re
import time

import numpy as np

MAV_FRAME_LOCAL_FRD = 20
MAV_FRAME_BODY_FRD = 12
MAV_ESTIMATOR_TYPE_VISION = 2
MAV_MODE_FLAG_SAFETY_ARMED = 128
MAV_MODE_FLAG_CUSTOM_MODE_ENABLED = 1
HIL_SENSOR_FIELDS_ALL = 0x1FFF
NAN = float("nan")

MAV_CMD_NAV_TAKEOFF = 22
MAV_CMD_NAV_LAND = 21
MAV_CMD_DO_SET_MODE = 176
MAV_CMD_COMPONENT_ARM_DISARM = 400

# PX4 独自モード（HEARTBEAT.custom_mode の上位バイト）
PX4_CUSTOM_MAIN_MODE_AUTO = 4
PX4_CUSTOM_SUB_MODE_AUTO_TAKEOFF = 2
PX4_CUSTOM_SUB_MODE_AUTO_LOITER = 3
PX4_CUSTOM_SUB_MODE_AUTO_LAND = 6

MAV_RESULT_TEXT = {
    0: "ACCEPTED", 1: "TEMPORARILY_REJECTED", 2: "DENIED",
    3: "UNSUPPORTED", 4: "FAILED", 5: "IN_PROGRESS", 6: "CANCELLED",
}

# MAV_SYS_STATUS_SENSOR: SYS_STATUS のビット。arm 拒否の原因特定に使う。
SYS_STATUS_SENSOR_BITS = [
    (0x00000001, "3D_GYRO"), (0x00000002, "3D_ACCEL"), (0x00000004, "3D_MAG"),
    (0x00000008, "ABSOLUTE_PRESSURE"), (0x00000010, "DIFFERENTIAL_PRESSURE"),
    (0x00000020, "GPS"), (0x00000040, "OPTICAL_FLOW"), (0x00000080, "VISION_POSITION"),
    (0x00000100, "LASER_POSITION"), (0x00000200, "EXTERNAL_GROUND_TRUTH"),
    (0x00000400, "ANGULAR_RATE_CONTROL"), (0x00000800, "ATTITUDE_STABILIZATION"),
    (0x00001000, "YAW_POSITION"), (0x00002000, "Z_ALTITUDE_CONTROL"),
    (0x00004000, "XY_POSITION_CONTROL"), (0x00008000, "MOTOR_OUTPUTS"),
    (0x00010000, "RC_RECEIVER"), (0x00020000, "3D_GYRO2"), (0x00040000, "3D_ACCEL2"),
    (0x00080000, "3D_MAG2"), (0x00100000, "GEOFENCE"), (0x00200000, "AHRS"),
    (0x00400000, "TERRAIN"), (0x00800000, "REVERSE_MOTOR"), (0x01000000, "LOGGING"),
    (0x02000000, "BATTERY"), (0x04000000, "PROXIMITY"), (0x08000000, "SATCOM"),
    (0x10000000, "PREARM_CHECK"), (0x20000000, "OBSTACLE_AVOIDANCE"),
    (0x40000000, "PROPULSION"),
]

# ESTIMATOR_STATUS_FLAGS: EKF2 がどこまで有効かを示す。
ESTIMATOR_STATUS_BITS = [
    (0x0001, "ATTITUDE"), (0x0002, "VELOCITY_HORIZ"), (0x0004, "VELOCITY_VERT"),
    (0x0008, "POS_HORIZ_REL"), (0x0010, "POS_HORIZ_ABS"), (0x0020, "POS_VERT_ABS"),
    (0x0040, "POS_VERT_AGL"), (0x0080, "CONST_POS_MODE"), (0x0100, "PRED_POS_HORIZ_REL"),
    (0x0200, "PRED_POS_HORIZ_ABS"), (0x0400, "GPS_GLITCH"), (0x0800, "ACCEL_ERROR"),
]


def decode_bits(value: int, table) -> list[str]:
    return [name for bit, name in table if value & bit]


class Px4HilLink:
    """Serial (USB) MAVLink connection to the autopilot."""

    def __init__(self, device: str, baud: int = 921600, dialect: str = "common"):
        # pymavlink は既定で MAVLink 1（WIRE_PROTOCOL 1.0）で動く。MAVLink 1 は
        # メッセージ ID を 255 までしか扱えないため、そのままだと
        #   - EVENT(410) を復号できない   -> PX4 の arm 拒否理由が一切見えない
        #   - ODOMETRY(331) を送れない    -> ev モード（外部視覚融合）が成立しない
        # という状態になる。mavutil.set_dialect は os.environ["MAVLINK20"] を見て
        # v20 の方言を読み込むので、import 順に依存せずここで強制できる。
        os.environ["MAVLINK20"] = "1"
        try:
            from pymavlink import mavutil
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "pymavlink が見つかりません。`uv sync --extra hil` を実行してください。"
            ) from exc
        mavutil.set_dialect(dialect)
        self._mavutil = mavutil
        # source_system は機体と同じ 1（PX4 の HIL はシミュレータを同一システムとして扱う）
        #
        # Windows の COM ポートは、他プロセス（QGC 等）が掴んでいると
        # os.path.isfile("COM4") が True を返すことがある。すると
        # mavlink_connection が「ログファイル」と誤判定し、
        #   PermissionError: [Errno 13] Permission denied: 'COM4'
        # という原因の分かりにくい例外になる。COM/tty は明示的にシリアルで開く。
        if re.match(r"^(COM\d+$|/dev/tty)", device, re.IGNORECASE):
            try:
                self.conn = mavutil.mavserial(
                    device, baud=baud, source_system=1, source_component=51
                )
            except Exception as exc:
                raise RuntimeError(
                    f"{device} を開けませんでした: {exc}\n"
                    "QGroundControl など他のアプリが同じ COM ポートを掴んでいませんか。"
                    "COM ポートは 1 プロセスしか掴めません（QGC を併用したい場合は --forward）。"
                ) from exc
        else:
            self.conn = mavutil.mavlink_connection(
                device, baud=baud, source_system=1, source_component=51
            )
        self.protocol_version = str(self.conn.WIRE_PROTOCOL_VERSION)
        if self.protocol_version != "2.0":   # pragma: no cover - 環境依存
            raise RuntimeError(
                f"MAVLink {self.protocol_version} で接続されました。"
                "MAVLink 2 でないと EVENT(410)/ODOMETRY(331) を扱えません。"
            )

        self.latest_controls: list[float] | None = None
        self.latest_controls_wall_time: float = 0.0
        self.controls_count = 0
        self.last_heartbeat_wall_time: float = 0.0
        self.armed = False
        self.custom_mode = 0
        self.statustexts: list[str] = []
        self.command_acks: dict[int, int] = {}
        self.sys_status: dict[str, int] | None = None
        self.estimator_flags: int | None = None
        self.events: list[int] = []
        self.gps_raw: dict[str, int] | None = None
        self.local_pos: tuple[float, float, float] | None = None
        self.global_origin: tuple[int, int, int] | None = None
        self.home_position: tuple[int, int, int] | None = None
        self.msg_counts: dict[str, int] = {}
        self._forward = None
        self.forwarded_to_gcs = 0
        self.forwarded_from_gcs = 0

    def enable_forward(self, target: str = "udpout:127.0.0.1:14550") -> None:
        """MAVLink を UDP へ中継し、QGC をブリッジと同時に使えるようにする。

        USB の COM ポートは 1 プロセスしか掴めないため、QGC を繋ぐとブリッジが
        動かせない。ブリッジを素通しの中継にすることで、HITL を回したまま QGC の
        表示（arm 拒否理由の EVENT デコードなど）が使える。

        QGC 側は既定の自動接続（UDP 14550 待ち受け）でそのまま繋がる。
        """
        self._forward = self._mavutil.mavlink_connection(target, source_system=1, source_component=52)
        print(f"MAVLink 中継: {target}（QGC を起動すると自動接続します）")

    def wait_heartbeat(self, timeout_s: float = 10.0) -> dict:
        msg = self.conn.wait_heartbeat(timeout=timeout_s)
        if msg is None:
            raise TimeoutError(
                "HEARTBEAT を受信できませんでした。COM ポート・ボーレート・"
                "QGC が同じポートを掴んでいないかを確認してください。"
            )
        self.last_heartbeat_wall_time = time.monotonic()
        self.armed = bool(int(getattr(msg, "base_mode", 0)) & MAV_MODE_FLAG_SAFETY_ARMED)
        return {
            "system": int(self.conn.target_system),
            "component": int(self.conn.target_component),
            "autopilot": int(getattr(msg, "autopilot", -1)),
            "base_mode": int(getattr(msg, "base_mode", 0)),
        }

    def _pump_forward(self) -> None:
        """QGC(UDP) からの入力を機体へ素通しする。"""
        if self._forward is None:
            return
        while True:
            try:
                msg = self._forward.recv_match(blocking=False)
            except Exception:
                return
            if msg is None or msg.get_type() == "BAD_DATA":
                return
            try:
                self.conn.write(msg.get_msgbuf())
                self.forwarded_from_gcs += 1
            except Exception:
                return

    def pump(self) -> None:
        """Drain incoming messages without blocking; update latest state."""
        self._pump_forward()
        while True:
            msg = self.conn.recv_match(blocking=False)
            if msg is None:
                return
            kind = msg.get_type()
            if self._forward is not None and kind != "BAD_DATA":
                try:
                    self._forward.write(msg.get_msgbuf())
                    self.forwarded_to_gcs += 1
                except Exception:
                    pass
            self.msg_counts[kind] = self.msg_counts.get(kind, 0) + 1
            if kind == "HIL_ACTUATOR_CONTROLS":
                self.latest_controls = [float(v) for v in msg.controls]
                self.latest_controls_wall_time = time.monotonic()
                self.controls_count += 1
            elif kind == "HEARTBEAT":
                self.last_heartbeat_wall_time = time.monotonic()
                self.armed = bool(int(getattr(msg, "base_mode", 0)) & MAV_MODE_FLAG_SAFETY_ARMED)
                self.custom_mode = int(getattr(msg, "custom_mode", 0))
            elif kind == "STATUSTEXT":
                text = str(getattr(msg, "text", "")).strip()
                if text:
                    self.statustexts.append(text)
            elif kind == "COMMAND_ACK":
                self.command_acks[int(msg.command)] = int(msg.result)
            elif kind == "SYS_STATUS":
                self.sys_status = {
                    "present": int(msg.onboard_control_sensors_present),
                    "enabled": int(msg.onboard_control_sensors_enabled),
                    "health": int(msg.onboard_control_sensors_health),
                }
            elif kind == "ESTIMATOR_STATUS":
                self.estimator_flags = int(msg.flags)
            elif kind == "GPS_RAW_INT":
                self.gps_raw = {
                    "fix_type": int(msg.fix_type),
                    "sats": int(msg.satellites_visible),
                    "eph_cm": int(msg.eph),
                    "lat": int(msg.lat),
                    "lon": int(msg.lon),
                }
            elif kind == "LOCAL_POSITION_NED":
                self.local_pos = (float(msg.x), float(msg.y), float(msg.z))
            elif kind == "GPS_GLOBAL_ORIGIN":
                self.global_origin = (int(msg.latitude), int(msg.longitude), int(msg.altitude))
            elif kind == "HOME_POSITION":
                self.home_position = (int(msg.latitude), int(msg.longitude), int(msg.altitude))
            elif kind == "EVENT":
                # PX4 v1.15 は arm 拒否の理由を STATUSTEXT ではなく EVENT で送る。
                # 名前の復元にはイベント定義 json が要るので、ここでは ID を残す。
                self.events.append(int(getattr(msg, "id", 0)))

    def unhealthy_sensors(self) -> list[str]:
        """enabled なのに health が落ちている項目（arm を止めている候補）。"""
        if self.sys_status is None:
            return []
        bad = self.sys_status["enabled"] & ~self.sys_status["health"]
        return decode_bits(bad, SYS_STATUS_SENSOR_BITS)

    def estimator_ok(self) -> list[str]:
        if self.estimator_flags is None:
            return []
        return decode_bits(self.estimator_flags, ESTIMATOR_STATUS_BITS)

    # ---------------- コマンド送信（RC が無くても arm・離陸できるようにする） ----------------
    #
    # PX4 は MAVLink 経由のコマンドを常に from_external=true として扱うため
    # （mavlink_receiver.cpp）、送信元 sysid がヴィークルと同じ 1 でも通常どおり
    # 処理され、プリフライトチェックも走る。拒否された理由は STATUSTEXT に出る。

    def send_command_long(self, command: int, *params: float) -> None:
        values = [float(v) for v in params] + [0.0] * (7 - len(params))
        self.command_acks.pop(command, None)
        self.conn.mav.command_long_send(
            self.conn.target_system, self.conn.target_component,
            command, 0, *values[:7],
        )

    def arm(self, force: bool = False) -> None:
        """アーム要求。HITL では実モータ出力が無効なので機体は動かない。

        force=True（21196）はプリフライト検査をスキップする。実機飛行では
        使ってはいけないが、HITL では rcS が dshot/pwm_out を起動しないため
        物理的な出力が存在せず、SITL の ``commander arm -f`` と同じ扱いでよい。
        """
        self.send_command_long(MAV_CMD_COMPONENT_ARM_DISARM, 1.0, 21196.0 if force else 0.0)

    def disarm(self, force: bool = False) -> None:
        # 21196 = 強制ディスアーム（飛行中でも受け付ける magic number）
        self.send_command_long(MAV_CMD_COMPONENT_ARM_DISARM, 0.0, 21196.0 if force else 0.0)

    def takeoff(self, altitude_amsl_m: float) -> None:
        """AUTO.TAKEOFF 要求。lat/lon は NaN で「現在位置」を意味する。"""
        self.send_command_long(
            MAV_CMD_NAV_TAKEOFF, 0.0, 0.0, 0.0, NAN, NAN, NAN, float(altitude_amsl_m)
        )

    def land(self) -> None:
        self.send_command_long(MAV_CMD_NAV_LAND, 0.0, 0.0, 0.0, NAN, NAN, NAN, 0.0)

    def set_gps_global_origin(self, lat_deg: float, lon_deg: float, alt_m: float) -> None:
        """EKF に全球原点を与える（SET_GPS_GLOBAL_ORIGIN）。

        GPS を使わない屋内（モーキャプ / ev モード）では EKF が全球原点を持てず、
        ホーム位置が確定しない。すると modeCheck が
        ``home_position_invalid && mode_req_home_position`` で arm を拒否する
        （AUTO.LOITER などホームを要求するモードのとき）。
        """
        self.conn.mav.set_gps_global_origin_send(
            self.conn.target_system,
            int(round(lat_deg * 1e7)),
            int(round(lon_deg * 1e7)),
            int(round(alt_m * 1e3)),
        )

    def set_home_here(self) -> None:
        """MAV_CMD_DO_SET_HOME param1=1（現在位置をホームにする）。"""
        self.send_command_long(179, 1.0)

    def request_message_interval(self, message_id: int, hz: float) -> None:
        """MAV_CMD_SET_MESSAGE_INTERVAL。診断で見たい telemetry を確実に流させる。"""
        self.send_command_long(511, float(message_id), 1e6 / max(hz, 0.1))

    def set_mode(self, main_mode: int, sub_mode: int = 0) -> None:
        self.send_command_long(
            MAV_CMD_DO_SET_MODE, float(MAV_MODE_FLAG_CUSTOM_MODE_ENABLED),
            float(main_mode), float(sub_mode),
        )

    def take_ack(self, command: int) -> int | None:
        """届いていれば COMMAND_ACK の result を取り出す（消費する）。"""
        return self.command_acks.pop(command, None)

    @property
    def mode_text(self) -> str:
        main = (self.custom_mode >> 16) & 0xFF
        sub = (self.custom_mode >> 24) & 0xFF
        return f"{main}.{sub}"

    def send_hil_sensor(self, sample) -> None:
        self.conn.mav.hil_sensor_send(
            int(sample.time_usec),
            float(sample.accel_frd[0]), float(sample.accel_frd[1]), float(sample.accel_frd[2]),
            float(sample.gyro_frd[0]), float(sample.gyro_frd[1]), float(sample.gyro_frd[2]),
            float(sample.mag_frd[0]), float(sample.mag_frd[1]), float(sample.mag_frd[2]),
            float(sample.abs_pressure_hpa),
            0.0,                                    # diff_pressure
            float(sample.pressure_alt_m),
            20.0,                                   # temperature [degC]
            HIL_SENSOR_FIELDS_ALL,
            0,                                      # sensor id
        )

    def send_hil_gps(self, time_usec: int, gps: dict) -> None:
        self.conn.mav.hil_gps_send(
            int(time_usec),
            3,                                      # fix_type: 3D
            gps["lat"], gps["lon"], gps["alt_mm"],
            100, 100,                               # eph/epv [cm]
            gps["vel_cm"],
            gps["vn_cm"], gps["ve_cm"], gps["vd_cm"],
            gps["cog_cdeg"],
            12,                                     # satellites_visible
        )

    def send_odometry(self, sample) -> None:
        """Ground-truth pose as External Vision (indoor mocap-equivalent mode).

        クォータニオンは送信前に必ず正規化する（EKF2 はノルムゲート
        |1-||q|| <= 1e-5 を外れた ODOMETRY を破棄するため）。
        """
        q = np.asarray(sample.quat_ned, dtype=float)
        q = q / np.linalg.norm(q)
        self.conn.mav.odometry_send(
            int(sample.time_usec),
            MAV_FRAME_LOCAL_FRD,
            MAV_FRAME_BODY_FRD,
            float(sample.pos_ned[0]), float(sample.pos_ned[1]), float(sample.pos_ned[2]),
            [float(q[0]), float(q[1]), float(q[2]), float(q[3])],
            NAN, NAN, NAN,                          # vx, vy, vz
            NAN, NAN, NAN,                          # rollspeed, pitchspeed, yawspeed
            [NAN] * 21,                             # pose covariance: 全要素 NaN（未使用の明示）
            [NAN] * 21,                             # velocity covariance
            0,                                      # reset_counter
            MAV_ESTIMATOR_TYPE_VISION,
            100,                                    # quality [%]
        )

    def close(self) -> None:
        for conn in (self._forward, self.conn):
            try:
                if conn is not None:
                    conn.close()
            except Exception:
                pass
