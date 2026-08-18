"""CLI: PX4 HITL bridge.

段階検証:
  Stage 1+2  uv run --extra hil python -m wheeled_uav.px4_hitl --device COM4 --check-link
  Stage 3    uv run --extra hil python -m wheeled_uav.px4_hitl --device COM4 --viewer --arm
  Stage 4    uv run --extra hil python -m wheeled_uav.px4_hitl --device COM4 --viewer --takeoff 2.5
  EV モード  uv run --extra hil python -m wheeled_uav.px4_hitl --device COM4 --mode ev --takeoff 1.0

--arm / --takeoff は RC が無くても MAVLink 経由で arm・離陸させるためのもの。
HITL では実モータ出力が無効化されているので実機は動かないが、念のため
プロペラを外した状態で行うこと。
"""

from __future__ import annotations

import argparse

from .bridge import HilConfig, run_bridge, run_check_link, run_diagnose
from ..runtime.scene import SimulationRequest


def main() -> int:
    parser = argparse.ArgumentParser(prog="wheeled_uav.px4_hitl", description="PX4 HITL bridge (MuJoCo plant)")
    parser.add_argument("--device", required=True, help="Pixhawk の COM ポート (例: COM5)")
    parser.add_argument("--baud", type=int, default=921600)
    parser.add_argument("--check-link", action="store_true", help="Stage1+2: 疎通確認のみ（物理は進めない）")
    parser.add_argument("--diagnose", action="store_true",
                        help="arm できない原因を PX4 に報告させる（SYS_STATUS / ESTIMATOR_STATUS / COMMAND_ACK）")
    parser.add_argument("--mode", choices=["gps", "ev"], default="gps",
                        help="gps: HIL_GPS 融合(標準HITL) / ev: ODOMETRY 融合(屋内モーキャプ相当)")
    parser.add_argument("--duration", type=float, default=None, help="実行時間 [s]（未指定は無制限）")
    parser.add_argument("--viewer", action="store_true", help="MuJoCo ビューアを表示")
    parser.add_argument("--params", default=None, help="vehicle_params.json のパス（未指定は既定）")
    parser.add_argument("--check-seconds", type=float, default=5.0, help="--check-link の送信時間 [s]")
    parser.add_argument("--arm", action="store_true",
                        help="RC を使わず MAVLink で arm する（COM_RC_IN_MODE=4 が必要）")
    parser.add_argument("--takeoff", type=float, default=None, metavar="ALT_M",
                        help="arm 後に AUTO.TAKEOFF で指定高度[m]へ（--arm を含む）")
    parser.add_argument("--arm-after", type=float, default=5.0,
                        help="センサ送信開始から arm 要求までの待ち [s]（EKF 収束待ち）")
    parser.add_argument("--forward", nargs="?", const="udpout:127.0.0.1:14550", default=None,
                        metavar="TARGET",
                        help="MAVLink を UDP へ中継し QGC を併用する（既定 udpout:127.0.0.1:14550）。"
                             "COM ポートは 1 プロセスしか掴めないため、QGC を使うにはこれが必要")
    args = parser.parse_args()

    config = HilConfig(
        device=args.device, baud=args.baud, mode=args.mode,
        auto_arm=args.arm or args.takeoff is not None,
        takeoff_alt_m=args.takeoff,
        arm_after_s=args.arm_after,
        forward=args.forward,
    )
    request = SimulationRequest(
        headless=True,
        duration_seconds=args.duration,
        params_path=args.params,
        hold_until_first_command=False,   # HIL 側で独自にスポーン保持する
    )

    if args.diagnose:
        run_diagnose(config, request, seconds=max(args.check_seconds, 15.0))
        return 0

    if args.check_link:
        stats = run_check_link(config, request, seconds=args.check_seconds)
        return 0 if (stats.heartbeat_ok and stats.controls_received > 0) else 1

    viewer_factory = None
    if args.viewer:
        import mujoco.viewer

        def viewer_factory(model, data):  # noqa: F811
            return mujoco.viewer.launch_passive(model, data)

    stats = run_bridge(config, request, viewer_handle_factory=viewer_factory)
    print(
        f"finished: steps={stats.steps} sensor_sent={stats.sensor_sent} gps_sent={stats.gps_sent} "
        f"odom_sent={stats.odom_sent} ctrl_msgs={stats.controls_received} "
        f"ctrl_applied_steps={stats.controls_applied_steps}"
    )
    if config.auto_arm:
        print(f"           arm_attempts={stats.arm_attempts} armed_reached={stats.armed_reached} "
              f"takeoff_commanded={stats.takeoff_commanded}")
        if not stats.armed_reached:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
