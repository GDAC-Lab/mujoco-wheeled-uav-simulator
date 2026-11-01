# 状態パケットの形式

シミュレータは `simulation.state_publish_every_n_steps` 物理ステップごとに 1 つの状態パケットを、1 つの UDP データグラムとして送ります。符号化の形式は `simulation.state_format`（または `simulate --state-format`）で選びます。

- `json`（既定）: 1 つの JSON オブジェクト。単体機のパケットは機体のフィールドを最上位に持ち、複数機のパケットは `uavs` の下に並べます。`timing` ブロックは [TIMING.ja.md](TIMING.ja.md) を参照してください。
- `binary`: 同じ状態を、下に示す固定のリトルエンディアンのレイアウトで送ります。大きさは JSON のパケットの約 4 分の 1 で、シミュレータ側の作成も軽く、約 170 機までが 1 データグラムに収まります（JSON のパケットは 1 機あたり約 1.5 KiB 増え、約 40 機で UDP データグラムの上限 65,507 バイトを超えます）。MATLAB では `jsondecode` が速いので、解読時間は JSON とほぼ同じです。

どちらのデコーダも、バイナリのパケットを同じ状態の JSON パケットと全く同じ結果に解読するので、コントローラはどちらの形式でも受け取れます。

- Python: `wheeled_uav.state_packets.decode_state_packet(datagram)` は `json.loads` と同じ dict を返します。
- MATLAB: `uavsim.Protocol.decode_state(datagram)` は `jsondecode` と同じ構造体を返します。`uavsim.Protocol.read_latest_state` はこれを使います。

バイナリのパケットは接触の集計を含みますが、接触 1 件ごとの詳細は含みません（`contacts` は空のリストに解読されます）。アクチュエータの統計（`logging_config.log_actuator_stats`）、センサ真値（`logging_config.log_sensor_truth`）、空力のログが有効なときは、バイナリ形式ではシミュレータが起動しません。

## バイナリのレイアウト（レイアウト版 1）

値はすべてリトルエンディアン、オフセットはデータグラム先頭からのバイト数です。

| オフセット | 型 | 内容 |
| --- | --- | --- |
| 0 | 4 バイト | マジック `WUAV` |
| 4 | uint16 | レイアウト版（1） |
| 6 | uint16 | `protocol_version` |
| 8 | uint16 | 機体数 `n` |
| 10 | uint16 | レコード長 `L`: 1 機あたりの float64 の個数（レイアウト版 1 では 46） |
| 12 | uint8 | `fidelity_mode`: 0 `baseline`、1 `hil`、255 なし |
| 13 | uint8 | `timing.pacing_mode`: 0 `realtime`、1 `accelerated`、2 `lockstep` |
| 14 | uint16 | フラグ: ビット `k`（0〜9）は timing の値 `k` があるとき、ビット 10 は `pacing_mode` があるとき、ビット 15 はパケットに `timing` ブロックがあるときに立ちます |
| 16 | int32 | `instance_id` |
| 20 | int64 | `sequence` |
| 28 | int64 | `wall_time_send_ns` |
| 36 | float64 | `sim_time` [s] |
| 44 | float64 | `realtime_factor` |
| 52 | float64 × 10 | timing の値 0〜9: `physics_timestep_seconds`、`control_period_seconds`、`publish_every_n_steps`、`session_wall_elapsed_seconds`、`session_sim_elapsed_seconds`、`sim_wall_skew_seconds`、`realtime_factor`、`step_wall_seconds`、`publish_wall_seconds`、`command_wait_wall_seconds`（ないものは NaN） |
| 132 | uint16 | 機体名の合計バイト数 `m` |
| 134 | `m` バイト | 機体名（UTF-8、改行区切り） |
| 134 + `m` | 0〜7 バイト | 8 の倍数までのゼロ埋め |
| ゼロ埋めの後 | float64 × `n` × `L` | 機体名の順に 1 機 1 レコード |

1 つのレコードの中身（添字）:

| 添字 | 値 |
| --- | --- |
| 0〜2 | `position` [m] |
| 3〜5 | `velocity` [m/s] |
| 6〜8 | `angular_velocity_world` [rad/s] |
| 9〜11 | `angular_velocity_body` [rad/s] |
| 12〜20 | `rotation_matrix`（JSON のリストと同じ 9 個、行優先） |
| 21〜25 | 機体全体の `contact_summary`: `count`、`total_force_magnitude`、`max_force_magnitude`、`total_normal_force`、`max_normal_force` |
| 26〜30 | `left_wheel` グループの同じ 5 つの値 |
| 31〜35 | `right_wheel` |
| 36〜40 | `surface` |
| 41〜45 | `wall` |

JSON の残りのフィールドはこれらから決まります。各機体の `time` は `sim_time`、`z`・`vz`・`yaw_rate` は `position`・`velocity`・`angular_velocity_world` の第 3 成分、`realtime_factor` と `instance_id` はヘッダの値です。独自のデコーダを書くときは、マジックとレイアウト版を確かめ、レコードはヘッダのレコード長で読み進めてください。後のレイアウト版でレコードに値が追加されることがあります。
