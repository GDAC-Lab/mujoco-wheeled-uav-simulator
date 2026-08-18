# 時刻同期モデル

本シミュレータは UDP 状態パケットを外部コントローラ（MATLAB、Python 等）とやり取りします。ログには 3 種類の時計が現れます。

| 時計 | ソース | 用途 |
| --- | --- | --- |
| **シミュレータ時刻** `state.time` | `mj_step` 後の MuJoCo `data.time` | 制御・ミッション・積分器の **唯一の基準** |
| **実時間（ウォールクロック）** | OS の monotonic / UTC | **シミュレータのペーシングのみ**；診断（`timing.sim_wall_skew_seconds`、`age_ms`） |
| **コントローラ実時間** | コントローラプロセスの時計 | ステータス表示間隔のみ — 制御則では **使わない** |

正本実装: [`wheeled_uav/timing.py`](../wheeled_uav/timing.py)

## 設計原則（破らないこと）

1. **コントローラ側で実時間がシミュレータ時刻に追いつくまで待たない。** RTF &lt; 1 のとき dead time になり、体感が遅くなる主因です。
2. **ミッション区切り・モード切替は `state.time`（シミュレーション秒）で行う。** 実時間は使わない。
3. **制御積分器の dt は連続するユニーク UDP サンプル間の `Δ(state.time)`** とし、公開周期（`timestep × state_publish_every_n_steps`）でクランプする。Python では `compute_control_dt_seconds()` を利用可能。
4. **実時間ペーシングはシミュレータのみ**（`StepPacer` + Windows では 1 ms タイマ分解能）。コントローラは新しい UDP 状態が来た分だけ反応する。
5. **重複 UDP 読み取りをスキップする。** ポーリングが公開レートより速いと同じ `(sequence, time)` を二度読む。`StateSampleTracker` / `udp_state_is_new` で **1 サンプル 1 回** 制御を評価する。

## 論理同期とペーシングモード（混同しないこと）

**論理同期**（常に有効）と **ペーシングモード**（シミュレータのみ切替）は別概念です。

| 概念 | 内容 | 切替 |
| --- | --- | --- |
| **論理同期** | `state.time` を基準に制御・ミッション・積分 dt を進める。重複 UDP スキップ。hold-last コマンド。 | 常時 |
| **`realtime` ペーシング** | `StepPacer` で実時間 ≈ シミュ時間（RTF≈1）。HIL・対話操作向け。 | `simulation.pacing_mode` |
| **`accelerated` ペーシング** | 実時間待ちなし。CPU 限界まで物理を進める。CI・バッチ実験向け。 | 同上 |
| **`lockstep` ペーシング** | 決定論的コシミュレーション：状態送信 → **コマンド受信までブロック** → 1 制御周期進める、の繰り返し。完全再現可能で、遅いコントローラでも取りこぼしなし。 | 同上 |

`lockstep` では実時間系の診断値の意味が変わります：`realtime_factor` と `sim_wall_skew_seconds` はコントローラの応答速度を映すだけになり、`simulation.hold_until_first_command` はそもそも物理がコマンド受信でしか進まないため無関係です。コントローラ側のルールは 3 モード共通で、準拠したコントローラは無変更で動きます。

**避けること:** コントローラが `sim_wall_skew` や `state.time` に合わせて wall clock で待つ「同期モード」。dead time が生じ、実時間より遅く感じる原因になります。

リモート HIL では **`realtime`** を使い、シミュレータが実時間の主役、コントローラは待たずイベント駆動、という構成が自然です。

## データフロー

```
シミュレータ（physics = simulation.timestep、既定 1 kHz）
  ├─ 毎ステップ mj_step
  ├─ N ステップごとに状態送信（state_publish_every_n_steps）
  │    time, sequence, wall_time_send_ns, timing{...}, realtime_factor
  ├─ 最新 UDP コマンドを適用（更新間は hold-last）
  └─ pacing_mode=realtime なら StepPacer で実時間に追従（RTF ≈ 1）
  └─ pacing_mode=accelerated ならペーシングなし（加速実行）
  └─ pacing_mode=lockstep なら各制御周期の前にコマンド受信までブロック

外部コントローラ
  ├─ 最新 UDP を非ブロッキング読み取り
  ├─ 重複 (sequence, time) をスキップ
  ├─ 新サンプルごとに 1 回制御計算
  └─ 即座にコマンド送信（wall/sim 合わせの sleep なし）
```

## 設定（`vehicle_params.json`）

| 項目 | 意味 | 既定 |
| --- | --- | --- |
| `simulation.timestep` | 物理ステップ (s) | `0.001` |
| `simulation.state_publish_every_n_steps` | UDP 間引き | `5` → 1 kHz 物理で 200 Hz |
| `simulation.viewer_fps` | ビューア `sync()` 上限 | `60` |
| `simulation.pacing_mode` | 実時間ペーシング | `realtime`（HIL/対話）、`accelerated`（加速）、`lockstep`（決定論的コシミュレーション） |
| `simulation.hold_until_first_command` | 最初のコマンド適用までスポーン姿勢で固定（自動実行の再現性確保） | 同梱 config は `true`、床置きスタートのプリセットは各自の JSON で `false` を設定、`lockstep` では無関係 |

CLI 上書き: `simulate --pacing-mode accelerated`、`simulate --no-hold-until-first-command`

**制御周期** = `timestep × state_publish_every_n_steps`（コントローラ側の公称 dt は `timing.control_period_seconds` から取得すること）

## 状態パケットの `timing` ブロック

すべての状態パケットに `timing` オブジェクトが載ります（`SessionTimingTracker.snapshot()`）。

| フィールド | 意味 |
| --- | --- |
| `physics_timestep_seconds` | MuJoCo 積分ステップ |
| `control_period_seconds` | 期待される制御サンプル周期 |
| `publish_every_n_steps` | 状態送信の間引き係数 |
| `pacing_mode` | `realtime` / `accelerated` / `lockstep` |
| `session_wall_elapsed_seconds` | セッション開始からの実経過 |
| `session_sim_elapsed_seconds` | セッション開始からのシミュ経過 |
| `sim_wall_skew_seconds` | シミュ経過 − 実経過（≈0 が健全、`realtime` 時） |
| `realtime_factor` | 実時間に対するシミュ速度（≈1.0、`realtime` 時） |

後方互換のため、トップレベルの `realtime_factor` にも同じ値が入ります。

## 診断

- **rtf ≈ 1.0** — 実時間追従 OK
- **skew ≈ 0** — シミュと実時間のずれ小
- **age** — ローカルでは通常数 ms 未満

`rtf < 0.9` のとき 5 秒ごとに警告。対策: `state_publish_every_n_steps` を増やす、接触詳細ログを切る、`viewer_fps` を下げる、ヘッドレス実行。

## コントローラ API

| 言語 | モジュール | 主な API |
| --- | --- | --- |
| Python（シミュレータ側） | `wheeled_uav.timing` | `StateSampleTracker`, `build_pacer`, `compute_control_dt_seconds`, `extract_sync_metrics`, `parse_simulation_timing` |
| Python（hover） | `wheeled_uav.controllers.hover` | `StateSampleTracker` を利用；ステータスに rtf / age / skew を表示 |
| MATLAB（サブモジュール） | `uavsim.Protocol` | `sim_time_seconds`, `udp_state_is_new`, `build_sync_metrics` |

サブモジュールとして本シミュレータを使う project リポジトリは、同じヘルパーを自前のコントローラでラップする想定です（本リポジトリの外）。

## リモート実行の補足

コントローラを別マシンで動かす場合：

- 制御則は引き続き **`state.time` のみ** を使う
- `packet_age_ms` にはネットワーク遅延が含まれる（大きくても想定内）
- `sim_wall_skew_seconds` は **シミュレータ側** のペーシングを表し、ネットワーク遅延ではない
- RTF ≈ 1 での計算資源評価は、設定した公開レートを維持できるマシン構成で行う

英語版の詳細: [TIMING.md](TIMING.md)
