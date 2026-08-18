# Fidelity Modes

この文書では、このリポジトリにおける `baseline` と `hil` の意味を定義します。

## 目的

このリポジトリでは、次の 2 つを別軸として扱います。

- `physical fidelity`: 理想的な通信・計測前提のもとで、基準となる物理モデルにどれだけ忠実か
- `real-time fidelity`: 壁時計時間、packet 遅延、jitter、packet loss、stale-command handling を含む閉ループ挙動がどれだけ妥当か

これらを曖昧にまとめて 1 つの「fidelity」と呼ばないことが重要です。論文や報告では、どの mode を使ったかと、どの指標を評価しているかを明示してください。また、もう 1 つの独立軸である `simulation.pacing_mode`（`realtime` / `accelerated` / `lockstep`）も併記してください（[TIMING.ja.md](TIMING.ja.md) 参照）。

## Baseline Mode

`baseline` は、runtime 擾乱を極力入れない基準実験用 mode です。

想定挙動:

- network 遅延注入なし
- packet loss 注入なし
- jitter 注入なし
- sensor 劣化注入なし
- actuator 劣化注入なし
- packet metadata は残すので、`hil` と同じログ構造で比較できます

主な用途:

- 理想条件での controller 開発
- 位置・姿勢追従の基準プロット
- network が主因ではない接触・編隊実験

## HIL Mode

`hil` は、remote controller や hardware-in-the-loop に近い timing 条件を模擬したいときに使います。

現時点で実装済みの内容:

- `network_fidelity.state_tx_latency_ms` による state 送信遅延注入
- `network_fidelity.command_rx_latency_ms` による command 受信遅延注入
- `network_fidelity.jitter_std_dev_ms` によるガウス jitter 注入
- `network_fidelity.packet_loss_percent` による packet drop 注入
- `network_fidelity.stale_command_threshold_ms` と `network_fidelity.stale_command_policy` による stale-command handling
- `actuator_dynamics` による一次遅れとロータ推力または角速度の rate limit
- `sensor_fidelity` と `logging_config` による加法ノイズ注入と truth 保存

モデル化していないもの:

- 加法ノイズモデルを超える sensor bias, quantization, delayed measurement

## リモート評価の 2 トラック

simulator と controller を別マシンで動かす場合、評価したいものは 1 つの曖昧な HIL fidelity ではありません。実際には、次の 2 つを分けて扱うのが適切です。

### 1. 低 RTF を許容する時刻忠実評価

`realtime_factor` が 1 未満でも、simulator 側と controller 側の間で壁時計ベースの packet timing 解釈が崩れていないかを確認したいときに使います。

想定解釈:

- packet timestamp と age は壁時計に対して評価する
- 遅延注入、jitter、loss、stale-command handling は引き続き意味を持つ
- `realtime_factor` が 1 未満でも、runtime instrumentation の整合性が取れていることを重視する

主な指標:

- state packet age
- command packet age
- sequence gap rate
- stale command rate
- timeout count

### 2. RTF≈1 を狙う計算資源評価

controller 側マシンが必要な閉ループ周期を維持できるかを、`realtime_factor` を 1 に近づけた条件で確認したいときに使います。

想定解釈:

- 構成は引き続き simulator ホスト 1 台と controller ホスト 1 台
- 主な判定軸は controller compute time と transport delay が timing budget 内に収まるかどうか
- 対象シナリオで `realtime_factor` が 1 に近いことを求める

主な指標:

- realtime factor
- controller compute time
- sustained load 下での state packet age
- `baseline` に対する tracking degradation

この 2 つは packet 形式を変える話ではなく、同じ logging 構造を別の評価目的で使い分ける話です。

## Runtime Metadata

state packet には現在、少なくとも次が入ります。

- `protocol_version`
- `sequence`
- `wall_time_send_ns`
- `fidelity_mode`
- `sim_time`

command packet には現在、少なくとも次が入ります。

- `protocol_version`
- `sequence`
- `source_state_sequence`
- `wall_time_send_ns`
- `fidelity_mode`

Python 側 runtime では現在、少なくとも次を追跡します。

- state packet age
- state sequence gap
- command packet age
- command sequence gap
- stale command count
- stale command apply count
- command timeout count
- controller compute time

MATLAB の `.mat` ログにも同じ packet metadata を保存するので、Python と MATLAB の両方で同じ downstream 解析系を使いやすくしています。

## 設定場所

共有設定は `vehicle_params.json` にあります。

- `fidelity_mode`
- `network_fidelity`
- `actuator_dynamics`
- `sensor_fidelity`
- `logging_config`

HIL を意識した最小例:

```json
{
  "fidelity_mode": "hil",
  "network_fidelity": {
    "enabled": true,
    "state_tx_latency_ms": 15.0,
    "command_rx_latency_ms": 8.0,
    "packet_loss_percent": 1.0,
    "jitter_std_dev_ms": 2.0,
    "stale_command_threshold_ms": 40.0,
    "stale_command_policy": "zero-thrust"
  }
}
```

## 推奨指標

`baseline` では:

- tracking error
- attitude error
- contact force の統計量
- control input の大きさ
- 同条件反復時の再現性

`hil` では:

- realtime factor
- state packet age
- command packet age
- sequence gap rate
- stale command rate
- timeout count
- `baseline` に対する tracking degradation

## 実行例

基準となる `baseline` 実行:

```powershell
uv run mujoco-wheeled-uav-simulator simulate --fidelity-mode baseline
uv run mujoco-wheeled-uav-simulator hover-controller --fidelity-mode baseline
```

HIL を意識した実行:

```powershell
uv run mujoco-wheeled-uav-simulator simulate --fidelity-mode hil --bind-ip 0.0.0.0 --state-target-ip 192.168.0.42
uv run mujoco-wheeled-uav-simulator hover-controller --fidelity-mode hil --bind-ip 0.0.0.0 --target-ip 192.168.0.10
```

## スコープ

runtime は、論文向けの instrumentation、network 擾乱注入、一次遅れ・レート制限の actuator dynamics、加法型の sensor noise と truth logging を提供します。HIL 的主張は full hardware equivalence ではなく、通信 timing と runtime behavior を主軸に位置付けるのが適切です。