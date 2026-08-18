# サンプルコントローラと実行シナリオ

本リポジトリ（シミュレータ submodule）が提供する **実行コンポーネント**、**時刻の扱い**、**想定用途** をまとめます。時刻同期の設計原則は [TIMING.ja.md](TIMING.ja.md) を正本とし、ここでは「どのプログラムをどの組み合わせで使うか」に焦点を当てます。

## 用語（混同しないこと）

| 用語 | 意味 |
| --- | --- |
| **論理同期** | 制御・ミッションは常に `state.time`（シミュレータ秒）を基準に進める。重複 UDP をスキップし、コマンドは hold-last。**全コントローラで常時有効。** |
| **`realtime` ペーシング** | シミュレータが `StepPacer` で実時間 ≈ シミュ時間（RTF≈1）を目指す。HIL・対話操作向け。 |
| **`accelerated` ペーシング** | シミュレータは実時間待ちなし。CI・バッチ実験向け。 |
| **`lockstep` ペーシング** | 各制御周期の前にコマンド受信までブロック。決定論的コシミュレーション・完全再現実行向け。 |
| **`baseline` / `hil` fidelity** | ネットワーク遅延・ノイズ等の注入有無。**ペーシングとは直交。** |

コントローラが `sim_wall_skew` や実時間に合わせて **待機する方式は採用しません**（dead time の原因になります）。

## 提供コンポーネント

| コンポーネント | 入口 | 役割 | コントローラ側の時刻ルール |
| --- | --- | --- | --- |
| **シミュレータ** | `mujoco-wheeled-uav-simulator simulate` | MuJoCo 物理、`mj_step`、UDP 状態送信 | `simulation.pacing_mode` で `realtime` / `accelerated` / `lockstep` |
| **Python サンプル** | `mujoco-wheeled-uav-simulator hover-controller` | 単体機ホバー（世界座標 PD + 姿勢制御の参照実装） | 新 UDP サンプルごとに 1 回制御。`state.time` のみ使用。待機なし |
| **MATLAB ホバーサンプル** | `hovering_controller` | 単体機ホバー（Python 参照実装と数値同一） | 同じ論理同期ルール |
| **MATLAB 壁面デモ** | `wall_demo_controller` | 壁面走行の最小デモ（無改造のホバリング制御＋壁の奥の目標位置。意図的に自明なベースライン） | 新規 UDP サンプルごとに 1 回評価 |
| **MATLAB 編隊サンプル** | `multi_uav_formation_controller` | バッチ複数機パケットで重心＋スロット編隊 | 新バッチサンプルごとに 1 回評価、1 サンプル 1 データグラム送信 |
| **MATLAB 共有ライブラリ** | `matlab/+uavsim/`（`uavsim.Protocol`, `uavsim.Session` など） | UDP 送受信・タイミングヘルパ・起動補助 | 上記と同じ論理同期 API（`uavsim.Protocol.udp_state_is_new` 等） |
| **外部プロジェクトの MATLAB** | 本リポジトリを submodule として使う各リポジトリのコントローラ | 論文制御則などプロジェクト固有ロジック | 同上（本リポジトリの外で、各プロジェクトが同型ルールをラップ） |

`hover-controller` は **単体機パケット専用** です。`uavs` 配列を含む複数機パケットは受け付けません。

## 推奨シナリオ一覧

| シナリオ | シミュレータ `pacing_mode` | `fidelity_mode` | コントローラ | 典型的な起動例 |
| --- | --- | --- | --- | --- |
| **A. 対話開発（ローカル）** | `realtime` | `baseline` | プロジェクト MATLAB または `hover-controller` | ビューア付き `simulate` + 別ターミナルでコントローラ |
| **B. リモート HIL（2 台構成）** | `realtime` | `hil`（遅延注入が必要なとき） | リモート側の実装 or `hover-controller` | PC: `simulate --bind-ip 0.0.0.0`、リモート側: `hover-controller --bind-ip 0.0.0.0` |
| **C. 実時間ヘッドレス確認** | `realtime` | `baseline` | 任意 | `simulate --headless --duration-seconds 30` + コントローラ |
| **D. CI / 回帰（物理のみ）** | `accelerated` | `baseline` | なし（開ループ）または高速ループ | `simulate --headless --pacing-mode accelerated --duration-seconds 10` |
| **E. バッチ・ゲイン探索** | `accelerated` | `baseline` | `hover-controller`（`--duration-seconds` 指定） | シミュ + Python コントローラを同時起動 |
| **F. タイミング単体テスト** | 両方 | `baseline` | なし | `pytest tests/test_simulation_timing.py` |
| **G. 壁面走行デモ** | `realtime` | `baseline` | `wall_demo_controller` | `simulate --preset wall_demo` + `wall_demo_controller` |
| **H. 編隊** | `realtime` | `baseline` | `multi_uav_formation_controller` | `simulate --num-uavs 3` + `multi_uav_formation_controller('num_uavs', 3)` |
| **I. 決定論的リプレイ／コシミュレーション** | `lockstep` | `baseline` | 任意の単機/複数機コントローラ | `simulate --headless --pacing-mode lockstep` + コントローラ |

### シナリオ A — 対話開発（既定）

```powershell
# ターミナル 1（シミュレータ、実時間・ビューア）
uv run mujoco-wheeled-uav-simulator simulate --params-file vehicle_params.json

# ターミナル 2（Python 参照コントローラ）
uv run mujoco-wheeled-uav-simulator hover-controller
```

`vehicle_params.json` の `simulation.pacing_mode` は既定で `realtime` です。

### シナリオ B — PC シミュレータ + リモートコントローラ

```powershell
# PC（シミュレータ側が実時間の主役）
uv run mujoco-wheeled-uav-simulator simulate ^
  --bind-ip 0.0.0.0 --state-target-ip 192.168.0.42 ^
  --fidelity-mode hil --params-file vehicle_params.json
```

```bash
# リモート側（待たない・イベント駆動）
uv run mujoco-wheeled-uav-simulator hover-controller \
  --bind-ip 0.0.0.0 --target-ip 192.168.0.10 --fidelity-mode hil
```

診断: 状態パケットの `rtf` / `skew`（シミュレータ側）、`age`（ネットワーク＋処理遅延）、コントローラログの `state_gap` / `dup_skip` / `compute` ms。

### シナリオ D/E — 加速実行

```powershell
uv run mujoco-wheeled-uav-simulator simulate --headless ^
  --pacing-mode accelerated --duration-seconds 60
```

制御則の検証まで行う場合も、コントローラは **待機を入れず** `state.time` 基準のまま動かします。ミッションの「実時間 5 秒後」といった解釈は **`state.time` の 5 秒** です（壁時計ではありません）。

## コントローラ実装チェックリスト

外部コントローラ（MATLAB / Python / C++）を新規実装するとき:

1. 制御・積分 dt は **`Δ(state.time)`**（連続するユニーク UDP サンプル間）
2. **重複 `(sequence, time)` は 1 回だけ** 制御評価（`StateSampleTracker` / `udp_state_is_new`）
3. **`sim_wall_skew` で pause しない**（ログ専用）
4. コマンドは受信後 **即送信**（wall/sim アライメント sleep なし）
5. シミュレータが遅い・速い場合でも、ミッション区切りは **`state.time`**

## 利用側リポジトリ（外部）との関係

本シミュレータを Git submodule として組み込むリポジトリは、同じ UDP 契約の上に自前のコントローラを載せます。タイミングルールは Python `hover-controller` と同一で、ペーシングはシミュレータ側の `pacing_mode` のみが切り替わります。プロジェクト固有のコントローラや設定は利用側リポジトリに置き、`--params-file` でプロジェクト所有のパラメータファイルを渡してください（相対メッシュパスはそのファイルの隣で解決されます）。

英語版: [CONTROLLERS.md](CONTROLLERS.md)
