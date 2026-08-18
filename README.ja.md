# mujoco-wheeled-uav-simulator

English documentation: [README.md](README.md)

[MuJoCo](https://mujoco.org/) 上の車輪付きクアッドロータを、Python シミュレータと MATLAB コントローラの UDP 通信で動かすサンプルです。今後の研究で使い回せるシミュレーション基盤として使うことと、論文中の制御則を再現する際の参照実装にすることを主な想定用途にしています。

![壁面走行の最小デモ: 接近・壁面接触・上昇・保持・下降を2つのカメラアングルで表示](docs/media/wall_demo.gif)

*同梱の `wall_demo`: 無改造のホバリング制御に「壁より奥の目標位置」を与えるだけで、壁の反力が押し付け力になり、車輪が壁面を転がります。**左**（斜め前から）: 緑の縦線は指令された登攀経路、緑の球はその上を動く目標位置（リファレンス）。**右**（真横から）: 接近で車輪と壁の隙間が閉じ、上昇の間ずっと接触が保たれる様子。車輪と壁の接触点・接触力ベクトルも重ねて描画しています。`simulate --preset wall_demo --record wall_demo.gif` で録画（解像度・カメラアングル・接触可視化はパラメータの `environment.recording`。下記「パラメータ管理」参照）。*

**空力効果モデル**（設定ゲート付きの壁面効果力注入）: [docs/AERODYNAMICS.ja.md](docs/AERODYNAMICS.ja.md) — 英語版 [docs/AERODYNAMICS.md](docs/AERODYNAMICS.md)。

`baseline` / `hil` の使い分けと、論文向けの fidelity 指標整理は [docs/FIDELITY_MODES.ja.md](docs/FIDELITY_MODES.ja.md) にまとめています。

**時刻同期**（シミュレータ時刻・実時間・UDP サンプル周期）: [docs/TIMING.ja.md](docs/TIMING.ja.md) — 英語版 [docs/TIMING.md](docs/TIMING.md)。

**サンプルコントローラと実行シナリオ**（`realtime` / `accelerated` / `lockstep`、HIL、CI の組み合わせ）: [docs/CONTROLLERS.ja.md](docs/CONTROLLERS.ja.md) — 英語版 [docs/CONTROLLERS.md](docs/CONTROLLERS.md)。

**PX4 HITL（実機 Pixhawk 接続）** — MuJoCo をプラントに、実機 PX4 の EKF2・制御スタックをそのまま動かす: `uv sync --extra hil` の後、`uv run python -m wheeled_uav.px4_hitl --help` を参照。

**壁面走行の最小デモ**（無改造のホバリング制御に壁の奥の目標位置を与えるだけ。車輪が壁面を転がる）: `wall_demo_controller`（MATLAB）＋ `wall_demo` プリセット — 下記のデモ節を参照。

## 実行シナリオ（要約）

| 目的 | シミュレータ `pacing_mode` | サンプルコントローラ |
| --- | --- | --- |
| 対話操作・リモート HIL | `realtime`（既定） | `hover-controller` または外部プロジェクトの MATLAB |
| 壁面走行の最小デモ | `realtime` | `wall_demo_controller`（MATLAB）+ `wall_demo` プリセット |
| CI・バッチ・高速実験 | `accelerated` | 同上（コントローラは待たない・`state.time` 基準） |
| 決定論的コシミュレーション・完全再現 | `lockstep` | 同上（コマンド受信まで物理がブロック） |

- シミュレータ: `simulate`（ペーシングは JSON の `simulation.pacing_mode` または `--pacing-mode`）
- Python 参照コントローラ: `hover-controller`（単体機ホバー、論理同期のみ・実時間待ちなし）
- 詳細な起動例・診断指標: [docs/CONTROLLERS.ja.md](docs/CONTROLLERS.ja.md)

## 特徴

- 車輪付きクアッドロータの MuJoCo シミュレーション
- MATLAB からのホバリング制御、壁面走行デモ、編隊制御ワークフロー
- Python と MATLAB で共有する `vehicle_params.json` ベースのパラメータ管理
- 単体機、独立 multi-instance、single-world multi-UAV に対応
- `.mat` ベースの接触ログ保存と解析
- 平面、傾斜面、関数ベース曲面地形に対応

## 必要なもの

- Python 3.12 以上
- `uv`
- MuJoCo の GUI を表示できるローカル実行環境
- MATLAB

## クイックスタート

依存関係を入れます。

```powershell
uv sync
```

通常の単体機シミュレーション（`wuav` は同じ CLI の短縮エイリアスです）:

```powershell
uv run mujoco-wheeled-uav-simulator simulate
```

```powershell
uv run wuav simulate
```

リモート controller 試験では、simulator の bind 先 IP と state 送信先 IP を分けて指定できます。

```powershell
uv run mujoco-wheeled-uav-simulator simulate --bind-ip 0.0.0.0 --state-target-ip 192.168.0.42
```

実行のオフスクリーン動画を保存する場合（拡張子で形式決定、解像度・fps は `environment.recording.{width,height,fps}`）:

```powershell
uv run wuav simulate --record logs/run.mp4
```

```matlab
hovering_controller
```

## よく使う実行例

### 単体機ホバリング

```powershell
uv run mujoco-wheeled-uav-simulator simulate
```

```matlab
hovering_controller
```

MATLAB の代わりに Python controller で hover 試験を回すこともできます。

```powershell
uv run mujoco-wheeled-uav-simulator hover-controller
uv run mujoco-wheeled-uav-simulator hover-controller --bind-ip 0.0.0.0 --target-ip 192.168.0.10
```

### 壁面走行の最小デモ

壁面専用の制御則を一切持たないデモです。**無改造の共有ホバリング制御に、壁面より奥の目標位置を与えるだけ**——機体は壁で止まり、位置誤差がそのまま押し付けになります。同じ目標の高さ成分を上昇→保持→下降のプロファイルで動かすと、車輪の転がりで壁面を昇降します。壁専用則・押し付け力の設計・接触維持の保証を持たない意図的に自明なベースラインで、シミュレータが「スポーン→壁接触→接触力の計測・ロギング」まで車輪付き壁面シナリオを一通り扱えることを示すためのものです。

```powershell
uv run wuav simulate --preset wall_demo
```

```matlab
wall_demo_controller
```

### リモート／分散構成

シミュレータとコントローラは別マシンでも動かせます。`vehicle_params.json` と packet 挙動を揃えるため、両方のマシンに同じリポジトリを clone してください。

```powershell
# MuJoCo を動かすマシン
uv run mujoco-wheeled-uav-simulator simulate --bind-ip 0.0.0.0 --state-target-ip 192.168.0.42
```

```bash
# controller を動かすマシン
uv run mujoco-wheeled-uav-simulator hover-controller --bind-ip 0.0.0.0 --target-ip 192.168.0.10
```

`192.168.0.42` は controller 側の IP、`192.168.0.10` はシミュレータ側の IP に読み替えてください。

MATLAB と Python の hover controller が共通で前提にしている単一 UAV packet 仕様は次のとおりです。実行シナリオの対応表は [docs/CONTROLLERS.ja.md](docs/CONTROLLERS.ja.md) を参照してください。

- simulator からの state packet は JSON object で、少なくとも `time`, `position`, `velocity`, `angular_velocity_body`, `rotation_matrix` を含みます。
- `position`, `velocity`, `angular_velocity_body` は 3 要素ベクトルです。
- `rotation_matrix` は row-major の 3x3 回転行列を 1 次元化した 9 要素です。
- controller から simulator への command packet は JSON object で、`rotor_thrusts` / `rotor_omega`（4 要素ベクトル）、または `wrench`（`[f_z, M_x, M_y, M_z]` の機体レンチ。ロータ幾何から導出したアロケーション行列の擬似逆で各ロータ推力へ分配）を持ちます。
- `hover-controller` は単一 UAV packet 専用で、`uavs` を含む複数 UAV packet を受けるとエラー終了します。
- controller 側の推力配分（Python / MATLAB とも）は `actuation.rotors` の各ロータ幾何（位置・推力軸・回転方向）から導出されるため、JSON 上のロータ順序や固定傾斜ロータでも生成される MuJoCo actuator と常に整合します。

`baseline` / `hil` モードの使い分け:

- `--fidelity-mode baseline` は理想化された基準経路です。network 遅延注入、packet loss 注入、追加の sensor / actuator 劣化は入りません。
- `--fidelity-mode hil` は `vehicle_params.json` の `network_fidelity` を有効にし、state 送信遅延、command 受信遅延、jitter、packet loss、stale-command handling を反映します。
- Python と MATLAB のログには `sequence`、`source_state_sequence`、`wall_time_send_ns`、`state_age_ms`、controller 計算時間などの metadata が残るので、remote 実行でも同じ形式で評価できます。

HIL を意識した実行例:

```powershell
uv run mujoco-wheeled-uav-simulator simulate --fidelity-mode hil --bind-ip 0.0.0.0 --state-target-ip 192.168.0.42
uv run mujoco-wheeled-uav-simulator hover-controller --fidelity-mode hil --bind-ip 0.0.0.0 --target-ip 192.168.0.10
```

### 1つの MuJoCo world での編隊制御

```powershell
uv run mujoco-wheeled-uav-simulator simulate --num-uavs 3
```

```matlab
multi_uav_formation_controller('num_uavs', 3)
multi_uav_formation_controller('num_uavs', 3, 'formation_radius', 2.0, 'spawn_radius', 2.0, 'base_height', 1.8)
```

### 独立した複数 simulator instance

このモードは、単体機の切り分け実験や比較用に向いています。

```powershell
uv run mujoco-wheeled-uav-simulator simulate --instance-id 0
uv run mujoco-wheeled-uav-simulator simulate --instance-id 1
uv run mujoco-wheeled-uav-simulator simulate --instance-id 0 --bind-ip 0.0.0.0 --state-target-ip 192.168.0.42
```

```matlab
hovering_controller('instance_id', 0)
hovering_controller('instance_id', 1)
```

### モデル生成確認のみ

```powershell
uv run mujoco-wheeled-uav-simulator check-model
uv run mujoco-wheeled-uav-simulator check-model --instance-id 1
uv run mujoco-wheeled-uav-simulator check-model --num-uavs 3
```

## Citation

研究でこのシミュレータを利用する場合は、本リポジトリの URL を参照してください。形式的な引用が必要な場合は [CITATION.cff](CITATION.cff) にメタデータがあります（GitHub の "Cite this repository" ボタンにも使われます）。物理エンジンには [MuJoCo](https://mujoco.org/) を利用しています。MuJoCo 自体の引用方法は MuJoCo のドキュメントを参照してください。

## ライセンス

このプロジェクトは [MIT License](LICENSE) で公開しています。

## 自分のリポジトリからこのシミュレータを使う

このリポジトリは、再利用可能なシミュレータ基盤＋少数のサンプルという構成を保ちます。自分のプロジェクトを作る場合は、本リポジトリを Git submodule として組み込み、コントローラや設定は自分のリポジトリ側に置いて、`simulate --params-file <your_params.json>` で自前のパラメータファイルを渡してください。パラメータファイル内の相対メッシュパスはそのファイルの隣で解決されるため、機体メッシュや較正済みパラメータをシミュレータ本体に手を入れずに同梱できます。

## 詳細リファレンス

<details>
<summary>リポジトリ構成</summary>

### トップレベルの主なファイル

| ファイル | 役割 |
|---------|------|
| `wheeled_uav/` | Python シミュレータの本体パッケージです（下の内訳を参照）。 |
| `hovering_controller.m` | MATLAB 側ホバリング制御のルート入口です。内部実装は `matlab/controllers/hovering_controller_impl.m` にあります。 |
| `wall_demo_controller.m` | 壁面走行最小デモのルート入口です。内部実装は `matlab/controllers/wall_demo_controller_impl.m`。 |
| `multi_uav_formation_controller.m` | 1 つの MuJoCo world に複数 UAV を生成し、単一の MATLAB controller でフォーメーション制御するルート入口です（内部実装は `matlab/controllers/multi_uav_formation_controller_impl.m`）。 |
| `matlab/+uavsim/` | MATLAB 共有ライブラリパッケージ（セッション、UDP プロトコル、パラメータ、制御計算、ジョイスティック。下の内訳を参照）。 |
| `matlab/shared/simulation_logger.m` | MATLAB 側のログ保存クラスです。`logs/` 以下に `.mat` を出力します。 |
| `vehicle_params.json` | 機体・アクチュエータ・環境の共有パラメータです。Python と MATLAB の両方から参照します。 |
| `configs/` | すぐ使えるパラメータプリセット（壁面走行デモ用の `vehicle_params.wall_demo.json` など）。 |
| `wheeled_uav.template.xml` | MuJoCo モデルのテンプレートです。実行時に `vehicle_params.json` から `build/generated_xml/` 配下へ XML を生成します。 |
| `pyproject.toml` | Python 依存関係の宣言です（CLI 名 `mujoco-wheeled-uav-simulator` と `wuav`）。 |
| `uv.lock` | `uv` 用のロックファイルです。 |

### Python パッケージ内訳

| モジュール | 役割 |
|---------|------|
| `wheeled_uav/cli.py` | CLI 入口です。`simulate` / `check-model` / `hover-controller` を振り分けます。 |
| `wheeled_uav/config.py` | `vehicle_params.json` の読込と fidelity / 空力設定の解釈を担当します。 |
| `wheeled_uav/paths.py` | リポジトリ直下の主要ファイルパスと共通定数をまとめています。 |
| `wheeled_uav/protocol.py` | シミュレータと全コントローラで共有する UDP ワイヤフォーマット（ポート、ソケット、packet の構築/解釈）。MuJoCo 非依存です。 |
| `wheeled_uav/timing.py` | ペーシングモード、実時間係数、コントローラ側のタイミングヘルパです。 |
| `wheeled_uav/types.py` | Python 側で共有する dataclass 型を定義します。 |
| `wheeled_uav/model/` | MuJoCo モデル生成: `builder.py`（XML 置換）、`surface.py`（地形の数式と hfield 出力）、`poses.py`（初期姿勢）、`xml_format.py`。 |
| `wheeled_uav/runtime/` | シミュレーション実行系: `scene.py`（リクエスト/シーン/検証）、`loop.py`（stepping / lockstep ループ）、`state_publisher.py`、`command_dispatcher.py`、`fidelity.py`（センサノイズ・アクチュエータ動特性）、`contact.py`、`aerodynamics.py`、`visuals.py`（カメラ・オーバーレイ・録画）。 |
| `wheeled_uav/controllers/` | 参照コントローラ（`hover.py`）。MuJoCo なしで import できるため controller 専用ホストでも使えます。 |

### MATLAB 構成内訳

ルート直下の `.m` ファイルは意図的に薄いラッパです。各ファイルは MATLAB パスの設定と、対応する `matlab/controllers/*_impl.m` 実装への委譲だけを行うため、リポジトリ直下から `hovering_controller` のようにパス管理なしで呼び出せます。ロジックはすべて実装側にあります。

| パス | 役割 |
|---------|------|
| `matlab/+uavsim/Session.m` | コントローラセッションの初期化（パラメータ読込、ポート診断つき UDP ソケット、任意のシミュレータ自動起動）。 |
| `matlab/+uavsim/Protocol.m` | UDP ワイヤフォーマット（状態 packet の解釈と計測、重複サンプル判定、コマンド packet の構築・送信）。 |
| `matlab/+uavsim/Params.m` | `vehicle_params.json` の解釈と、ロータ幾何から導く allocation 行列 / mixer。 |
| `matlab/+uavsim/Control.m` | 幾何制御の部品（ホバー PD、SO(3) 姿勢モーメント、レンチ→推力変換）。 |
| `matlab/+uavsim/Joystick.m` | ジョイスティック入力（`sim3d.io.Joystick` / `vrjoystick`）とデッドゾーン・軸割り当て。 |
| `matlab/+uavsim/Launch.m` | シミュレータ起動コマンドの生成と UDP ポート確認。 |
| `matlab/+uavsim/RunOptions.m`, `Metrics.m`, `LogFiles.m`, `Util.m` | オプション解析、ランタイム計測、ログ探索、小物ヘルパ。 |
| `matlab/controllers/` | サンプルコントローラ実装（ホバー、壁面走行デモ、編隊）。ルート入口 1 つにつき `..._impl.m` が 1 つ対応します。 |
| `matlab/shared/simulation_logger.m` | 状態、制御入力、接触サマリを `.mat` に保存するロガークラスです。 |

</details>

<details>
<summary>通信ポートと実行モード</summary>

単一インスタンスでは、Python 側が `127.0.0.1:5001` へ状態を送信し、MATLAB 側が `127.0.0.1:5000` へ各ロータ推力または各ロータ角速度を返します。

複数インスタンスでは `instance_id = i` に対して次の規則でポートをずらします。

- simulator receive port: `5000 + 2*i`
- simulator state send port: `5001 + 2*i`

たとえば `instance_id = 1` なら、Python は `5002` で制御入力を受け、`5003` へ状態を送信します。MATLAB 側は `hovering_controller('instance_id', 1)` のように同じ `instance_id` を指定してください。

MATLAB controller のローカル UDP ポートが既に使用中だと表示された場合、典型的には別の MATLAB セッションや以前の controller process が同じポートを保持しています。shared controller runtime は、期待していたポート番号と、Windows では取得できる場合は所有 process 情報も含めて早めに停止します。テストを繰り返す場合は、先に古い controller セッションを完全に閉じるか、別の `instance_id` を使うのが安全です。

`multi_uav_formation_controller` は独立インスタンス方式とは別で、編隊制御の推奨経路です。`simulate --num-uavs N` で 1 つの MuJoCo world に `N` 台の UAV を生成し、状態 packet も制御 packet も配列でまとめて送受信します。

</details>

<details>
<summary>パラメータ管理</summary>

機体やアクチュエータの主要パラメータは `vehicle_params.json` に集約しています。現時点では少なくとも以下が共有化されています。

- 重力とシミュレーション刻み幅
- 任意のソルバ設定: `simulation.integrator`（`Euler`, `RK4`, `implicit`, `implicitfast`）、`simulation.cone`（`pyramidal`, `elliptic`）、`simulation.iterations`、`simulation.noslip_iterations`
- 状態送信の間引き `simulation.state_publish_every_n_steps`（既定 1 = 毎ステップ；同梱 config は 5 = 200 Hz）
- ビューア更新レート `simulation.viewer_fps`（既定 60；物理レートとは非同期）
- 起動時フリーズ `simulation.hold_until_first_command`（既定 config では `true`）: 最初の control command が届くまで機体を spawn 姿勢で固定します。hover・編隊・バッチ実行など自動実行で、接続待ちの間に姿勢が乱れないきれいな初期条件を保証します。床置きスタートのプリセットは各自の JSON で `false` を設定します。実行ごとの上書きは `simulate --hold-until-first-command` / `--no-hold-until-first-command`。`lockstep` pacing では command 受信でしか物理が進まないため無関係です。
- アーム長（または `actuation.rotors` による各ロータ幾何の明示指定：位置・推力軸・回転方向・反トルク比）
- 反トルク係数
- 最大ロータ推力
- ロータ推力換算係数 `thrust_coefficient`
- 機体初期位置
- スポーン姿勢ポリシー `drone.initial_spawn`: `mode` は `"surface"`（既定；地形に接地静定）、`"wall_contact"`（床＋壁スタート。`wall_clearance_m` 離隔（既定 5 mm）と任意の `b2_body`/`b3_body` 初期姿勢付き）、`"explicit"`（`positions_xy` で複数機を任意座標に配置；`center_x` で壁法線方向オフセットを上書き）
- 機体ボディと車輪の主要寸法・質量；`drone.inertial_reference` は `"body_only"`（既定；`drone.mass`/`drone.inertia` は中央ボディのみで車輪分は加算）または `"total_vehicle"`（全機体を表し、MuJoCo ボディ構築時に車輪の解析寄与を減算）
- 床、壁、機体、車輪の接触設定（`solref`、任意で `solimp` と `condim`）。キーの綴りは要素ごとに異なります：車輪はネスト（`drone.wheels.solimp`/`condim`）、surface もネスト（`environment.surface.solimp`/`condim`）、壁はフラット接頭辞（`environment.wall_solimp`, `environment.wall_condim`）
- `environment.surface` による平面または関数ベース曲面の指定
- オフスクリーン録画 `environment.recording`（`width`, `height`, `fps`；`simulate --record PATH` が使用）。任意の `views` で複数カメラアングルを横並び合成（自動フレーミングに対する `{azimuth, elevation, distance_scale}` 上書きのリスト）、`show_contacts: true` で録画に接触点・接触力ベクトルを描画（対話ビューアの表示設定には影響しない）
- MuJoCo センサ名とセンサ対象ボディ
- `fidelity_mode`, `network_fidelity`, `actuator_dynamics`, `sensor_fidelity`, `logging_config` による baseline / HIL 実験設定（`logging_config.include_contact_details` で接触詳細の per-contact 記録を切替；サマリは常時送信）
- `aerodynamics` による空力効果モデル（[docs/AERODYNAMICS.ja.md](docs/AERODYNAMICS.ja.md)）

壁と車輪の接触摩擦を調整したい場合は、`vehicle_params.json` の次の項目を編集してください。

- `drone.wheels.friction`
- `environment.wall_friction`

どちらも MuJoCo の `friction` 属性に渡される 3 要素ベクトルです。順に sliding, torsional, rolling friction を表します。`drone.wheels.friction` の sliding 値は、壁面転がり中に車輪が支えられる横方向力を決める μ でもあります。

論文向けの運用では、`baseline` と `hil` は単なるオプション差ではなく別モードとして扱うのを推奨します。現在の意味づけ、推奨指標、現実装の境界は [docs/FIDELITY_MODES.ja.md](docs/FIDELITY_MODES.ja.md) を参照してください。

Python 側は `vehicle_params.json` と `wheeled_uav.template.xml` から MuJoCo 用 XML を生成して読み込みます。既定では出力先は `build/generated_xml/` で、`instance_id = 0` では `wheeled_uav.generated.xml`、それ以外では `wheeled_uav.generated.instance_N.xml` を使います。

MATLAB 側は同じ `vehicle_params.json` から、制御に必要な質量、重力、アーム長、反トルク係数、最大推力、推力換算係数に加え、hover/contact 系 controller の既定ゲインも読み込みます。

controller の既定値は `vehicle_params.json` の `controller` に集約しています。現在は少なくとも以下をここから読めます。

- `desired_heading`
- `position_gain`, `velocity_gain`
- `attitude_gain`, `angular_velocity_gain`
- `position_error_limit_m`, `max_tilt_deg` — MATLAB / Python 共通 hover 制御則の大変位セーフティクランプ（既定 1.5 m / 35°、`<= 0` で無効化）。PD へ入る位置誤差ノルムを飽和させ、目標推力ベクトルに鉛直成分の下限と傾き上限を課します。viewer 上で機体を大きく引きずっても、推力喪失や反転ではなく、傾きが抑えられた復帰機動になります。

編隊制御の既定値は `vehicle_params.json` の `formation` へ集約しています。現在は少なくとも以下をここから読めます。

- `num_uavs`
- `spawn_radius`
- `base_height`
- `centroid_target_xy`
- `formation_radius`
- `centroid_gain`
- `formation_gain`
- `duration_seconds`
- `idle_sleep_seconds`
- `status_display_interval`
- ゲイン・クランプの上書き（`desired_heading`, `position_gain`, ..., `position_error_limit_m`, `max_tilt_deg`）— 既定値は `controller` セクションの値

編隊実行では、既定で 1 ファイルの `formation_bundle*.mat` だけを残します。bundle と UAV ごとの `.mat` を両方残したい場合は、`multi_uav_formation_controller('formation_log_mode', 'bundle_and_individual')` を使ってください。bundle の中では、順序付きの `formation_log.logs` に加えて、`formation_log.uavs.uav_1`, `formation_log.uavs.uav_2` のような名前付きフィールドでも各 UAV のログへアクセスできます。

</details>

<details>
<summary>曲面環境</summary>

`vehicle_params.json` の `environment.surface` で、平面または `z = h(x, y)` 型の曲面を指定できます。

日常的な切替は `environment.surface.mode` を変えるのが一番簡単です。

- `"mode": "plane"` または `"mode": "floor"` で床
- `"mode": "slope"` で傾斜面
- `"mode": "paraboloid"`, `"mode": "sinusoidal"`, `"mode": "gaussian"` も同様に切替可能

たとえば床と傾斜面の切替はこの 1 行だけで済みます。

```json
"surface": {
	"mode": "plane",
	...
}
```

または

```json
"surface": {
	"mode": "slope",
	...
}
```

`mode` は簡易トグル用で、詳細形状は従来どおり `type`, `plane`, `height_function`, `parameters` の設定が使われます。

既定では `follow_surface_for_initial_position = true` なので、機体の `initial_position.z` はその地点の地表高さに対する相対高さとして扱われます。曲面や盛り上がった地形に切り替えたときに、初期状態で機体が地面へ埋まるのを防ぐためです。必要なら `false` にして従来どおり絶対座標として扱えます。

接地初期化では、左右車輪の接地条件からロール角を決め、さらに地形の `dh/dx` から初期ピッチ角も入れます。車輪と地表の初期クリアランスは `environment.surface.initial_wheel_contact_clearance` で調整できます。既定値は `0.0001` m です。

`type = "plane"`:

- 従来どおり MuJoCo の plane geom を使います
- 既定の平面フロアと同じ接触挙動です

`type = "height_function"`:

- Python 側が `height_function` の設定から地形を生成します
- `flat` と `slope` のように平面で表せる場合は MuJoCo の plane geom に自動変換します
- `paraboloid` や `sinusoidal` のような非平面形状は MuJoCo の hfield として埋め込みます
- 現在対応している関数名は `flat`, `slope`, `paraboloid`, `sinusoidal`, `gaussian` です

代表例:

```json
"surface": {
	"type": "height_function",
	"material": "floor_mat",
	"solref": [0.002, 1.0],
	"contact": {
		"contype": 1,
		"conaffinity": 1
	},
	"height_function": {
		"x_range": [-3.0, 3.0],
		"y_range": [-3.0, 3.0],
		"grid_resolution": [121, 121],
		"name": "slope",
		"parameters": {
			"z_offset": 0.0,
			"slope_x": 0.08,
			"slope_y": 0.0
		}
	}
}
```

将来的に数式文字列そのものを評価する方式ではなく、当面は named function とパラメータ指定にしています。これは安全性と保守性を優先したためです。

`gaussian` はガウス分布状の盛り上がりやくぼみを作るための関数です。代表的なパラメータは次です。

- `amplitude`: 山の高さ。負にするとくぼみになります
- `center_x`, `center_y`: 山の中心位置
- `sigma_x`, `sigma_y`: 山の広がり

</details>

<details>
<summary>固定傾斜ロータと入力モード切替</summary>

モデル側だけで固定傾斜ロータを表現したい場合は、`vehicle_params.json` の `actuation.rotors` を使います。各ロータは body frame での位置と推力軸を持ちます。推力軸は正規化されていない値を書いても読み込み時に正規化されます。

```json
"actuation": {
	"command_mode": "omega",
	"max_rotor_thrust": 20.0,
	"yaw_moment_ratio": 0.02,
	"thrust_coefficient": 2.0e-5,
	"rotors": [
		{
			"name": "fr",
			"position_body": [0.08, -0.1, 0.012],
			"thrust_axis_body": [-0.14834, 0.197905, 0.968912],
			"yaw_moment_ratio": 0.02,
			"spin_sign": 1
		}
	]
}
```

`spin_sign` は反トルクの向きを表します。`1` と `-1` を使ってください。

既定の `vehicle_params.json` では、4 つのロータ推力軸はすべて通常の上向き (`[0, 0, 1]`) です。

固定傾斜ロータの実装例は [vehicle_params.tilted_rotor_example.json](vehicle_params.tilted_rotor_example.json) に置いてあります。これは前後左右へ対称に約 14.3 度だけ外向きへ傾けた 4 ロータ例です。`uv run mujoco-wheeled-uav-simulator check-model` で生成結果を確認できます。controller 側の配分行列も同じロータ幾何から導出されるため、傾斜配置でも生成される MuJoCo actuator と整合した推力配分になります。

現在の既定値は `vehicle_params.json` の `actuation.command_mode = "omega"` です。ここを切り替えると MATLAB 側の送信形式が変わります。

```json
"actuation": {
	"command_mode": "omega"
}
```

`command_mode = "thrust"`:

- MATLAB は `rotor_thrusts` を直接送信します
- 既存の動作と互換です

`command_mode = "omega"`:

- MATLAB はコントローラ内部で計算したロータ推力を `rotor_omega = sqrt(T / k_f)` に変換して送信します
- Python は `vehicle_params.json` の `actuation.thrust_coefficient` を使って再び推力へ変換し、MuJoCo へ適用します
- モータ一次遅れ・レート制限は `actuator_dynamics` でオプション指定できます（[docs/FIDELITY_MODES.ja.md](docs/FIDELITY_MODES.ja.md) 参照）

</details>

<details>
<summary>MATLAB からの自動起動</summary>

各サンプルコントローラは `auto_launch` オプションを受け付け、MATLAB から MuJoCo シミュレータを直接起動できます。実運用では責務分離とデバッグ性の観点から既定値は無効で、シミュレータを別ターミナルで起動する運用を推奨します。

```matlab
hovering_controller('auto_launch', true)
wall_demo_controller('auto_launch', true)
```

有効時は以下の順で動作します。

1. MATLAB 側が UDP 受信ポートを確保する（`uavsim.Session`）
2. シミュレータルートに `.venv\Scripts\python.exe` があれば、`uavsim.Launch` がそのインタプリタで `python -m wheeled_uav.cli simulate ...` を起動する
3. `.venv` が無ければ `uv run --project <simulator_root> mujoco-wheeled-uav-simulator simulate ...` にフォールバックする

`'shutdown_on_exit', true` を併用すると、コントローラ終了時に起動したシミュレータプロセスも停止します。

</details>

<details>
<summary>ログ保存と解析</summary>

MATLAB 側の `simulation_logger` が、以下を `.mat` に保存します。

- `meta`: 保存時刻や保存理由などのメタデータ
- `config`: 制御ゲイン、配分行列、目標値などの設定
- `state`: 時刻、位置、速度、角速度、姿勢行列
- `control`: 各ロータ推力と、必要に応じて各ロータ角速度
- `reference`: 目標位置
- `contact`: 接触数、接触力サマリ、各時刻の接触詳細

保存先は `logs/` です。`logs/` は生成物なので `.gitignore` に含めています。

保存モードは各コントローラ実装内の `uavsim.RunOptions.build_logging_options` 呼び出し（例: `matlab/controllers/hovering_controller_impl.m`）で変更できます。

- `finalize`: 終了時に 1 回保存
- `periodic`: 指定秒数ごとに上書き保存
- `periodic_and_finalize`: 定期保存しつつ終了時にも保存

`contact` には全接触のサマリに加えて、`left_wheel`、`right_wheel`、`surface`、`wall` の接触力サマリも入ります（壁面走行では `wall` グループの `total_normal_force` が押し付け力の実測値になります）。`contact.details` には各時刻の接触ごとの相手 geom 名、接触位置、貫入距離、接触座標系での力・トルク、法線力が入ります。曲面地形との接触では `surface_contact`、`surface_height`、`surface_normal` も保存されます。MuJoCo が各ステップで計算した接触力をそのまま保存しており、衝撃インパルスの後処理は行いません。

</details>

<details>
<summary>トラブルシュート</summary>

- `uv` が見つからない場合は、`uv` のインストール後にシェルを開き直してください。
- MuJoCo ウィンドウが表示されない場合は、GUI を利用できるローカル環境で実行しているか確認してください。
- MATLAB 起動時にポート競合が出る場合は、同じ MATLAB セッション内に古い `udpport` が残っていないか確認してください。
- 自動起動を使う場合は、`.venv` または `uv` のどちらかで Python 実行経路が通っている必要があります。

</details>

## 補足

- `vehicle_params.json`・`wheeled_uav.template.xml` はリポジトリ直下に置く前提です。Python パッケージと MATLAB 実装はそこを基準に参照します。Python 側のパスは環境変数 `WHEELED_UAV_PARAMS_PATH`・`WHEELED_UAV_XML_TEMPLATE_PATH`・`WHEELED_UAV_GENERATED_XML_DIR` でも上書きできます。
- 機体メッシュは任意です。パラメータファイルの `drone.mesh.file` に STL を指定すると表示されます。相対パスは「パラメータファイルのディレクトリ→テンプレートのディレクトリ」の順に解決されるため、オーバーレイ側リポジトリが自分のメッシュを同梱できます。`drone.mesh` 無しではプリミティブ形状で表示されます。
- UDP の既定は `127.0.0.1` です。リモート／HIL 構成ではシミュレータ側 `--bind-ip` / `--state-target-ip`、コントローラ側 `--bind-ip` / `--target-ip` で上書きします（上記リモート節参照）。
- MATLAB 側は起動時に古い `udpport` 残骸を解放するようにしています。