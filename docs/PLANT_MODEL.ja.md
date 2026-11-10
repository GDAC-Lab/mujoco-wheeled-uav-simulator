# プラントモデル

`vehicle_params.json` の `plant` セクションは、コントローラの指令と剛体の物理の間、物理とコントローラが測るものの間を表します。ロータの遅れ（むだ時間と応答の遅れ）、推力の較正誤差と上限、電池、抗力と地面効果、データで当てはめた補正、センサの遅れ・雑音・バイアス・欠測です。トップレベルの `randomization` セクションは、実行ごとにパラメータを引きます（モンテカルロ法での評価のため）。

すべて任意です。`plant` セクションが無ければ（または `"enabled": false` なら）以前とまったく同じに動き、各サブセクションも書いたときだけ効きます。プラントモデルは物理の一部なので、**どの fidelity モード・どのペーシングでも**効きます（HIL のときだけ効く `actuator_dynamics`・`sensor_fidelity` の代わりになるもので、両方を同時に使うとエラーにします）。PX4 HITL のブリッジでも効きます（`plant.sensors` を除く。機体に送るセンサは真の状態から作るため）。遅れは**シミュレーション時間**で、決定的で、`realtime`・`accelerated`・`lockstep` のどれでも同じです（`network_fidelity` の遅れは実時間です。[TIMING.ja.md](TIMING.ja.md)）。

知らないキーはエラーにします（綴りを間違えたキーが黙って既定値のままになるのを防ぐため）。`_` で始まるキーはコメントとして読み飛ばし、`"enabled": false` で消さずに止められます。

物理の 1 ステップ（`simulation.timestep`）ごとに:

```text
指令 ──► むだ時間 ──► 倍率・上下限 ──► 回転数の一次遅れ ──► 地面効果 ──► MuJoCo
(パケット) (delay_s)   (thrust_scale,   (時定数 上げ/下げ)    (ロータごとの倍率)  のアクチュエータ
                       idle, max(V))
                                 電池 ◄── 各ロータの推力
抗力（ロータ・機体・風）＋データで当てはめた補正 ──► 機体への力とトルク
MuJoCo のセンサ ──► 履歴 ──► 遅れ・雑音・バイアス・欠測 ──► 状態パケット
```

## アクチュエーション（`plant.actuation`）

| キー | 意味 | 既定 |
| --- | --- | --- |
| `delay_s` | 指令のむだ時間（物理ステップに丸める） | `0` |
| `motor_time_constant_up_s` | 回転数が上がるときの時定数 | `0`（遅れなし） |
| `motor_time_constant_down_s` | 下がるときの時定数 | 上げと同じ |
| `thrust_scale` | 出る推力 ÷ 指令の推力（較正の誤差） | `1` |
| `rotor_thrust_scales` | ロータごとの追加の倍率（4 つ、`actuation.rotors` の順） | なし |
| `max_thrust_n` | 1 ロータの推力の上限（電池の基準電圧で） | `actuation.max_rotor_thrust` |
| `idle_thrust_n` | 回っているロータの推力の下限。指令 0 ではロータは止まる | `0` |

回転数 ω は定常値に一次遅れで近づき、推力は `T = k_f ω²` です。シミュレータは ω に比例する `sqrt(T)` で厳密に積分するので、結果は `k_f` に依りません:

```text
s* = sqrt(clip(thrust_scale · 指令, idle, max))      s = sqrt(T)
s ← s* + (s − s*) · exp(−dt / τ)       τ = s* > s なら τ_up、そうでなければ τ_down
```

`actuation.thrust_coefficient`（`k_f`）は回転数を報告するためだけに使います。ヨーモーメントはアクチュエータのギア（`yaw_moment_ratio`）を通して出る推力に比例します。ロータ自身の慣性は入れていません。

**最初の状態。** `lockstep`、プロセス内の `Simulator`、`simulation.hold_until_first_command` で固定が解けたときは、ロータは最初の指令の定常状態から始まります（すでに飛んでいる機体）。それ以外（地面に置いた機体）は止まった状態から回り始めます。

## 電池（`plant.battery`）

機体ごとに 1 パック。開放電圧は充電率に線形、電力は運動量理論、端子電圧は電力の釣り合いを解きます:

```text
V_oc = empty + (full − empty) · 充電率
P    = c · Σ T_i^1.5 + P_aux                  (c = power_coefficient_w_per_n1p5)
V    = (V_oc + sqrt(V_oc² − 4 R P)) / 2        I = P / V
充電率 ← 充電率 − I dt / (容量 · 3.6 C/mAh)
```

パックの最大電力（`V_oc² / 4R`）を超えると電圧は `V_oc / 2` に留まります。推力が電圧で決まるところ（ロータが上限にあるとき、電圧を補償しないとき）では、電圧と推力を同じステップの中で解き合わせます（ロータの遅れが無くても落ち着きます）。ロータの推力の上限は `(V / reference_voltage_v)²` に比例します。`voltage_compensation: true`（既定）では指令はそのまま推力です（PX4 の電圧つき推力曲線のように、電圧を補償するフライトコントローラ）。`false` では指令を基準電圧での較正として読み、出る推力も `(V / V_ref)²` に比例して落ちます。

| キー | 既定 |
| --- | --- |
| `full_voltage_v`、`empty_voltage_v` | `16.8`、`13.6`（4S） |
| `initial_charge` | `1.0` |
| `capacity_mah` | `1500` |
| `internal_resistance_ohm` | `0.03` |
| `power_coefficient_w_per_n1p5` | `12`（5 インチのロータ: 理想値 5.7 W/N^1.5 を、フィギュア・オブ・メリットとモータ・ESC の効率 ≈ 0.45 で割った値） |
| `auxiliary_power_w` | `0` |
| `reference_voltage_v` | `16.0` |
| `voltage_compensation` | `true` |

## 抗力と地面効果（`plant.aerodynamics`）

`v_b` を機体座標の対気速度（速度 − `wind_m_per_s`）として:

```text
F_b = −D v_b − ½ ρ A ∘ |v_b| ∘ v_b        D = diag(rotor_drag_coefficients_n_s_per_m)
                                           A = body_drag_area_m2（機体の軸ごとの抗力面積 C_d·A）
```

1 次の項は Faessler・Franchi・Scaramuzza（RA-L 2018）のロータ抗力で、小型マルチコプタの抗力の主な部分です。2 次の項は機体の圧力抗力です。どちらも重心に働きます。

`ground_effect` は各ロータの推力に Cheeseman–Bennett の係数 `1 / (1 − (R / 4z)²)` を掛けます（`max_gain` で頭打ち）。`z` はロータの地面からの高さ（`ground_height_m`、無ければ地形の高さ関数、それも無ければ 0）、`R` は `rotor_radius_m`（既定 `drone.propeller.radius`）です。

| キー | 既定 |
| --- | --- |
| `air_density_kg_per_m3` | `1.204` |
| `wind_m_per_s` | `[0, 0, 0]` |
| `rotor_drag_coefficients_n_s_per_m` | `[0, 0, 0]` |
| `body_drag_area_m2` | `[0, 0, 0]` |
| `ground_effect.rotor_radius_m`、`.max_gain`、`.ground_height_m` | プロペラ半径、`1.5`、地形 |

別のセクション `aerodynamics.wall_effect`（[AERODYNAMICS.ja.md](AERODYNAMICS.ja.md)）の壁の近くの力は、これに加えて効きます。

## データで当てはめた補正（`plant.correction`）

```json
"correction": {"file": "identified/correction.json", "scale": 1.0}
```

上の物理モデルで表しきれない力とトルク（機体座標）を、対気速度・角速度・各ロータの推力・高さなどの関数として、物理パラメータを同定した後の飛行ログに当てはめたものです（`wheeled_uav.sysid`、[SYSTEM_IDENTIFICATION.ja.md](SYSTEM_IDENTIFICATION.ja.md)）。`linear`（リッジ回帰）と `knn`（近いサンプルの平均）の 2 種類があります。出力は、データで見た範囲に切ります。ファイルの形式は `wheeled_uav/plant/correction.py` にあります。相対パスは params ファイルからの相対です。

## センサ（`plant.sensors`）

遅れは、この節が無いときにシミュレータが送る状態から数えます。その状態は、積分した状態より物理の 1 ステップ古いものです。測った状態から同定した遅れには、この 1 ステップが入ります。

| キー | 意味 | 既定 |
| --- | --- | --- |
| `pose_delay_s` | 位置と姿勢の古さ（モーションキャプチャ → 推定器） | `0` |
| `velocity_delay_s` | 速度の古さ | `0` |
| `angular_velocity_delay_s` | 角速度（ジャイロ）の古さ | `0` |
| `position_noise_std_m`、`attitude_noise_std_rad`、`velocity_noise_std_m_per_s`、`angular_velocity_noise_std_rad_per_s` | 送る状態ごとの白色雑音 | `0` |
| `angular_velocity_bias_std_rad_per_s` | 始めのジャイロのバイアス（機体・軸ごと） | `0` |
| `angular_velocity_bias_walk_rad_per_s_per_sqrt_s` | ジャイロのバイアスのランダムウォーク | `0` |
| `pose_dropout_probability` | 送る状態ごとに欠測が始まる確率 | `0` |
| `pose_dropout_mean_duration_s` | 欠測の平均の長さ（指数分布） | `0.05` |

欠測の間、位置・姿勢・速度は最後に届けた値のままで、角速度は届き続けます。遅れは物理ステップに丸めます。状態パケットの `time` は送った時刻のままで、`plant.measurement_age_s`（位置と姿勢の古さ）が付くので、コントローラはそれを見て分かります。

## 状態パケット（JSON）

プラントのセクションがある間、各機体に `plant` オブジェクトが付きます:

| フィールド | 付くとき |
| --- | --- |
| `rotor_speeds_rad_s` | ロータのモデルがあり `thrust_coefficient > 0`（双方向 DShot の回転数テレメトリに当たるもの。雑音なし） |
| `battery_voltage_v`、`battery_current_a`、`battery_charge` | 電池 |
| `plant_force_body`、`plant_torque_body` | 抗力か補正 |
| `measurement_age_s` | センサ |

`logging_config.log_actuator_stats` を付けると、`actuator` ブロックはロータのモデルの指令と出た推力を報告します。バイナリの状態パケットは測った状態を運びますが、`plant` ブロックは運びません。

## ランダム化（`randomization`）

```json
"randomization": {
  "parameters": {
    "drone.mass":                               {"uniform": [0.85, 0.95]},
    "drone.inertia":                            {"scale_uniform": [0.9, 1.1]},
    "plant.actuation.motor_time_constant_up_s": {"uniform": [0.02, 0.04]},
    "plant.actuation.thrust_scale":             {"normal": [1.0, 0.03], "clip": [0.9, 1.1]},
    "plant.aerodynamics.wind_m_per_s":          {"uniform": [[-1, -1, 0], [1, 1, 0]]}
  }
}
```

パラメータはドットでつないだパスで指定します（リストの要素は番号）。無いパスはエラーにします（`plant` の下だけは新しく作れます。そこのキーは `plant` の読み込みが確かめます）。`uniform`・`normal`・`choice` は値を決め、`scale_uniform` と `scale_normal: 標準偏差` は今の値に 1 つの倍率を掛けます（行列なら全要素に同じ倍率。`"per_element": true` で要素ごと）。`clip` で範囲を切ります。モデルを作る前に引くので、質量・慣性・形状もプラントのパラメータと同じように変えられます。引いた値は、実行の種（`simulation.seed`、`--seed`）とそのパスだけで決まります。引いた値は生成したモデルの隣（`<model>.randomization.json`）に書き出し、`check-model` も一覧を出します。

## プロセス内のシミュレーション

`wheeled_uav.sim.Simulator` は、同じ場面とプラントモデルを UDP なしで動かします。Python のコントローラが制御周期ごとに `step()` を呼び、状態パケットを dict で受け取ります。同定の道具とテストはこれを使います:

```python
from wheeled_uav.sim import Simulator

with Simulator(overrides={"plant": {"actuation": {"motor_time_constant_up_s": 0.03}}}, seed=1) as sim:
    state = sim.reset()
    while state["time"] < 2.0:
        state = sim.step([2.45, 2.45, 2.45, 2.45])
    print(sim.truth()["position"], state["position"])
```

## 入っていないもの

1 次のロータ抗力を超えるブレードのフラッピングや誘導速度の変化、ロータの慣性（回転数が変わるときのジャイロ効果と反トルク）、前進飛行での流入による推力の変化、天井効果、モータの発熱、推定器そのもの（センサのモデルは推定器の出力を表す）。これらは、当てはめたデータの範囲の中で、データで当てはめた補正が受け持ちます。
