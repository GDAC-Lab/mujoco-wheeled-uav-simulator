# 推進系較正データの読み込み（実測値でシミュレータを更新する）

ベンチ推力試験の同定結果をシミュレータに取り込み、**実測に基づく推進系パラメータでゲイン調整**できるようにする仕組みです。下記スキーマの JSON を出力できれば、試験環境は問いません。

## 使い方

1. 推力試験のフィッティング処理で **`thrust_<日付>_<条件>_sim.json`** を出力する。
2. そのファイルを利用側リポジトリ等に置き、`vehicle_params.json` から参照する:

```json
"actuation": {
    "command_mode": "omega",
    "thrust_coefficient": 2.0e-5,
    "yaw_moment_ratio": 0.02,
    "calibration_file": "configs/thrust_20260101_myconfig_sim.json",
    ...
}
```

3. `load_vehicle_params()` が読み込み時に較正値を上書きし、1行のログを出す:

```
calibration: thrust_..._sim.json -> thrust_coefficient=1.72386e-06, yaw_moment_ratio=0.0200054
```

- 相対パスは **vehicle_params.json のあるディレクトリ基準**で解決される。
- 上書き結果と出所は `params["calibration_applied"]`（file / source / applied）に記録され、パラメータの根拠となった試験セッションを後から追跡できる。

## 上書きされる値（スキーマ `uav-propulsion-calibration/1`）

| `sim_params` のキー | 反映先 | 物理量 |
|---|---|---|
| `thrust_coefficient` | `actuation.thrust_coefficient` | kf [N·s²/rad²]（F = kf·Ω²） |
| `yaw_moment_ratio` | `actuation.yaw_moment_ratio` **＋全 `rotors[].yaw_moment_ratio`** | km/kf [m]（PX4 `CA_ROTORn_KM` 相当） |
| `motor_tau_ms` | `actuator_dynamics.motor_tau_ms` | モータ一次遅れ [ms] |

- `null`（未測定）の項目は**スキップ**され、vehicle_params の値が残る。
- `rotors[]` の個別 `yaw_moment_ratio` も較正値で統一される（ベンチ試験は推進系一式の単一値を測る前提のため。個別に差を付けたい場合は、較正適用後の値を手で編集せず、較正ファイル側を分けること）。

## ⚠ kf を電圧モデルから換算する場合の注意

回転数 Ω を直接計測しないベンチ試験では、kf を実効電圧モデル `T = c2(δV)² + c1(δV) + c0` と `Ω ≈ Kv·δV` から

```
kf = c2 / Kv_rad²   （Kv_rad = motor_kv × 2π/60）
```

のように**暫定換算**することになる。負荷時のすべりの分だけ**過小になりうる**ため、適用後は必ず単位系の妥当性を確認すること:

- ホバリング回転数 `Ω_hover = √(mg/(4·kf))` が実機と桁で合うか
- ホバリング時のスロットル／総推力が実機ログと合うか

Ω を直接計測できる環境（ESC テレメトリ等）では、実測 kf をそのまま `thrust_coefficient` に入れればよい（スキーマ変更は不要）。

なお `thrust_coefficient` は **omega コマンドモードの Ω↔推力換算とレート制限にのみ**使われる。thrust コマンドモードで直接推力を送る場合、kf のズレはシミュレーション力学に影響しない。

## 関連

- テスト: `tests/test_calibration.py`
- 実装: `wheeled_uav/calibration.py`（`load_vehicle_params()` から適用）
