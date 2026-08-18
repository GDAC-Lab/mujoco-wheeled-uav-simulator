# 空力効果モデル

シミュレータはオプションの空力効果モデルをサポートします。これらはプラント物理の
一部であり、fidelity モード（`baseline` / `hil`）に関係なく適用され、
`vehicle_params.json` のトップレベル `aerodynamics` セクションだけで制御されます。
既定ではすべて無効のため、既存の設定には影響しません。

## 壁近傍相互作用（`wall_effect`）

機体と大きな面が近接すると、ロータ後流の再循環により剛体モデルでは表現できない
力が発生します。`wall_effect` モデルは、壁の外向き法線（壁フレームにおける
`environment` 壁ボックスの `-x` 面）方向の力を各 UAV ベースボディに
`xfrc_applied` で加えます:

```text
F = clip(C(s) * exp(-max(0, c - c_ref) / L), -F_max, F_max) * n_out
C(s) = c0 + c1 * s + c2 * s^2
```

clip は減衰後の**印加力**にかかるため、`max_force_n` はクリアランスによらず
ボディに届く力の絶対上限です。

- `s`    : 機体の壁接線方向速度 [m/s]
- `c`    : ボディ原点と壁面のクリアランス [m]
- `n_out`: 外向き壁法線（壁面から作業空間へ向かう）

正の係数は機体を壁から**引き離す**方向（高速走行時の後流再循環による押しつけ
欠損）、負の係数は壁への吸引を表します。力はボディ原点に作用し、モーメントは
モデル化しません。

## 設定

```json
"aerodynamics": {
    "enabled": true,
    "wall_effect": {
        "enabled": true,
        "coeff_const_n": 0.0,
        "coeff_linear_n_per_mps": 0.0,
        "coeff_quadratic_n_per_mps2": 8.0,
        "reference_clearance_m": 0.2,
        "decay_length_m": 0.15,
        "max_force_n": 6.0
    }
}
```

| フィールド | 意味 | 既定 |
|-----------|------|------|
| `enabled`（セクション） | 空力効果全体のマスタースイッチ | `false` |
| `wall_effect.enabled` | 壁近傍相互作用モデルの有効化 | `false` |
| `coeff_const_n` | 定数項 `c0` [N] | `0.0` |
| `coeff_linear_n_per_mps` | 線形項 `c1` [N/(m/s)] | `0.0` |
| `coeff_quadratic_n_per_mps2` | 二次項 `c2` [N/(m/s)^2] | `0.0` |
| `reference_clearance_m` | 効果がフル強度になるクリアランス [m] | `0.2` |
| `decay_length_m` | 参照クリアランスを超えた分の指数減衰長 [m] | `0.15` |
| `max_force_n` | 力の上限クリップ；`0` でモデル無効 | `0.0` |

`environment.wall_position` / `environment.wall_size` が
ない環境では自動的に無効化されます。

## 状態パケット

モデルが有効なとき、各 UAV の状態に `aero` フィールドが追加され、
コントローラや解析パイプラインが印加力をログできます。

```json
"aero": {
    "wall_effect_force": [fx, fy, fz],
    "wall_clearance_m": 0.16,
    "tangential_speed_mps": 0.42
}
```

英語版: [AERODYNAMICS.md](AERODYNAMICS.md)
