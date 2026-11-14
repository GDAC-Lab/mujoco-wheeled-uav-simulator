# シーンの拡張（extensions）

パラメータの `extensions` セクションを使うと、シミュレータ自身が持たないもの（ボディ、関節、サイト、それらにかかる力）を自分の Python コードで場面に足し、物理の毎ステップに加われます。コントローラからの指令を受け、自分の状態を状態パケットで返すこともできます。研究ごとに必要なものを自分のリポジトリに置き、シミュレータを書き換えずに使えます。

すべて任意です。`extensions` セクションが無ければ（またはすべての項目が `"enabled": false` なら）、生成される XML まで含めて以前とまったく同じに動きます。拡張があっても、機体の部分のモデルはまったく同じです（場面は、XML を書き直して読み込むのではなく、編集したモデルの記述から直接コンパイルします）。

## 拡張の宣言

```json
"extensions": {
  "goal": {
    "class": "wheeled_uav.extensions.marker:Marker",
    "position": [1.0, 0.0, 1.2],
    "max_speed": 0.5
  }
}
```

項目の名前は、英字で始まり英数字と `_` だけからなる名前です（拡張への指令と拡張の状態のキーになり、MATLAB の構造体のフィールド名にもなるため）。次のキーは読み込み側が使い、それ以外のキーは拡張自身の設定としてコンストラクタに渡します。

| キー | 意味 | 既定 |
| --- | --- | --- |
| `class` | クラスを `module:ClassName` の形で（プロセス内から使うときはクラスそのものでもよい） | 必須 |
| `path` | import の前に `sys.path` に加えるディレクトリ（またはそのリスト）。相対パスはパラメータファイルのディレクトリから | なし |
| `enabled` | `false` でその項目を外す | `true` |

`_` で始まる名前とキーは、パラメータのほかの場所と同じくコメントです。拡張はセクションに書いた順に作ります。

クラスを名指すパラメータファイルは、そのクラスのコードを実行します。Python のスクリプトと同じく、信頼できる出どころのパラメータファイルだけを読み込んでください。

拡張の設定もほかのパラメータと同じで、`randomization` セクションで実行ごとに引けます（例: `"extensions.goal.radius": {"uniform": [0.04, 0.08]}`。[PLANT_MODEL.ja.md](PLANT_MODEL.ja.md) 参照）。`Simulator(overrides=...)` でも変えられます。

## 拡張の書き方

拡張は `Class(name, config)` で作られるクラスです。`wheeled_uav.extensions.Extension` を継承すると何もしない既定のメソッドが付くので、必要なメソッドだけを書けば済みます。`keys` を決めておくと、それ以外の設定キーはエラーになります（綴りを間違えたキーが黙って通らないように）。機体で押して動かせる箱で、自分の位置を返す例:

```python
import mujoco

from wheeled_uav.extensions import Extension, ExtensionError


class Box(Extension):
    keys = frozenset({"position", "size", "mass"})

    def __init__(self, name, config):
        super().__init__(name, config)
        self.position = [float(v) for v in self.config.get("position", [1.0, 0.0, 0.1])]
        self.size = float(self.config.get("size", 0.1))
        self.mass = float(self.config.get("mass", 0.5))
        if self.size <= 0.0 or self.mass <= 0.0:
            raise ExtensionError(f"extensions.{name}: size and mass must be positive")

    def edit_spec(self, spec, context):
        body = spec.worldbody.add_body(name=f"{self.name}_box", pos=self.position)
        body.add_freejoint(name=f"{self.name}_box_joint")
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[self.size] * 3, mass=self.mass)

    def bind(self, model):
        self.body_id = model.body(f"{self.name}_box").id

    def state(self, model, data):
        return {"position": data.xpos[self.body_id]}
```

自分のリポジトリのモジュールに置き、パラメータで `"class": "my_extensions:Box", "path": "."` と名指します。

シミュレータが呼ぶもの:

| メソッド | いつ | 何のために |
| --- | --- | --- |
| `edit_spec(spec, context)` | 1 回。生成したモデルの `mujoco.MjSpec` に対して、コンパイルの前に | ボディ・関節・サイト・ジオムを足す |
| `bind(model)` | 1 回。コンパイルしたモデルに対して | 足したものの id を引く |
| `reset(model, data)` | 1 回。作ったばかりの data（スポーン姿勢と初速度を設定済み）に対して、最初の `mj_forward` の前に | 自分の初期状態 |
| `apply_command(command)` | 適用される指令ごとに、その `extensions[name]` の値を渡す | コントローラの求めを覚える |
| `before_step(model, data)` | 物理の毎ステップ。ロータ指令・外力レンチ・プラントモデル・空力を書いた後、`mj_step` の直前 | 自分の力をかける、足したものを動かす |
| `state(model, data)` | 状態パケットごと | `extensions[name]` として送る辞書（または `None`）。numpy の値は普通の数に直す |

`context`（`ExtensionContext`）には、場面の `vehicles`（それぞれ `index`、状態パケットでの名前 `name`、機体の自由関節を持つボディ `body`）、物理の `timestep`、その実行の `params`（ランダム化の後。読むだけにして変えない）、`params_dir`、その実行の `seed` が入っています。

拡張どうしがぶつからないための決まり:

- **自分が足したものだけを触る**: 自分が作ったボディの `xfrc_applied` の行、関節、アクチュエータ、mocap ボディ。ロータ指令、機体のボディとその `xfrc_applied` の行はシミュレータのもの（指令の振り分け、プラントモデル、空力モデルが書きます）。
- **足すものの名前に拡張の名前を付ける**（`f"{self.name}_box"`）。同じクラスの項目を 2 つ置いてもぶつからないように。
- **変える前に確かめる**: 使えない指令を受けた `apply_command` は、何かを変える前に例外（`ExtensionError` か `ValueError`）を投げる。
- **乱数は実行の seed から**: 自分で乱数を引く拡張は、生成器を `context.seed` から作る。同じ seed と同じ入力なら同じ実行になるように。

## 指令と状態

指令パケットは、拡張への指令をトップレベルに持ちます（複数機のパケットでも同じ）:

```json
{"rotor_thrusts": [2.1, 2.1, 2.1, 2.1], "extensions": {"goal": {"position": [1.0, 0.5, 1.2]}}}
```

拡張への指令は、一緒に届いたロータ指令と運命を共にします。HIL のネットワークモデルはロータ指令と一緒に遅らせ・落とし、古すぎるパケットの指令は適用しません。場面が使えない指令（その名前の拡張が無い、拡張が受け付けない）は警告を出して無視し（同じ問題の警告は 1 回だけ）、ロータ指令はそのまま適用します。`extensions` の無いパケットの扱いは以前とまったく同じです。

状態パケットは、拡張の状態をトップレベルに持ちます（1 機でも複数機でも同じ）:

```json
{"sequence": 412, "sim_time": 2.06, "position": [...], "extensions": {"goal": {"position": [1.0, 0.5, 1.2], "target": [1.0, 0.5, 1.2], "distance": [0.31]}}}
```

MATLAB から:

```matlab
command = uavsim.Protocol.build_control_command(rotor_thrusts, command_options, ...
    'extensions', struct('goal', struct('position', [1.0 0.5 1.2])));
uavsim.Protocol.send_control_command(controller_socket, command, target_ip, target_port);
% 状態では state.extensions.goal.position
```

複数機のときは、各機の指令の `extensions` を 1 つのオブジェクトにまとめて送ります（同じ名前が 2 回あれば後の機体のもの）。`uavsim.Protocol.encode_control_command` と `encode_multi_uav_control_command` は、送らずに JSON の文字列を返します。

拡張の状態をログに残すには、ロガー自身の extension グループ（呼び出し側が登録するログのチャネル。同じ言葉の別の使い方です）として登録し、毎サンプルで状態の構造体を渡します:

```matlab
logger = simulation_logger(base_directory, config, options, ...
    struct('group_name', 'goal', 'channels', {{'position', 3; 'distance', 1}}));
logger.append(state, command, target_position, state.extensions.goal);
```

Python のプロセス内（`wheeled_uav.sim.Simulator`）から:

```python
with Simulator(overrides={"extensions": {"goal": {"class": "wheeled_uav.extensions.marker:Marker"}}}) as sim:
    state = sim.reset()
    state = sim.step(rotor_thrusts, extensions={"goal": {"position": [1.0, 0.5, 1.2]}})
    print(state["extensions"]["goal"]["distance"], sim.extensions["goal"].target)
```

プロセス内では、拡張が使えない指令は警告ではなく例外になります。Python の参照コントローラの `build_hover_command_payload_with_metadata(..., extensions=...)` は、パケットに拡張への指令を加えます。

## モデルのファイル

`check-model` とすべての実行は、機体のモデルをこれまでどおり書き（`build/generated_xml/` の `<model>.xml`）、拡張があれば、場面全体を MuJoCo が書いた XML をその隣に `<model>.extended.xml` として書きます。見るためと確かめるためのファイルです（`check-model` はこのパスと拡張の名前を表示します）。XML は一部の数を丸めて書くので、このファイルから読んだモデルは、シミュレートしたモデルと最後の桁で違うことがあります。シミュレータ自身は、編集したモデルを直接コンパイルします。

## 拡張が動くところ

| 実行のしかた | 拡張 |
| --- | --- |
| UDP のシミュレータ（`realtime`・`accelerated`・`lockstep`） | すべて呼ぶ。指令はパケットから、状態は JSON のパケットで |
| プロセス内の `Simulator` | すべて呼ぶ。指令は `step(..., extensions=...)` から、状態は返す状態で |
| PX4 HITL のブリッジ | `apply_command` 以外を呼ぶ（機体はロータ出力しか送らないため）。状態パケットは無し |
| 並列の物理（`simulation.physics_threads` > 0） | 使えない（エラー）。各機体が拡張の無い自分だけの世界で進むため |
| バイナリの状態パケット（`simulation.state_format` が `"binary"`） | 使えない（エラー）。バイナリの配置に拡張の状態を入れる場所が無いため |

`hold_until_first_command` で機体をスポーン時の状態に保持している間は、拡張が足したボディもその初期状態に保持します（状態全体を保持するため）。ビューアと `--record` には、拡張が足したものも映ります。

## 同梱の例: `Marker`

`wheeled_uav/extensions/marker.py` は、目標点や経由点のような、指令された点を示します。mocap ボディ上の小さな球で、ビューアと録画には映り、物理にはまったく関わりません（接触なし、機体への質量なし）。

| キー | 意味 | 既定 |
| --- | --- | --- |
| `position` | 最初の位置 [m] | `[0, 0, 1]` |
| `radius` | [m] | `0.05` |
| `rgba` | 色 | `[1, 0.5, 0, 0.6]` |
| `max_speed` | [m/s]。`null` なら指令された点へ跳ぶ | `null` |

指令: `{"position": [x, y, z]}`。状態: `{"position": [x, y, z], "target": [x, y, z], "distance": [機体ごとに、機体のボディ原点からの距離, m]}`。
