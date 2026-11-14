# Scene extensions

The `extensions` section of the params lets your own Python code add to the scene what the
simulator does not model itself — bodies, joints, sites, and the forces on them — and take part in
every physics step: it can take commands from the controller and report a state of its own in the
state packets. A study can then add what it needs in its own repository, without changing the
simulator.

Everything is opt-in. Without an `extensions` section (or with every entry `"enabled": false`) the
simulator behaves exactly as before, down to the generated XML. With extensions, the vehicles' part
of the model is still exactly the same: the scene is compiled from the edited model description, not
from a re-written XML file.

## Declaring extensions

```json
"extensions": {
  "goal": {
    "class": "wheeled_uav.extensions.marker:Marker",
    "position": [1.0, 0.0, 1.2],
    "max_speed": 0.5
  }
}
```

Each entry has a name (a letter, then letters, digits or `_`; it is the key of the extension's
commands and state, and a MATLAB struct field) and these keys for the loader; every other key is the
extension's own configuration and goes to its constructor:

| Key | Meaning | Default |
| --- | --- | --- |
| `class` | The class, as `module:ClassName` (in-process callers may give the class itself) | required |
| `path` | A directory, or a list of them, put on `sys.path` before the import; relative paths are taken from the params file's directory | none |
| `enabled` | `false` leaves the entry out | `true` |

Names and keys starting with `_` are comments, as elsewhere in the params. Extensions are built in
the order of the section.

A params file that names a class runs that class's code. Load params files only from sources you
trust, as you would a Python script.

An extension's settings are params like any other: the `randomization` section can draw them per
run (for example `"extensions.goal.radius": {"uniform": [0.04, 0.08]}`, see
[PLANT_MODEL.md](PLANT_MODEL.md)), and `Simulator(overrides=...)` can change them.

## Writing an extension

An extension is a class constructed as `Class(name, config)`. Deriving from
`wheeled_uav.extensions.Extension` gives no-op defaults, so a class defines only the methods it needs;
setting `keys` makes any other configuration key an error, so that a misspelt key cannot pass
unnoticed. A box that the vehicles can push around, and that reports where it is:

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

Put it in a module of your repository and name it in the params:
`"class": "my_extensions:Box", "path": "."`.

The simulator calls:

| Method | When | For |
| --- | --- | --- |
| `edit_spec(spec, context)` | Once, on the `mujoco.MjSpec` of the rendered model, before it is compiled | Adding bodies, joints, sites, geoms |
| `bind(model)` | Once, on the compiled model | Looking up the ids of what it added |
| `reset(model, data)` | Once, on the fresh data (spawn poses and initial velocities set), before the first `mj_forward` | Its initial state |
| `apply_command(command)` | With the `extensions[name]` value of each command that is applied | Storing what the controller asked for |
| `before_step(model, data)` | Every physics step, after the rotor commands, body wrenches, plant model and aerodynamics are written, just before `mj_step` | Its forces, or moving what it added |
| `state(model, data)` | For each state packet | A dict (or `None`) sent as `extensions[name]`; numpy values are turned into plain numbers |

`context` (`ExtensionContext`) holds the scene's `vehicles` (each with `index`, `name` as in the state
packets, and `body`, the body with the vehicle's free joint), the physics `timestep`, the run's
`params` (after randomization; read them, do not change them), the `params_dir` and the run's
`seed`.

Rules that keep extensions from getting in each other's way:

- **Touch only what the extension added**: the `xfrc_applied` rows, joints, actuators and mocap bodies
  it created. The rotor commands, the vehicles' bodies and their `xfrc_applied` rows belong to the
  simulator (the command dispatcher, the plant model and the aerodynamics model write them).
- **Prefix the names** of what you add with the extension's name (`f"{self.name}_box"`), so that two
  entries of the same class do not clash.
- **Check before changing**: an `apply_command` that cannot use a command raises (an `ExtensionError`,
  or any `ValueError`) before it changes anything.
- **Draw from the run's seed**: an extension with random values of its own seeds its generator from
  `context.seed`, so that two runs with the same seed and inputs are the same.

## Commands and states

A command packet carries the commands at its top level, also in multi-UAV packets:

```json
{"rotor_thrusts": [2.1, 2.1, 2.1, 2.1], "extensions": {"goal": {"position": [1.0, 0.5, 1.2]}}}
```

The commands go with the rotor command they arrive with: the HIL network model delays and drops
them with it, and a stale packet's commands are not applied. A command the scene cannot use (no
extension of that name, or one that rejects the command) is ignored with a warning, printed once per
problem; the rotor command is applied all the same. Packets without `extensions` are handled exactly
as before.

The state packets carry the states at their top level, in single-UAV and multi-UAV packets alike:

```json
{"sequence": 412, "sim_time": 2.06, "position": [...], "extensions": {"goal": {"position": [1.0, 0.5, 1.2], "target": [1.0, 0.5, 1.2], "distance": [0.31]}}}
```

From MATLAB:

```matlab
command = uavsim.Protocol.build_control_command(rotor_thrusts, command_options, ...
    'extensions', struct('goal', struct('position', [1.0 0.5 1.2])));
uavsim.Protocol.send_control_command(controller_socket, command, target_ip, target_port);
% ... and in the state: state.extensions.goal.position
```

For several UAVs, the `extensions` of the per-UAV commands are sent as one object (for a name given
twice, the later UAV's command). `uavsim.Protocol.encode_control_command` and
`encode_multi_uav_control_command` return the JSON text without sending it.

To log an extension's state, register it as the logger's own extension group (a different use of
the word: caller-registered log channels) and pass the state struct on each sample:

```matlab
logger = simulation_logger(base_directory, config, options, ...
    struct('group_name', 'goal', 'channels', {{'position', 3; 'distance', 1}}));
logger.append(state, command, target_position, state.extensions.goal);
```

From Python, in process (`wheeled_uav.sim.Simulator`):

```python
with Simulator(overrides={"extensions": {"goal": {"class": "wheeled_uav.extensions.marker:Marker"}}}) as sim:
    state = sim.reset()
    state = sim.step(rotor_thrusts, extensions={"goal": {"position": [1.0, 0.5, 1.2]}})
    print(state["extensions"]["goal"]["distance"], sim.extensions["goal"].target)
```

In process, a command an extension cannot use raises instead of warning. The Python reference
controller's `build_hover_command_payload_with_metadata(..., extensions=...)` adds commands to its
packets.

## Model files

`check-model` and every run write the vehicles' model as before (`<model>.xml` under
`build/generated_xml/`) and, with extensions, MuJoCo's XML of the whole scene next to it as
`<model>.extended.xml`, for viewing and checking (`check-model` prints its path and the extension
names). XML writes some numbers rounded, so a model loaded from that file can differ from the
simulated one in the last digits; the simulator itself compiles the edited model directly.

## Where extensions run

| Runtime | Extensions |
| --- | --- |
| UDP simulator (`realtime`, `accelerated`, `lockstep`) | Every call; commands from the packets; states in the JSON packets |
| In-process `Simulator` | Every call; commands from `step(..., extensions=...)`; states in the returned state |
| PX4 HITL bridge | Every call but `apply_command` (the board sends rotor outputs only); no state packets |
| Parallel physics (`simulation.physics_threads` > 0) | Refused: each vehicle steps in a world of its own, without the extensions |
| Binary state packets (`simulation.state_format` `"binary"`) | Refused: the binary layout has no room for extension states |

While `hold_until_first_command` holds the vehicles at their spawn state, the bodies an extension
added are held at theirs too (the whole state is held). The viewer and `--record` show what the
extensions added.

## The bundled example: `Marker`

`wheeled_uav/extensions/marker.py` shows a commanded point, such as a setpoint or a waypoint: a small
sphere on a mocap body, drawn by the viewer and in recordings, without any part in the physics (no
contacts, no mass to the vehicles).

| Key | Meaning | Default |
| --- | --- | --- |
| `position` | Start [m] | `[0, 0, 1]` |
| `radius` | [m] | `0.05` |
| `rgba` | Colour | `[1, 0.5, 0, 0.6]` |
| `max_speed` | [m/s]; `null`: the marker jumps to each commanded point | `null` |

Command: `{"position": [x, y, z]}`. State: `{"position": [x, y, z], "target": [x, y, z], "distance":
[one per vehicle: from the vehicle's body origin, m]}`.
