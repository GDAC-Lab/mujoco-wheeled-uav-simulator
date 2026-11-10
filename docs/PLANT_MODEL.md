# Plant Model

The `plant` section of `vehicle_params.json` models what lies between a
controller's command and the rigid-body physics, and between the physics and
what the controller measures: the rotors' delay and lag, thrust calibration
error and saturation, the battery, drag and ground effect, a data-fitted
correction, and the sensors' delays, noise, bias and dropouts. A top-level
`randomization` section draws parameters per run (Monte Carlo studies).

Everything is opt-in. Without a `plant` section (or with `"enabled": false`)
the simulator behaves exactly as before; every subsection is off unless
present. The plant model is physics, so it applies in **every fidelity mode
and pacing mode** (unlike the HIL-only `actuator_dynamics` and
`sensor_fidelity`, which it replaces; combining both is refused). The PX4
HITL bridge applies it as well, except `plant.sensors`: the board's sensors
are made from the true state. Its delays
are **simulation time**: deterministic and the same in `realtime`,
`accelerated` and `lockstep`, unlike the wall-clock delays of
`network_fidelity` ([TIMING.md](TIMING.md)).

Unknown keys are errors (a misspelt key would otherwise leave an effect at
its default without a word). Keys starting with `_` are comments, and
`"enabled": false` turns a subsection off without deleting it.

Each step of the physics (`simulation.timestep`):

```text
command ──► delay ──► scale, limits ──► rotor-speed lag ──► ground effect ──► MuJoCo
 (packet)   (delay_s)  (thrust_scale,    (time constants     (gain per rotor)   actuators
                        idle, max(V))     up / down)
                                    battery ◄── rotor thrusts
drag (rotor, body, wind) + data-fitted correction ──► force and torque on the body
MuJoCo sensors ──► history ──► delay, noise, bias, dropouts ──► state packet
```

## Actuation (`plant.actuation`)

| Key | Meaning | Default |
| --- | --- | --- |
| `delay_s` | Delay of the command, rounded to physics steps | `0` |
| `motor_time_constant_up_s` | Time constant of the rotor speed when speeding up | `0` (no lag) |
| `motor_time_constant_down_s` | … when slowing down | the up value |
| `thrust_scale` | Delivered / commanded thrust (calibration error) | `1` |
| `rotor_thrust_scales` | Extra factor per rotor (4 values, `actuation.rotors` order) | none |
| `max_thrust_n` | Largest thrust of one rotor (at the battery's reference voltage) | `actuation.max_rotor_thrust` |
| `idle_thrust_n` | Smallest thrust of a running rotor; a command of 0 stops the rotor | `0` |

The rotor speed ω follows its steady-state value as a first-order lag, with
`T = k_f ω²`; the simulator integrates it exactly in `sqrt(T)`, which is
proportional to ω, so the result does not depend on `k_f`:

```text
s* = sqrt(clip(thrust_scale · command, idle, max))      s = sqrt(T)
s ← s* + (s − s*) · exp(−dt / τ)       τ = τ_up if s* > s else τ_down
```

`actuation.thrust_coefficient` (`k_f`) is used only to report the rotor
speeds. The yaw moment follows the delivered thrust through the actuator
gear (`yaw_moment_ratio`); the rotors' own inertia is not modeled.

**Initial state.** The rotors start in the steady state of the first command
(a vehicle already flying) in `lockstep`, in the in-process `Simulator`, and
when `simulation.hold_until_first_command` releases the vehicle. Otherwise
(a vehicle resting on the ground) they start at rest and spin up.

## Battery (`plant.battery`)

One pack per vehicle. Open-circuit voltage is linear in the state of charge;
electrical power follows momentum theory; the terminal voltage solves the
power balance:

```text
V_oc = empty + (full − empty) · charge
P    = c · Σ T_i^1.5 + P_aux                  (c = power_coefficient_w_per_n1p5)
V    = (V_oc + sqrt(V_oc² − 4 R P)) / 2        I = P / V
charge ← charge − I dt / (capacity · 3.6 C/mAh)
```

Past the pack's maximum power (`V_oc² / 4R`) the voltage stays at `V_oc / 2`.
Where the thrust depends on the voltage (a rotor at its limit, or no voltage
compensation) the two are solved together within the step, so they settle
even without a rotor lag.
The maximum rotor thrust scales with `(V / reference_voltage_v)²`. With
`voltage_compensation: true` (default) the command is the thrust itself (a
flight controller that compensates for the voltage, as PX4 does with a
voltage-dependent thrust curve); with `false` it is read as calibrated at the
reference voltage and the delivered thrust also scales with `(V / V_ref)²`.

| Key | Default |
| --- | --- |
| `full_voltage_v`, `empty_voltage_v` | `16.8`, `13.6` (4S) |
| `initial_charge` | `1.0` |
| `capacity_mah` | `1500` |
| `internal_resistance_ohm` | `0.03` |
| `power_coefficient_w_per_n1p5` | `12` (5-inch rotor: ideal 5.7 W/N^1.5 over figure of merit and motor/ESC efficiency ≈ 0.45) |
| `auxiliary_power_w` | `0` |
| `reference_voltage_v` | `16.0` |
| `voltage_compensation` | `true` |

## Drag and ground effect (`plant.aerodynamics`)

With `v_b` the body-frame airspeed (velocity minus `wind_m_per_s`):

```text
F_b = −D v_b − ½ ρ A ∘ |v_b| ∘ v_b        D = diag(rotor_drag_coefficients_n_s_per_m)
                                           A = body_drag_area_m2 (drag areas C_d·A per body axis)
```

The linear term is the rotor drag of Faessler, Franchi and Scaramuzza
(RA-L 2018), the dominant drag of a small multicopter; the quadratic term is
the body's pressure drag. Both act at the centre of mass.

`ground_effect` multiplies each rotor's thrust by the Cheeseman–Bennett
factor `1 / (1 − (R / 4z)²)`, capped at `max_gain`, with `z` the rotor's
height above the ground (`ground_height_m`, else the terrain's height
function, else 0) and `R` = `rotor_radius_m` (default `drone.propeller.radius`).

| Key | Default |
| --- | --- |
| `air_density_kg_per_m3` | `1.204` |
| `wind_m_per_s` | `[0, 0, 0]` |
| `rotor_drag_coefficients_n_s_per_m` | `[0, 0, 0]` |
| `body_drag_area_m2` | `[0, 0, 0]` |
| `ground_effect.rotor_radius_m`, `.max_gain`, `.ground_height_m` | propeller radius, `1.5`, terrain |

The near-wall force of the separate `aerodynamics.wall_effect` section
([AERODYNAMICS.md](AERODYNAMICS.md)) still applies on top.

## Data-fitted correction (`plant.correction`)

```json
"correction": {"file": "identified/correction.json", "scale": 1.0}
```

A force and torque (body frame) that the physical model above misses, as a
function of the airspeed, body rates, rotor thrusts, height and so on,
fitted to flight logs after the physical parameters are identified
(`wheeled_uav.sysid`, [SYSTEM_IDENTIFICATION.md](SYSTEM_IDENTIFICATION.md)).
Two kinds: `linear` (ridge regression) and `knn` (mean of the nearest
recorded samples). Outputs are clipped to the range seen in the data. The file format is
described in `wheeled_uav/plant/correction.py`. Relative paths are relative
to the params file.

## Sensors (`plant.sensors`)

The delays count from the state the simulator publishes without this
section, which is one physics step behind the integrated state; a delay
identified from measured states includes that step.

| Key | Meaning | Default |
| --- | --- | --- |
| `pose_delay_s` | Age of position and attitude (motion capture → estimator) | `0` |
| `velocity_delay_s` | Age of the velocity | `0` |
| `angular_velocity_delay_s` | Age of the body rates (gyro) | `0` |
| `position_noise_std_m`, `attitude_noise_std_rad`, `velocity_noise_std_m_per_s`, `angular_velocity_noise_std_rad_per_s` | White noise per published state | `0` |
| `angular_velocity_bias_std_rad_per_s` | Gyro bias at the start (per vehicle and axis) | `0` |
| `angular_velocity_bias_walk_rad_per_s_per_sqrt_s` | Random walk of the gyro bias | `0` |
| `pose_dropout_probability` | Chance per published state that a dropout starts | `0` |
| `pose_dropout_mean_duration_s` | Mean length of a dropout (exponential) | `0.05` |

During a dropout the position, attitude and velocity hold their last
delivered values; the body rates keep coming. Delays are rounded to physics
steps. The state packet keeps its `time` (when it was published) and gains a
`plant.measurement_age_s` (age of the position and attitude) so that a
controller can tell.

## State packet (JSON)

Each vehicle gains a `plant` object while a plant section is active:

| Field | When |
| --- | --- |
| `rotor_speeds_rad_s` | rotor model and `thrust_coefficient > 0` (like bidirectional-DShot RPM telemetry, without its noise) |
| `battery_voltage_v`, `battery_current_a`, `battery_charge` | battery |
| `plant_force_body`, `plant_torque_body` | drag or correction |
| `measurement_age_s` | sensors |

With `logging_config.log_actuator_stats` the `actuator` block reports the
commanded and the delivered thrust of the rotor model. Binary state packets
carry the measured state but not the `plant` block.

## Randomization (`randomization`)

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

Parameters are named by their dotted path (list entries by index); a path
that does not exist is an error, except under `plant`, whose own parser
checks the keys.
`uniform`, `normal` and `choice` set the value; `scale_uniform` and
`scale_normal: std` multiply the present value by one factor (every number
of a matrix by the same factor, unless `"per_element": true`); `clip` bounds
the result. The values are drawn before the model is built, so masses,
inertias and geometry can be randomized as well as plant parameters. A
parameter's draw depends only on the run's seed (`simulation.seed`,
`--seed`) and its path. The draws are written next to the generated model
(`<model>.randomization.json`) and listed by `check-model`.

## In-process simulation

`wheeled_uav.sim.Simulator` runs the same scene and plant model without UDP:
a Python controller calls `step()` once per control period and gets the
state packet as a dict. It is what the identification tools and the tests
use:

```python
from wheeled_uav.sim import Simulator

with Simulator(overrides={"plant": {"actuation": {"motor_time_constant_up_s": 0.03}}}, seed=1) as sim:
    state = sim.reset()
    while state["time"] < 2.0:
        state = sim.step([2.45, 2.45, 2.45, 2.45])
    print(sim.truth()["position"], state["position"])
```

## What it does not model

Blade flapping and induced-velocity changes beyond the linear rotor drag;
rotor inertia (gyroscopic and reaction torques of a changing rotor speed);
thrust changes with the inflow (forward flight); the ceiling effect; motor
heating; the estimator itself (the sensors model its output). These are what
the data-fitted correction is for, within the range of the data it was
fitted to.
