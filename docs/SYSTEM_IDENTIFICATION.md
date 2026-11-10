# System Identification

From flight logs to the [plant model](PLANT_MODEL.md), and back: the
simulator's estimators fit the rotor actuation, the drag and the battery to a
flight, optionally a data-fitted correction to what remains, and the
**k-step prediction error** measures how well a simulator (with or without
those parameters) reproduces a flight it was not fitted to. The same code
runs on simulator flights, where the true plant is known (the tests check
that the estimators recover it), and on real flights.

```text
real flight (.ulg) ── convert-ulog ──┐
                                     ├─► flight log (.npz) ── identify ──► plant section (+ correction)
simulator ── excite ─────────────────┘                    └── predict ◄── simulator with that plant
```

## Flight logs

`wheeled_uav.flightlog.FlightLog`: one time base, named channels, metadata,
saved as `.npz` (`wheeled_uav.flightlog/1`). World z up (NWU for PX4 logs),
body FLU, quaternions `[w, x, y, z]` body to world. Channels:

| Channel | Width | Content |
| --- | --- | --- |
| `position`, `quaternion` | 3, 4 | required |
| `velocity` | 3 | world velocity (else differentiated from the position) |
| `angular_velocity_body` | 3 | body rates |
| `rotor_thrust_command` | 4 | commanded thrust per rotor [N], `actuation.rotors` order — needed for the actuation |
| `rotor_command` | 4 | normalized motor command, when no thrust curve is known |
| `rotor_speed` | 4 | measured rotor speed [rad/s] |
| `battery_voltage`, `battery_current` | 1, 1 | needed for the battery |
| `specific_force_body` | 3 | accelerometer (else derived from the velocity) |
| `valid` | 1 | 0 inside a source dropout (excluded from the fits) |

Commands are held between samples, measurements interpolated.

**From PX4.** `convert-ulog flight.ulg --out flight.npz [--mapping mapping.json]`
(`uv sync --extra logs` for pyulog). Each channel is interpolated from its
topic's own time stamps (not resampled to the nearest sample of another
topic); samples inside a topic's gap (longer than 3 median periods) are
marked invalid. The default mapping reads `vehicle_local_position`,
`vehicle_attitude`, `vehicle_angular_velocity`, `vehicle_acceleration`,
`actuator_motors`, `esc_status` and `battery_status`; a mapping file
overrides topics and fields, `motor_order` (log motor for each simulator
rotor) and `thrust_curve` (`T = Σ c[i][j] u^i V^j`, e.g. a bench
calibration's 2-D polynomial in command and voltage), which turns the motor
commands into `rotor_thrust_command`. Without a thrust curve only the drag
(and the battery) can be identified.

**From the simulator.** `excite --out flight.npz [--plant plant.json]
[--duration 20] [--measured]` takes off and follows a Lissajous set-point
(about 1 m/s, for the drag) with a random collective perturbation switching
every 40–160 ms (for the delay and the rotor time constants) and a small
differential one (attitude). It is also the template of a real excitation
flight: these are the motions the estimators need.

## Estimators

`identify flight.npz [--params-file ...] [--correction knn|linear] --out result.json`
runs what the log supports and writes the fits with their residuals, and the
`plant` section they imply (`--start`/`--end` pick a window; fit only
in-flight stretches without wall or ground contact).

| Fit | Model | Needs |
| --- | --- | --- |
| actuation | `m f_bz = s Σ lag(delay(T_cmd)) − d_z v_bz + b`, the plant model's rotor lag in `sqrt(T)` with separate up and down time constants; grid search over delay and time constants, least squares for the rest | `rotor_thrust_command` |
| drag | `m f_b = F_thrust,b − D v_b (− q \|v_b\| v_b) + b` per body x, y; non-negative least squares | velocity |
| battery | `V = V_oc0 − R I − k Q` and `V I = c Σ T^1.5 + P_aux` | voltage, current |
| correction | k-nearest neighbours or ridge regression of the force left over by the fits above | actuation |

`m` is the vehicle's mass from the params (the plant's thrust scale also
absorbs a mass error). `f_b` is the accelerometer's specific force or the
derivative of the velocity; the model is averaged the same way (the central
difference averages the force over ±1 sample), so the timing is not biased.

On simulated flights the estimators recover delay (to the physics step), up
and down time constants (within 1–3 %), thrust scale (0.3 %), drag (3 %) and
battery resistance and power coefficient (1–3 %) (`tests/test_sysid.py`).

## k-step prediction error

`predict flight.npz --plant result.json --horizons 0.05 0.1 0.2 0.5`
starts the simulator from the logged state every `--stride` seconds, replays
the logged rotor commands (the controller is not in the loop), and reports
position, velocity and attitude errors after each horizon (mean, median,
90th percentile, maximum). Before each start the plant's rotor model runs
through the previous 0.3 s of commands. Compare plant models on flights they
were **not** fitted to. On a simulated flight with a known plant, the
identified plant predicts 0.5 s ahead within 2 mm, the rigid-body model
without a plant drifts 6–8 cm.

## Recommended real flights

1. Weigh the vehicle; measure the inertia (bifilar pendulum) — the estimators take the mass as known.
2. Hover and excitation flights inside the motion-capture volume: collective
   steps and pseudo-random perturbations (actuation), set-point sweeps up to
   the speeds of the experiments (drag), at several battery levels (battery).
3. Log at the full rate: `vehicle_acceleration`, `vehicle_angular_velocity`,
   `actuator_motors`, `battery_status`, and `esc_status` with RPM telemetry
   (bidirectional DShot) when available.
4. Identify on some flights, check the prediction error on others, report both.
