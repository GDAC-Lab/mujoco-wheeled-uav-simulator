# Aerodynamic Effects

The simulator supports optional, config-gated aerodynamic effect models.
They are part of the plant physics: they apply in every fidelity mode
(`baseline` and `hil`) and are controlled purely by the top-level
`aerodynamics` section of `vehicle_params.json`. Everything is disabled by
default, so existing configurations are unaffected.

## Near-Wall Interaction (`wall_effect`)

Rotor wake recirculation between a vehicle and a large nearby surface
produces forces that a rigid-body model does not capture. The `wall_effect`
model applies a force along the outward wall normal (the `-x` face of the
`environment` wall box in the wall frame) to each UAV base body via
`xfrc_applied`:

```text
F = clip(C(s) * exp(-max(0, c - c_ref) / L), -F_max, F_max) * n_out
C(s) = c0 + c1 * s + c2 * s^2
```

The clip applies to the **delivered** (decayed) force, so `max_force_n` is an
absolute bound on what reaches the body regardless of clearance.

- `s`   : wall-tangential speed of the vehicle [m/s]
- `c`   : clearance between the body origin and the wall face [m]
- `n_out`: outward wall normal (from the wall face into the workspace)

Positive coefficients push the vehicle **away** from the wall (a pressing
deficit, as produced by wake recirculation at high traversal speeds);
negative coefficients model suction toward the wall. The force acts at the
body origin; induced moments are not modeled.

## Configuration

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

| Field | Meaning | Default |
|-------|---------|---------|
| `enabled` (section) | Master switch for all aerodynamic effects | `false` |
| `wall_effect.enabled` | Enable the near-wall interaction model | `false` |
| `coeff_const_n` | Constant force coefficient `c0` [N] | `0.0` |
| `coeff_linear_n_per_mps` | Linear coefficient `c1` [N/(m/s)] | `0.0` |
| `coeff_quadratic_n_per_mps2` | Quadratic coefficient `c2` [N/(m/s)^2] | `0.0` |
| `reference_clearance_m` | Clearance at which the effect is at full strength [m] | `0.2` |
| `decay_length_m` | Exponential decay length beyond the reference clearance [m] | `0.15` |
| `max_force_n` | Force magnitude clip; the model is inactive when `0` | `0.0` |

The model requires `environment.wall_position` / `environment.wall_size`;
it silently deactivates when the environment has no wall.

## State Payload

When the model is active, each UAV state gains an `aero` field so
controllers and analysis pipelines can log the applied force:

```json
"aero": {
    "wall_effect_force": [fx, fy, fz],
    "wall_clearance_m": 0.16,
    "tangential_speed_mps": 0.42
}
```
