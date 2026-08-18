# Time synchronization model

This simulator exchanges UDP state packets with external controllers (MATLAB, Python, etc.). Three clocks appear in logs:

| Clock | Source | Use |
| --- | --- | --- |
| **Simulator time** `state.time` | MuJoCo `data.time` after `mj_step` | **Single source of truth** for control, mission timeline, integrators, differentiators |
| **Wall clock** | OS monotonic / UTC | **Simulator pacing only**; diagnostics (`timing.sim_wall_skew_seconds`, packet `age_ms`) |
| **Controller wall time** | Controller process clock | Status printing interval only — **never** used in control laws |

Canonical implementation: [`wheeled_uav/timing.py`](../wheeled_uav/timing.py).

## Design rules (do not break)

1. **Never block the controller waiting for wall clock to catch up to simulator time.** That produces dead time and makes the system feel slower than real time when RTF &lt; 1.
2. **Mission segments and mode switches use `state.time` (seconds of simulation), not wall clock.**
3. **Control integrators use `Δ(state.time)` between consecutive unique UDP samples**, clamped to the publish cadence (`timestep × state_publish_every_n_steps`). Use `compute_control_dt_seconds()` in Python or the equivalent in your controller.
4. **Real-time pacing lives only in the simulator** (`StepPacer` + Windows 1 ms timer resolution via `high_resolution_os_timer()`). Controllers react as fast as new UDP states arrive.
5. **Skip duplicate UDP reads.** When the controller polls faster than the simulator publishes, the same `(sequence, time)` sample may be read twice. Run control **once per new sample** (`StateSampleTracker` / `udp_state_is_new`).

## Logical sync vs pacing mode (do not conflate)

**Logical synchronization** (always on) and **pacing mode** (simulator-only switch) are separate concepts.

| Concept | Meaning | Switch |
| --- | --- | --- |
| **Logical sync** | Advance control/mission/integrators on `state.time`; skip duplicate UDP; hold-last commands. | Always |
| **`realtime` pacing** | `StepPacer` tracks wall clock (RTF ≈ 1). For HIL and interactive runs. | `simulation.pacing_mode` |
| **`accelerated` pacing** | No wall-clock waits; physics runs as fast as the host allows. For CI and batch sweeps. | Same |
| **`lockstep` pacing** | Deterministic co-simulation: publish state, **block until a control command arrives**, advance one control period, repeat. Exactly reproducible runs; slow controllers never fall behind. | Same |

In `lockstep`, wall-clock diagnostics lose their usual meaning: `realtime_factor` and `sim_wall_skew_seconds` track how fast the controller responds rather than physical pacing, and `simulation.hold_until_first_command` is irrelevant because physics only advances on received commands anyway. The controller-side rules above stay identical — a compliant controller works in all three modes unchanged.

**Avoid:** a controller-side “sync mode” that pauses on `sim_wall_skew` or `state.time` to align wall clocks. That introduces dead time and makes the loop feel slower than real time.

For remote HIL, use **`realtime`**: the simulator owns wall-clock pacing; the controller never waits and reacts event-driven to new UDP samples.

## Data flow

```
Simulator (physics at simulation.timestep, default 1 kHz)
  ├─ mj_step every timestep
  ├─ publish state every N steps (simulation.state_publish_every_n_steps)
  │    payload: time, sequence, wall_time_send_ns, timing{...}, realtime_factor
  ├─ apply latest UDP command (hold-last between updates)
  └─ `pacing_mode=realtime`: `StepPacer` tracks wall clock (RTF ≈ 1)
  └─ `pacing_mode=accelerated`: no pacing (run ahead of wall clock)
  └─ `pacing_mode=lockstep`: block until a command arrives before each control period

External controller
  ├─ read latest UDP datagram (non-blocking)
  ├─ skip duplicate (sequence, time) samples
  ├─ compute control once per new sample
  └─ send command immediately (no wall/sim alignment sleep)
```

## Configuration (`vehicle_params.json`)

| Field | Meaning | Default |
| --- | --- | --- |
| `simulation.timestep` | Physics step (s) | `0.001` |
| `simulation.state_publish_every_n_steps` | UDP decimation factor | `5` → 200 Hz at 1 kHz physics |
| `simulation.viewer_fps` | `viewer.sync()` rate cap (interactive runs) | `60` |
| `simulation.pacing_mode` | Wall-clock pacing | `realtime` (HIL/interactive), `accelerated` (fast batch), or `lockstep` (deterministic co-simulation) |
| `simulation.hold_until_first_command` | Freeze vehicles at the spawn pose until the first control command is applied (reproducible initial condition for scripted runs) | `true` in the bundled config; set to `false` by presets that start resting on the ground; no effect in `lockstep` |

CLI overrides: `simulate --pacing-mode accelerated`, `simulate --no-hold-until-first-command`

**Control period** = `timestep × state_publish_every_n_steps` (e.g. 5 ms at defaults).

Controller-side nominal dt (in project-repo controllers layered on top of this simulator) should match this value; derive it from the state packet's `timing.control_period_seconds` rather than hardcoding.

## State packet `timing` block

Every published state includes a `timing` object (see `SessionTimingTracker.snapshot()`):

| Field | Meaning |
| --- | --- |
| `physics_timestep_seconds` | MuJoCo integrator step |
| `control_period_seconds` | Expected controller sample period |
| `publish_every_n_steps` | State decimation factor |
| `pacing_mode` | `realtime`, `accelerated`, or `lockstep` |
| `session_wall_elapsed_seconds` | Wall time since session start |
| `session_sim_elapsed_seconds` | Simulator time since session start |
| `sim_wall_skew_seconds` | `session_sim_elapsed − session_wall_elapsed` (meaningful in `realtime`) |
| `realtime_factor` | Rolling sim/wall rate (≈1.0 in `realtime`) |

Top-level `realtime_factor` duplicates the rolling estimate for backward compatibility.

## Diagnostics

Healthy interactive run:

- `realtime_factor` (rtf) ≈ **1.0**
- `sim_wall_skew_seconds` (skew) ≈ **0**
- `packet_age_ms` (age) typically **&lt; 5 ms** on localhost

If `rtf < 0.9`, the simulator prints a warning every 5 s. Mitigations:

- Increase `simulation.state_publish_every_n_steps`
- Disable `logging_config.include_contact_details`
- Lower `simulation.viewer_fps`
- Run headless for timing-faithful batch evaluation

## Controller helpers

| Language | Module | Key APIs |
| --- | --- | --- |
| Python (simulator) | `wheeled_uav.timing` | `StateSampleTracker`, `build_pacer`, `compute_control_dt_seconds`, `extract_sync_metrics`, `parse_simulation_timing` |
| Python (hover) | `wheeled_uav.controllers.hover` | Uses `StateSampleTracker`; status shows rtf / age / skew |
| MATLAB (submodule) | `uavsim.Protocol` | `sim_time_seconds`, `udp_state_is_new`, `build_sync_metrics` |

Project repositories that consume this simulator as a submodule typically wrap the same helpers in their own controller classes; those live outside this repo.

## Remote notes

When the controller runs on a separate machine:

- Control laws still use **`state.time` only**.
- `packet_age_ms` includes network latency; large age is expected.
- `sim_wall_skew_seconds` still reflects **simulator-side** pacing, not network delay.
- For compute-budget studies at RTF ≈ 1, run simulator and controller on machines that can sustain the configured publish rate.
