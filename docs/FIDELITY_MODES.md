# Fidelity Modes

This document defines how `baseline` and `hil` runs should be interpreted in this repository.

## Purpose

The repository separates two different evaluation axes:

- `physical fidelity`: how close the simulator stays to the intended reference physics model under idealized communication and measurement assumptions
- `real-time fidelity`: how well the closed loop behaves when wall-clock timing, packet delay, jitter, packet loss, and stale-command handling are part of the runtime

These should not be merged into one vague “fidelity” claim. A paper or report should state which mode is used and which metrics are evaluated — and, as the other independent axis, which `simulation.pacing_mode` (`realtime`, `accelerated`, or `lockstep`) the runs used; see [TIMING.md](TIMING.md).

## Baseline Mode

Use `baseline` for reproducible reference runs with minimal runtime disturbance.

Expected behavior:

- no injected network delay
- no injected packet loss
- no injected jitter
- no additional sensor degradation
- no additional actuator degradation
- packet metadata stays present so that logs remain structurally compatible with `hil`

Typical use cases:

- controller development under idealized assumptions
- reference plots for position or attitude tracking
- contact and formation studies where network effects are not the experimental variable

## HIL Mode

Use `hil` for remote-controller or hardware-in-the-loop style timing conditions.

Current implemented behavior:

- state transmit delay injection from `network_fidelity.state_tx_latency_ms`
- command receive delay injection from `network_fidelity.command_rx_latency_ms`
- Gaussian jitter around those delays from `network_fidelity.jitter_std_dev_ms`
- packet dropping from `network_fidelity.packet_loss_percent`
- stale-command handling from `network_fidelity.stale_command_threshold_ms` and `network_fidelity.stale_command_policy`
- first-order actuator lag and rotor thrust or omega rate limiting from `actuator_dynamics`
- additive sensor noise and truth logging from `sensor_fidelity` and `logging_config`

Not modeled:

- sensor bias, quantization, or delayed measurement export beyond the additive-noise model

## Remote Evaluation Tracks

When the simulator runs on one machine and the controller runs on another, the main question is usually not a single vague "HIL fidelity" claim. In practice, two different evaluation tracks are supported.

### 1. Timing-Faithful Remote Evaluation

Use this when the goal is to preserve wall-clock timing semantics between the simulator host and the controller host even if the simulator does not maintain `realtime_factor ~= 1`.

Expected interpretation:

- packet timestamps and age are still evaluated against wall clock
- delayed delivery, jitter, loss, and stale-command handling remain meaningful
- `realtime_factor` may be lower than one, but the closed-loop timing instrumentation should still be internally consistent

Primary metrics:

- state packet age
- command packet age
- sequence gap rate
- stale command rate
- timeout count

### 2. Near-Real-Time Compute-Budget Evaluation

Use this when the goal is to keep `realtime_factor` close to one and judge whether the controller host can sustain the required closed-loop rate.

Expected interpretation:

- the topology is still one simulator host plus one controller host
- the main acceptance condition is that controller compute time and transport delay stay below the available timing budget
- `realtime_factor` should stay close to one for the scenario of interest

Primary metrics:

- realtime factor
- controller compute time
- state packet age under sustained load
- tracking degradation relative to `baseline`

The two tracks can use the same packet schema and logging structure. The difference is the evaluation objective, not the packet format.

## Runtime Metadata

State packets carry:

- `protocol_version`
- `sequence`
- `wall_time_send_ns`
- `fidelity_mode`
- `sim_time`

Command packets carry:

- `protocol_version`
- `sequence`
- `source_state_sequence`
- `wall_time_send_ns`
- `fidelity_mode`

Python runtime tracking currently measures:

- state packet age
- state sequence gap
- command packet age
- command sequence gap
- stale command count
- stale command apply count
- command timeout count
- controller compute time

MATLAB `.mat` logs retain the same packet metadata fields so that MATLAB and Python runs can be compared with the same downstream analysis structure.

## Configuration

Shared fidelity-related fields in `vehicle_params.json`:

- `fidelity_mode`
- `network_fidelity`
- `actuator_dynamics`
- `sensor_fidelity`
- `logging_config`

Minimal HIL-oriented example:

```json
{
  "fidelity_mode": "hil",
  "network_fidelity": {
    "enabled": true,
    "state_tx_latency_ms": 15.0,
    "command_rx_latency_ms": 8.0,
    "packet_loss_percent": 1.0,
    "jitter_std_dev_ms": 2.0,
    "stale_command_threshold_ms": 40.0,
    "stale_command_policy": "zero-thrust"
  }
}
```

## Recommended Metrics

For `baseline`:

- tracking error
- attitude error
- contact force statistics
- control magnitude
- repeatability under the same seed and setup

For `hil`:

- realtime factor
- state packet age
- command packet age
- sequence gap rate
- stale command rate
- timeout count
- tracking degradation relative to `baseline`

## Commands

Reference baseline run:

```powershell
uv run mujoco-wheeled-uav-simulator simulate --fidelity-mode baseline
uv run mujoco-wheeled-uav-simulator hover-controller --fidelity-mode baseline
```

Reference HIL-style run:

```powershell
uv run mujoco-wheeled-uav-simulator simulate --fidelity-mode hil --bind-ip 0.0.0.0 --state-target-ip 192.168.0.42
uv run mujoco-wheeled-uav-simulator hover-controller --fidelity-mode hil --bind-ip 0.0.0.0 --target-ip 192.168.0.10
```

## Scope

The runtime provides publication-oriented instrumentation, network disturbance injection, first-order actuator lag / rate limiting, and additive sensor noise with truth logging. Frame HIL-style claims as communication-timing and runtime-behavior evidence rather than as full hardware equivalence.