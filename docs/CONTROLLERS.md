# Sample controllers and run scenarios

This document maps **runnable components**, **timing behavior**, and **intended use cases** in the simulator submodule. Design rules live in [TIMING.md](TIMING.md); here we focus on *which program to run in which combination*.

## Terminology (do not conflate)

| Term | Meaning |
| --- | --- |
| **Logical sync** | Control and missions always advance on `state.time` (simulator seconds). Skip duplicate UDP; hold-last commands. **Always on for every controller.** |
| **`realtime` pacing** | Simulator uses `StepPacer` to track wall clock (RTF ≈ 1). For HIL and interactive runs. |
| **`accelerated` pacing** | Simulator does not wait on wall clock. For CI and batch sweeps. |
| **`lockstep` pacing** | Simulator blocks until each control command arrives before advancing one control period. Deterministic co-simulation and exact replays. |
| **`baseline` / `hil` fidelity** | Whether network delay / noise injection is enabled. **Orthogonal to pacing.** |

Controllers must **not** pause to align wall clock with `sim_wall_skew` or simulator time (that causes dead time).

## Provided components

| Component | Entry | Role | Controller-side timing |
| --- | --- | --- | --- |
| **Simulator** | `mujoco-wheeled-uav-simulator simulate` | MuJoCo physics, `mj_step`, UDP state publish | `simulation.pacing_mode`: `realtime`, `accelerated`, or `lockstep` |
| **Python sample** | `mujoco-wheeled-uav-simulator hover-controller` | Single-UAV hover reference (world PD + attitude) | One control eval per new UDP sample; `state.time` only; no sync waits |
| **MATLAB hover sample** | `hovering_controller` | Single-UAV hover, numerically identical to the Python reference | Same logical-sync rules |
| **MATLAB wall demo** | `wall_demo_controller` | Minimal wall riding (unmodified hover control with a target behind the wall; deliberately trivial baseline) | One control eval per new UDP sample |
| **MATLAB formation sample** | `multi_uav_formation_controller` | Centroid + slot formation over batched multi-UAV packets | One eval per new batched sample; one command datagram per sample |
| **MATLAB shared library** | `matlab/+uavsim/` (`uavsim.Protocol`, `uavsim.Session`, ...) | UDP I/O, timing helpers, launch utilities | Same logical-sync APIs (`uavsim.Protocol.udp_state_is_new`, etc.) |
| **External project MATLAB** | controllers in repos that consume this submodule | Project-specific laws (paper stack) | Same rules, wrapped in the project's own classes (outside this repo) |

`hover-controller` accepts **single-UAV packets only**. Multi-UAV packets with a `uavs` array are rejected.

## Recommended scenarios

| Scenario | Simulator `pacing_mode` | `fidelity_mode` | Controller | Typical launch |
| --- | --- | --- | --- | --- |
| **A. Interactive dev (local)** | `realtime` | `baseline` | Project MATLAB or `hover-controller` | `simulate` with viewer + controller in second terminal |
| **B. Remote HIL (two machines)** | `realtime` | `hil` (when injecting delay) | Remote implementation or `hover-controller` | PC: `simulate --bind-ip 0.0.0.0`; remote host: `hover-controller --bind-ip 0.0.0.0` |
| **C. Headless real-time check** | `realtime` | `baseline` | Any | `simulate --headless --duration-seconds 30` + controller |
| **D. CI / physics regression** | `accelerated` | `baseline` | None (open loop) or fast loop | `simulate --headless --pacing-mode accelerated` |
| **E. Batch / gain sweeps** | `accelerated` | `baseline` | `hover-controller` with `--duration-seconds` | Sim + Python controller together |
| **F. Timing unit tests** | Both | `baseline` | None | `pytest tests/test_simulation_timing.py` |
| **G. Minimal wall-riding demo** | `realtime` | `baseline` | `wall_demo_controller` | `simulate --preset wall_demo` + `wall_demo_controller` |
| **H. Formation** | `realtime` | `baseline` | `multi_uav_formation_controller` | `simulate --num-uavs 3` + `multi_uav_formation_controller('num_uavs', 3)` |
| **I. Deterministic replay / co-simulation** | `lockstep` | `baseline` | Any single/multi-UAV controller | `simulate --headless --pacing-mode lockstep` + controller |

### Command examples

```powershell
# A. Interactive dev
uv run wuav simulate
```
```matlab
hovering_controller
```

```powershell
# B. Remote HIL (PC side; the second machine runs the controller)
uv run wuav simulate --bind-ip 0.0.0.0 --state-target-ip 192.168.0.10
uv run wuav hover-controller --bind-ip 0.0.0.0 --target-ip 192.168.0.20   # on the remote host
```

```powershell
# D/E. Accelerated batch
uv run wuav simulate --headless --pacing-mode accelerated --duration-seconds 30
uv run wuav hover-controller --duration-seconds 25
```

```powershell
# G. Minimal wall-riding demo
uv run wuav simulate --preset wall_demo
```
```matlab
wall_demo_controller
```

```powershell
# H. Formation
uv run wuav simulate --num-uavs 3
```
```matlab
multi_uav_formation_controller('num_uavs', 3)
```

## Controller implementation checklist

When adding an external controller (MATLAB / Python / C++):

1. Control / integrator dt = **`Δ(state.time)`** between consecutive unique UDP samples
2. Run control **once** per unique `(sequence, time)` (`StateSampleTracker` / `udp_state_is_new`)
3. **Do not pause on `sim_wall_skew`** (diagnostics only)
4. **Send commands immediately** after compute (no wall/sim alignment sleep)
5. Mission segment boundaries use **`state.time`**, not controller wall clock

## Consuming repositories (external)

Repositories that embed this simulator as a Git submodule run their own controllers on top of the same UDP contract. Their timing rules match `hover-controller`; only the simulator's `pacing_mode` toggles wall-clock pacing. Keep project-specific controllers and configs in the consuming repository and pass a project-owned parameter file via `--params-file` (relative mesh paths resolve next to that file).

Japanese summary: [CONTROLLERS.ja.md](CONTROLLERS.ja.md)
