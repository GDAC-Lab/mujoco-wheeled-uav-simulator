# State packet formats

The simulator publishes one state packet every `simulation.state_publish_every_n_steps` physics steps, as a single UDP datagram. `simulation.state_format` (or `simulate --state-format`) selects its encoding:

- `json` (default): one JSON object. A single-UAV packet carries the vehicle's fields at the top level; a multi-UAV packet lists them under `uavs`. The `timing` block is described in [TIMING.md](TIMING.md), the `extensions` object of scenes with extensions in [EXTENSIONS.md](EXTENSIONS.md).
- `binary`: the same state in the fixed little-endian layout below. It is about a quarter of the size of the JSON packet and quicker for the simulator to build, and a team of up to about 170 UAVs fits one datagram (a JSON packet grows by about 1.5 KiB per UAV and passes the 65,507-byte limit of a UDP datagram at about 40 UAVs). Decoding takes about as long as `jsondecode` in MATLAB, whose JSON decoder is fast.

Both decoders turn a binary packet into exactly what the JSON packet of the same state decodes to, so a controller can accept either:

- Python: `wheeled_uav.state_packets.decode_state_packet(datagram)` returns the `json.loads` dict.
- MATLAB: `uavsim.Protocol.decode_state(datagram)` returns the `jsondecode` struct; `uavsim.Protocol.read_latest_state` uses it.

Binary packets carry the contact summaries but not the per-contact details (`contacts` decodes to an empty list), and the simulator refuses to start with the binary format when the actuator stats (`logging_config.log_actuator_stats`), the sensor truth (`logging_config.log_sensor_truth`) or the aerodynamics log are switched on, and when the params have extensions.

## Binary layout (layout version 1)

All values are little-endian; offsets are in bytes from the start of the datagram.

| Offset | Type | Content |
| --- | --- | --- |
| 0 | 4 bytes | magic `WUAV` |
| 4 | uint16 | layout version (1) |
| 6 | uint16 | `protocol_version` |
| 8 | uint16 | number of UAVs `n` |
| 10 | uint16 | record length `L`: float64 values per UAV (46 in layout 1) |
| 12 | uint8 | `fidelity_mode`: 0 `baseline`, 1 `hil`, 255 none |
| 13 | uint8 | `timing.pacing_mode`: 0 `realtime`, 1 `accelerated`, 2 `lockstep` |
| 14 | uint16 | flags: bit `k` (0-9) set when timing value `k` is present, bit 10 when `pacing_mode` is, bit 15 when the packet has a `timing` block at all |
| 16 | int32 | `instance_id` |
| 20 | int64 | `sequence` |
| 28 | int64 | `wall_time_send_ns` |
| 36 | float64 | `sim_time` [s] |
| 44 | float64 | `realtime_factor` |
| 52 | 10 x float64 | timing values 0-9: `physics_timestep_seconds`, `control_period_seconds`, `publish_every_n_steps`, `session_wall_elapsed_seconds`, `session_sim_elapsed_seconds`, `sim_wall_skew_seconds`, `realtime_factor`, `step_wall_seconds`, `publish_wall_seconds`, `command_wait_wall_seconds` (NaN when absent) |
| 132 | uint16 | byte length `m` of the names |
| 134 | `m` bytes | the UAV names, UTF-8, separated by newlines |
| 134 + `m` | 0-7 bytes | zero padding to a multiple of 8 |
| after the padding | `n` x `L` x float64 | one record per UAV, in the order of the names |

A record holds, by index:

| Index | Values |
| --- | --- |
| 0-2 | `position` [m] |
| 3-5 | `velocity` [m/s] |
| 6-8 | `angular_velocity_world` [rad/s] |
| 9-11 | `angular_velocity_body` [rad/s] |
| 12-20 | `rotation_matrix`, the nine values of the JSON list (row-major) |
| 21-25 | `contact_summary` of the whole vehicle: `count`, `total_force_magnitude`, `max_force_magnitude`, `total_normal_force`, `max_normal_force` |
| 26-30 | the same five values for the `left_wheel` group |
| 31-35 | `right_wheel` |
| 36-40 | `surface` |
| 41-45 | `wall` |

The remaining JSON fields follow from these: every UAV's `time` is `sim_time`, `z`, `vz` and `yaw_rate` are the third components of `position`, `velocity` and `angular_velocity_world`, and `realtime_factor` and `instance_id` are the header's. A decoder of its own should check the magic and the layout version, and step through the records with the record length of the header: a later layout may append values to each record.
