# Propulsion calibration loading

Load bench thrust-test results into the simulator so gain tuning runs on
measured propulsion parameters. Any test rig works as long as its fitting
step writes a JSON file in the schema below.
Japanese version with full details: [CALIBRATION.ja.md](CALIBRATION.ja.md).

## Usage

Have your thrust-test fitting step write a `thrust_<date>_<config>_sim.json`
next to the calibration data, then reference it from `vehicle_params.json`:

```json
"actuation": {
    "calibration_file": "configs/thrust_20260101_myconfig_sim.json"
}
```

Both loaders overlay the measured values at load time and log one line
(`calibration: <file> -> ...`): `load_vehicle_params()` on the Python side and
`uavsim.Params.load` on the MATLAB side. A MATLAB controller therefore converts
thrust to rotor speed with the kf the simulator uses, and in `command_mode: omega`
the two conversions cancel. Up to v1.1.1 the MATLAB loader ignored
`calibration_file`, so the vehicle got only kf_calibrated / kf_file of the
commanded thrust (issue #1).
Relative paths resolve against the directory containing vehicle_params.json.
Provenance is kept in `params["calibration_applied"]`
(MATLAB: `vehicle_params.raw.calibration_applied`).

## Schema `uav-propulsion-calibration/1`

| `sim_params` key | applied to | quantity |
|---|---|---|
| `thrust_coefficient` | `actuation.thrust_coefficient` | kf [N·s²/rad²] |
| `yaw_moment_ratio` | `actuation.yaw_moment_ratio` + every `rotors[].yaw_moment_ratio` | km/kf [m] (measured a1) |
| `motor_tau_ms` | `actuator_dynamics.motor_tau_ms` | first-order motor lag [ms] |

`null` entries are skipped, so a calibration overrides only what it measured.

**Note:** kf is currently a provisional conversion `kf = c2 / Kv_rad²` (no RPM
measurement yet); after applying, verify hover throttle/rotor speed against flight
logs. See CALIBRATION.ja.md for the verification checklist and planned extensions
(Tm up/down, interference model).
