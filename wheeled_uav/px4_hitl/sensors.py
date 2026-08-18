"""MuJoCo state -> PX4 HIL sensor values.

Frame conventions:

- Simulator world: z-up right-handed, treated as **NWU** (x=North, y=West, z=Up).
- Simulator body: x-forward, z-up.
- PX4 side: **NED** world / **FRD** body.

Both conversions are the same axis flip ``D = diag(1, -1, -1)`` — the same
mapping used by a motion-capture external-vision bridge whose signs were
verified on hardware. Rotations map as ``R_ned = D @ R_nwu @ D``; quaternions
as ``(w, x, y, z) -> (w, x, -y, -z)`` (roll invariant, pitch/yaw negated).

The IMU is derived analytically from the simulator's existing frame sensors
(no model changes): the accelerometer is the specific force
``R^T (dv/dt - g)`` with the linear acceleration taken as a one-step finite
difference of the sensed world velocity. At the physics timestep (~2 ms) this
matches a MuJoCo accelerometer site closely enough for HIL purposes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

AXIS_FLIP = np.diag([1.0, -1.0, -1.0])
GRAVITY_W = np.array([0.0, 0.0, -9.80665])
SEA_LEVEL_HPA = 1013.25
EARTH_RADIUS_M = 6378137.0


def rotation_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    """Rotation matrix -> unit quaternion (w, x, y, z), w >= 0."""
    m = rotation
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        q = np.array([0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s])
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        q = np.array([(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s])
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        q = np.array([(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s])
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        q = np.array([(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s])
    if q[0] < 0.0:
        q = -q
    return q / np.linalg.norm(q)


@dataclass(frozen=True)
class HilSensorSample:
    """One PX4-facing sensor snapshot (all PX4/NED/FRD conventions)."""

    time_usec: int
    accel_frd: np.ndarray       # specific force [m/s^2]; at rest ~ (0, 0, -9.81)
    gyro_frd: np.ndarray        # [rad/s]
    mag_frd: np.ndarray         # [gauss]
    abs_pressure_hpa: float
    pressure_alt_m: float
    pos_ned: np.ndarray         # [m]
    vel_ned: np.ndarray         # [m/s]
    quat_ned: np.ndarray        # (w, x, y, z), body FRD -> NED


class HilSensorExtractor:
    """Reads the named frame sensors of one UAV and converts to PX4 frames."""

    def __init__(
        self,
        model: mujoco.MjModel,
        sensor_names,
        mag_ned_gauss=(0.21523, 0.00771, 0.42741),
        origin_alt_m: float = 488.0,
    ):
        self._model = model
        self._timestep = float(model.opt.timestep)
        self._mag_ned = np.asarray(mag_ned_gauss, dtype=float)
        self._origin_alt_m = float(origin_alt_m)
        self._adr = {
            key: self._sensor_adr(getattr(sensor_names, key))
            for key in ("position", "linear_velocity", "angular_velocity", "x_axis", "y_axis", "z_axis")
        }
        self._prev_vel_w: np.ndarray | None = None

    def _sensor_adr(self, name: str) -> int:
        sensor_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        if sensor_id < 0:
            raise KeyError(f"sensor not found in model: {name}")
        return int(self._model.sensor_adr[sensor_id])

    def _read(self, data: mujoco.MjData, key: str) -> np.ndarray:
        adr = self._adr[key]
        return np.array(data.sensordata[adr:adr + 3], dtype=float)

    def reset(self) -> None:
        self._prev_vel_w = None

    def sample(self, data: mujoco.MjData) -> HilSensorSample:
        pos_w = self._read(data, "position")
        vel_w = self._read(data, "linear_velocity")
        omega_w = self._read(data, "angular_velocity")
        # Body axes in world coordinates = columns of R (body -> world)
        r_wb = np.column_stack([self._read(data, "x_axis"), self._read(data, "y_axis"), self._read(data, "z_axis")])

        if self._prev_vel_w is None:
            accel_w = np.zeros(3)
        else:
            accel_w = (vel_w - self._prev_vel_w) / self._timestep
        self._prev_vel_w = vel_w

        specific_force_body = r_wb.T @ (accel_w - GRAVITY_W)
        accel_frd = AXIS_FLIP @ specific_force_body
        gyro_frd = AXIS_FLIP @ (r_wb.T @ omega_w)

        r_ned = AXIS_FLIP @ r_wb @ AXIS_FLIP
        quat_ned = rotation_to_quaternion(r_ned)
        mag_frd = r_ned.T @ self._mag_ned

        pos_ned = AXIS_FLIP @ pos_w
        vel_ned = AXIS_FLIP @ vel_w

        alt_amsl = self._origin_alt_m + pos_w[2]
        abs_pressure = SEA_LEVEL_HPA * (1.0 - 2.25577e-5 * alt_amsl) ** 5.255877

        return HilSensorSample(
            time_usec=int(data.time * 1e6),
            accel_frd=accel_frd,
            gyro_frd=gyro_frd,
            mag_frd=mag_frd,
            abs_pressure_hpa=float(abs_pressure),
            pressure_alt_m=float(alt_amsl),
            pos_ned=pos_ned,
            vel_ned=vel_ned,
            quat_ned=quat_ned,
        )


def gps_from_ned(
    pos_ned: np.ndarray,
    vel_ned: np.ndarray,
    origin_lat_deg: float,
    origin_lon_deg: float,
    origin_alt_m: float,
) -> dict:
    """NED position/velocity -> HIL_GPS integer fields (flat-earth around origin)."""
    lat_deg = origin_lat_deg + math.degrees(pos_ned[0] / EARTH_RADIUS_M)
    lon_deg = origin_lon_deg + math.degrees(pos_ned[1] / (EARTH_RADIUS_M * math.cos(math.radians(origin_lat_deg))))
    alt_m = origin_alt_m - pos_ned[2]
    speed_mps = float(np.linalg.norm(vel_ned[:2]))
    cog_deg = math.degrees(math.atan2(vel_ned[1], vel_ned[0])) % 360.0 if speed_mps > 0.05 else 0.0
    return {
        "lat": int(round(lat_deg * 1e7)),
        "lon": int(round(lon_deg * 1e7)),
        "alt_mm": int(round(alt_m * 1000.0)),
        "vn_cm": int(round(vel_ned[0] * 100.0)),
        "ve_cm": int(round(vel_ned[1] * 100.0)),
        "vd_cm": int(round(vel_ned[2] * 100.0)),
        "vel_cm": int(round(speed_mps * 100.0)),
        "cog_cdeg": int(round(cog_deg * 100.0)),
    }
