"""A neutral flight-log format shared by simulator runs and real flights.

One time base, named channels, metadata; saved as ``.npz`` (format
``wheeled_uav.flightlog/1``). The identification tools (wheeled_uav.sysid)
read only this, so a log from the simulator and a log from a real vehicle
(converted, e.g. from PX4 ULog with wheeled_uav.flightlog_ulog) go through
the same code.

Frames: the world frame is the simulator's (z up; NWU for converted PX4
logs), the body frame is x forward, y left, z up (FLU). Quaternions are
``[w, x, y, z]`` and rotate body vectors into the world.

Channels (all optional except ``position`` and ``quaternion``):

==========================  ======  ===================================================
name                        width   meaning
==========================  ======  ===================================================
position                    3       world position [m]
velocity                    3       world velocity [m/s]
quaternion                  4       attitude, body to world, [w, x, y, z]
angular_velocity_body       3       body rates [rad/s]
rotor_thrust_command        4       commanded thrust per rotor [N] (actuation.rotors order)
rotor_command               4       normalized motor command [0, 1] (when no thrust is known)
rotor_speed                 4       measured rotor speed [rad/s]
battery_voltage             1       [V]
battery_current             1       [A]
specific_force_body         3       accelerometer: specific force in the body frame [m/s^2]
valid                       1       1 where the sample is trustworthy (0: a source dropout)
==========================  ======  ===================================================

Commands are zero-order held between samples (``rotor_thrust_command[k]``
acts from ``time[k]`` to ``time[k+1]``); measurements are interpolated
linearly. That is exact when every sample is a command update. A log that
only samples a faster command stream (a logger slower than the controller)
says so with ``metadata["commands"] = "sampled"``: its commands are then
interpolated linearly too, since holding them would place every change half
a sample late on average.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["CHANNEL_WIDTHS", "FLIGHTLOG_FORMAT", "FlightLog", "quaternion_to_rotation", "rotation_to_quaternion"]

FLIGHTLOG_FORMAT = "wheeled_uav.flightlog/1"
CHANNEL_WIDTHS = {
    "position": 3,
    "velocity": 3,
    "quaternion": 4,
    "angular_velocity_body": 3,
    "rotor_thrust_command": 4,
    "rotor_command": 4,
    "rotor_speed": 4,
    "battery_voltage": 1,
    "battery_current": 1,
    "specific_force_body": 3,
    "valid": 1,
}
REQUIRED_CHANNELS = ("position", "quaternion")
# Held between samples instead of interpolated.
HELD_CHANNELS = ("rotor_thrust_command", "rotor_command", "valid")
COMMAND_CHANNELS = ("rotor_thrust_command", "rotor_command")
COMMAND_SEMANTICS = ("held", "sampled")


def quaternion_to_rotation(quaternion: np.ndarray) -> np.ndarray:
    """Rotation matrices (N, 3, 3) from quaternions (N, 4) [w, x, y, z] (normalized here)."""
    q = np.asarray(quaternion, dtype=float)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], axis=-1),
            np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], axis=-1),
            np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], axis=-1),
        ],
        axis=-2,
    )


def rotation_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    """Quaternions (N, 4) [w, x, y, z] with w >= 0 from rotation matrices (N, 3, 3)."""
    matrices = np.asarray(rotation, dtype=float).reshape(-1, 3, 3)
    quaternions = np.empty((matrices.shape[0], 4))
    for index, m in enumerate(matrices):
        trace = m[0, 0] + m[1, 1] + m[2, 2]
        if trace > 0.0:
            s = 2.0 * np.sqrt(trace + 1.0)
            q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
            q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
        elif m[1, 1] > m[2, 2]:
            s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
            q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
        else:
            s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
            q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
        q = np.asarray(q)
        quaternions[index] = q if q[0] >= 0.0 else -q
    return quaternions / np.linalg.norm(quaternions, axis=1, keepdims=True)


@dataclass
class FlightLog:
    time: np.ndarray
    channels: dict[str, np.ndarray]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.time = np.asarray(self.time, dtype=float).reshape(-1)
        if self.time.size < 2 or not np.all(np.diff(self.time) > 0.0):
            raise ValueError("flight log: time must hold at least two strictly increasing samples")
        checked = {}
        for name, values in self.channels.items():
            if name not in CHANNEL_WIDTHS:
                raise ValueError(f"flight log: unknown channel {name!r}; known: {', '.join(CHANNEL_WIDTHS)}")
            array = np.asarray(values, dtype=float).reshape(self.time.size, -1)
            if array.shape[1] != CHANNEL_WIDTHS[name]:
                raise ValueError(f"flight log: channel {name!r} must have {CHANNEL_WIDTHS[name]} columns, got {array.shape[1]}")
            checked[name] = array
        missing = [name for name in REQUIRED_CHANNELS if name not in checked]
        if missing:
            raise ValueError(f"flight log: missing channel(s) {', '.join(missing)}")
        if self.metadata.get("commands", "held") not in COMMAND_SEMANTICS:
            raise ValueError(f'flight log: metadata "commands" must be one of {", ".join(COMMAND_SEMANTICS)}')
        norms = np.linalg.norm(checked["quaternion"], axis=1, keepdims=True)
        if np.any(np.abs(norms - 1.0) > 1.0e-12):
            checked["quaternion"] = checked["quaternion"] / norms
        self.channels = checked

    # -- access -------------------------------------------------------------
    def __len__(self) -> int:
        return self.time.size

    def has(self, name: str) -> bool:
        return name in self.channels

    def __getitem__(self, name: str) -> np.ndarray:
        return self.channels[name]

    @property
    def rotation(self) -> np.ndarray:
        return quaternion_to_rotation(self.channels["quaternion"])

    @property
    def sample_period(self) -> float:
        return float(np.median(np.diff(self.time)))

    def valid_mask(self) -> np.ndarray:
        if "valid" not in self.channels:
            return np.ones(len(self), dtype=bool)
        return self.channels["valid"][:, 0] > 0.5

    # -- files --------------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        """Write the log and return the file's path (``.npz`` is appended when the name lacks it)."""
        path = Path(path)
        if path.suffix != ".npz":
            path = path.with_name(path.name + ".npz")
        arrays = {f"channel_{name}": values for name, values in self.channels.items()}
        np.savez_compressed(
            path,
            format=np.array(FLIGHTLOG_FORMAT),
            time=self.time,
            metadata=np.array(json.dumps(self.metadata)),
            **arrays,
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> FlightLog:
        with np.load(Path(path), allow_pickle=False) as archive:
            if "format" not in archive or str(archive["format"]) != FLIGHTLOG_FORMAT:
                raise ValueError(f"{path}: not a {FLIGHTLOG_FORMAT} file")
            channels = {key.removeprefix("channel_"): archive[key] for key in archive.files if key.startswith("channel_")}
            return cls(time=archive["time"], channels=channels, metadata=json.loads(str(archive["metadata"])))

    # -- reshaping ------------------------------------------------------------
    def window(self, start: float | None = None, end: float | None = None) -> FlightLog:
        keep = np.ones(len(self), dtype=bool)
        if start is not None:
            keep &= self.time >= start
        if end is not None:
            keep &= self.time <= end
        return FlightLog(self.time[keep], {name: values[keep] for name, values in self.channels.items()}, dict(self.metadata))

    def sample(self, name: str, times: np.ndarray) -> np.ndarray:
        """Channel ``name`` at ``times``: held for commands and validity, linear for measurements, normalized quaternions.

        Commands of a log marked ``"commands": "sampled"`` are interpolated linearly (module docstring).
        """
        times = np.asarray(times, dtype=float)
        values = self.channels[name]
        if name in HELD_CHANNELS and not (name in COMMAND_CHANNELS and self.metadata.get("commands") == "sampled"):
            # A sample at (almost) the instant a command changes gets the new command: time stamps
            # that are sums of steps differ from computed instants by rounding.
            tolerance = 1.0e-6 * self.sample_period
            index = np.clip(np.searchsorted(self.time, times + tolerance, side="right") - 1, 0, len(self) - 1)
            return values[index]
        result = np.column_stack([np.interp(times, self.time, values[:, column]) for column in range(values.shape[1])])
        if name == "quaternion":
            # Keep consecutive quaternions in one hemisphere before interpolating, then renormalize.
            aligned = values.copy()
            signs = np.sign(np.sum(aligned[1:] * aligned[:-1], axis=1))
            signs[signs == 0] = 1.0
            aligned[1:] *= np.cumprod(signs)[:, None]
            result = np.column_stack([np.interp(times, self.time, aligned[:, column]) for column in range(4)])
            result /= np.linalg.norm(result, axis=1, keepdims=True)
        return result

    def resample(self, times: np.ndarray) -> FlightLog:
        times = np.asarray(times, dtype=float)
        return FlightLog(times, {name: self.sample(name, times) for name in self.channels}, dict(self.metadata))

    def uniform(self, rate_hz: float) -> FlightLog:
        """The log on a uniform grid of ``rate_hz`` over its time span."""
        count = int(np.floor((self.time[-1] - self.time[0]) * rate_hz)) + 1
        return self.resample(self.time[0] + np.arange(count) / rate_hz)
