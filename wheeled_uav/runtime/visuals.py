"""Purely visual runtime helpers: camera framing, wall overlay, video recording.

Nothing in this module writes to the physics state; removing it from a run
changes only what you see, never what the vehicles do.
"""

from __future__ import annotations

import math
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from .scene import SimulationScene

__all__ = [
    "FrameRecorder",
    "WallOverlayAnimator",
    "configure_viewer",
]


def _wall_view_fit(model: mujoco.MjModel, params: dict, aspect: float = 16.0 / 9.0):
    """(lookat, distance) that frames the whole wall-running region, or None.

    The region is whatever the run is about: every reference path end and every obstacle
    (read from environment.wall_overlay), plus a margin. Fitting to that keeps the full
    trajectory in frame no matter how tall the climb or where the obstacle sits, instead of
    a hand-tuned distance that clips a taller scenario. None if there is nothing to frame,
    so the caller can fall back to fixed defaults.
    """
    try:
        environment = params.get("environment") or {}
        overlay = environment.get("wall_overlay") or {}
        points = []
        for reference in overlay.get("references") or []:
            points.append(reference["start_yz"])
            points.append(reference["target_yz"])
        for obstacle in overlay.get("obstacles") or []:
            center = obstacle["center_yz"]
            radius = float(obstacle["radius"])
            points += [
                [center[0] - radius, center[1]], [center[0] + radius, center[1]],
                [center[0], center[1] - radius], [center[0], center[1] + radius],
            ]
        if not points:
            return None
        points = np.asarray(points, dtype=float)
        y_min, z_min = points.min(axis=0)
        y_max, z_max = points.max(axis=0)
        margin = 0.6
        y_min -= margin; y_max += margin
        z_min = max(0.0, z_min - margin); z_max += margin
        center_y = 0.5 * (y_min + y_max)
        center_z = 0.5 * (z_min + z_max)
        fovy = math.radians(model.vis.global_.fovy if model.vis.global_.fovy > 0 else 45.0)
        # Distance so both spans fit: the vertical span against fovy, the horizontal span
        # against the wider horizontal field (fovy scaled by the frame aspect).
        distance_v = (0.5 * (z_max - z_min)) / math.tan(0.5 * fovy)
        distance_h = (0.5 * (y_max - y_min)) / math.tan(math.atan(math.tan(0.5 * fovy) * aspect))
        distance = max(distance_v, distance_h) * 1.12
        wall_x = float(environment["wall_position"][0])
        wall_thickness = float((environment.get("wall_size") or [0.0])[0])
        lookat = (wall_x - wall_thickness - 0.15, center_y, center_z)
        return lookat, distance
    except (KeyError, IndexError, TypeError, ValueError):
        return None


def _apply_wall_camera(cam, model: mujoco.MjModel | None = None, params: dict | None = None) -> None:
    # Single source for the wall-running view, shared by the interactive viewer and the
    # offscreen FrameRecorder so a recorded clip frames the scene exactly like the viewer.
    # When the scene is known, frame the whole region automatically; otherwise fall back to
    # fixed values.
    lookat = (2.0, 0.0, 0.5)
    distance = 8.0
    azimuth = 45.0
    elevation = -20.0
    fit = _wall_view_fit(model, params) if (model is not None and params is not None) else None
    if fit is not None:
        lookat, distance = fit
        # View from IN FRONT of the wall (the -x side, where the vehicle runs), not through
        # it -- the wall is semi-transparent, so a camera behind it would show the vehicle
        # hazed through the wall. azimuth 335 keeps the camera on the -x, +y side for a
        # slight 3/4 angle that shows the vehicle pressed onto the near wall face.
        azimuth = 335.0
        elevation = -8.0   # keep the whole vertical trajectory in frame
    cam.lookat[0], cam.lookat[1], cam.lookat[2] = lookat
    cam.distance = distance
    cam.azimuth = azimuth
    cam.elevation = elevation


def configure_viewer(viewer: mujoco.viewer.Handle, scene: SimulationScene | None = None) -> None:
    if scene is not None:
        _apply_wall_camera(viewer.cam, scene.model, scene.params)
    else:
        _apply_wall_camera(viewer.cam)


class WallOverlayAnimator:
    """Drives the mocap spheres that mark where each reference is right now.

    The static half of the overlay (obstacle outlines, reference paths) is baked
    into the model by model.builder._build_wall_overlay_block. This moves the one
    body per reference that has to follow the clock: each entry advances along
    start_yz -> target_yz at a constant speed and then holds at the target, which
    matches a constant-speed straight-line reference. Give one entry per UAV.

    The clock is "time since the controller took over", not data.time. data.time keeps
    advancing while hold_until_first_command freezes the vehicle at its spawn pose, so
    driving the markers off data.time would walk the target away from a stationary
    vehicle -- and the reference a controller tracks starts at its own first state
    anyway. update(active=False) therefore parks every marker at its start_yz and keeps
    the origin unset.

    Purely visual: the marker bodies are mocap (no joints, contype/conaffinity=0),
    so they add no dynamics and never touch the vehicles.
    """

    def __init__(self, scene: SimulationScene) -> None:
        self._entries: list[tuple[int, float, np.ndarray, np.ndarray, float, float]] = []
        self._time_origin: float | None = None
        overlay = (scene.params.get("environment") or {}).get("wall_overlay")
        if not isinstance(overlay, dict) or not bool(overlay.get("enabled", False)):
            return
        references = overlay.get("references") or []
        if not references:
            return
        environment = scene.params["environment"]
        marker_x = (
            float(environment["wall_position"][0])
            - float(environment["wall_size"][0])
            - float(overlay.get("surface_offset", 0.02))
        )
        for index, reference in enumerate(references):
            body_id = mujoco.mj_name2id(
                scene.model, mujoco.mjtObj.mjOBJ_BODY, f"overlay_reference_marker_{index}"
            )
            if body_id < 0:
                continue
            mocap_id = int(scene.model.body_mocapid[body_id])
            if mocap_id < 0:
                continue
            start = np.asarray([float(value) for value in reference["start_yz"]], dtype=float)
            target = np.asarray([float(value) for value in reference["target_yz"]], dtype=float)
            speed = float(reference.get("speed", 0.0))
            path_length = float(np.linalg.norm(target - start))
            self._entries.append((mocap_id, marker_x, start, target, speed, path_length))

    @property
    def enabled(self) -> bool:
        return bool(self._entries)

    def update(self, data: mujoco.MjData, active: bool = True) -> None:
        # active: the controller has taken over (first command applied). Until then the
        # references have not started, so hold every marker at its start point.
        if not active:
            self._time_origin = None
            reference_time = 0.0
        else:
            if self._time_origin is None:
                self._time_origin = float(data.time)
            reference_time = float(data.time) - self._time_origin
        for mocap_id, marker_x, start, target, speed, path_length in self._entries:
            if path_length <= 1.0e-9 or speed <= 0.0:
                point = target if active else start
            else:
                travelled = min(speed * max(0.0, reference_time), path_length)
                point = start + (target - start) * (travelled / path_length)
            data.mocap_pos[mocap_id] = (marker_x, float(point[0]), float(point[1]))


class FrameRecorder:
    """Writes an offscreen render of the run to a video file (MP4/GIF by extension).

    Disabled unless a path is given (scene.request.record_path). Renders with the same
    camera as the interactive viewer, so the clip matches what you would have watched.

    Resolution and frame rate come from the environment.recording config block:
        "recording": {"width": 1280, "height": 720, "fps": 30}
    The clip plays at real time: one physics second maps to one video second, by capturing
    a frame every round(1/(fps*timestep)) steps.

    Optional keys:
        "views": a list of camera angles rendered side by side into one frame,
            each {"azimuth": deg, "elevation": deg, "distance_scale": 1.0}
            applied on top of the shared auto-framed wall view. Each pane is
            width x height, so the written frame is (n_views * width) x height.
            Omitted -> the single default view, exactly as before.
        "show_contacts": true draws contact points and contact-force arrows in
            the recording (arrow length scales with the force via the template's
            visual map.force). Recording-only: the interactive viewer keeps its
            own toggles.

    Observational only: it reads data and never writes it, so it cannot perturb the physics
    or the controller. Like the overlay, capture is gated on 'the controller has taken over'
    so the clip covers the actual run, not the frozen spawn-wait. imageio (and, for MP4, the
    bundled ffmpeg) is imported lazily, so a non-recording run needs neither installed.
    """

    def __init__(self, scene: SimulationScene, output_path: str | Path | None) -> None:
        self._writer = None
        self._renderer = None
        self._cameras: list[mujoco.MjvCamera] = []
        self._scene_option = None
        self._capture_every = 1
        self._step = 0
        self._frames_written = 0
        self._output_path = ""
        if not output_path:
            return
        config = (scene.params.get("environment") or {}).get("recording") or {}
        width = int(config.get("width", 1280))
        height = int(config.get("height", 720))
        fps = float(config.get("fps", 30.0))
        # The offscreen framebuffer is fixed at model-compile time (visual/global in the
        # template). Asking mujoco.Renderer for more than that is a hard error, so clamp.
        max_width = int(scene.model.vis.global_.offwidth)
        max_height = int(scene.model.vis.global_.offheight)
        if width > max_width or height > max_height:
            print(
                f"recording: requested {width}x{height} exceeds the offscreen framebuffer "
                f"{max_width}x{max_height}; clamping (raise visual/global offwidth/offheight "
                f"in the template for more)"
            )
            width = min(width, max_width)
            height = min(height, max_height)
        timestep = float(scene.model.opt.timestep)
        self._capture_every = max(1, round(1.0 / (fps * timestep)))
        try:
            import imageio.v2 as imageio  # lazy: only needed when recording
        except ImportError as exc:  # pragma: no cover - depends on install
            raise RuntimeError(
                "recording was requested but imageio is not installed; run "
                "`uv sync --project <simulator>` to pull the recording extras"
            ) from exc
        self._renderer = mujoco.Renderer(scene.model, height=height, width=width)
        views = config.get("views")
        view_specs = views if isinstance(views, list) and views else [None]
        for view in view_specs:
            camera = mujoco.MjvCamera()
            camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            _apply_wall_camera(camera, scene.model, scene.params)
            if isinstance(view, dict):
                if "azimuth" in view:
                    camera.azimuth = float(view["azimuth"])
                if "elevation" in view:
                    camera.elevation = float(view["elevation"])
                camera.distance *= float(view.get("distance_scale", 1.0))
            self._cameras.append(camera)
        # Defaults of MjvOption match the default rendering; show_contacts only
        # flips the two contact-visualization flags on top.
        self._scene_option = mujoco.MjvOption()
        if bool(config.get("show_contacts", False)):
            self._scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
            self._scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
        # macro_block_size=None: keep the exact width/height instead of rounding to 16.
        self._writer = imageio.get_writer(str(output_path), fps=fps, macro_block_size=None)
        self._output_path = str(output_path)
        print(
            f"recording: {len(self._cameras)} view(s) x {width}x{height} @ {fps:g} fps "
            f"(1 frame / {self._capture_every} steps) -> {self._output_path}"
        )

    @property
    def enabled(self) -> bool:
        return self._writer is not None

    def capture(self, data: mujoco.MjData, active: bool = True) -> None:
        # active: the controller has taken over. Before that the vehicle is frozen at spawn,
        # so recording it would only add a static lead-in.
        if self._writer is None or not active:
            return
        self._step += 1
        if self._step % self._capture_every != 0:
            return
        panes = []
        for camera in self._cameras:
            self._renderer.update_scene(data, camera=camera, scene_option=self._scene_option)
            panes.append(self._renderer.render())
        self._writer.append_data(panes[0] if len(panes) == 1 else np.hstack(panes))
        self._frames_written += 1

    def close(self) -> None:
        # Idempotent, and safe to call on early exit: whatever was captured is flushed to a
        # playable file rather than lost.
        if self._writer is not None:
            self._writer.close()
            print(f"recording: wrote {self._frames_written} frames -> {self._output_path}")
            self._writer = None
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
