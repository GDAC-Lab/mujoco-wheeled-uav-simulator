"""Simulation runtime: scene loading, stepping loops, UDP streams, and effects.

The runtime consumes the artifacts produced by :mod:`wheeled_uav.model` and runs
the MuJoCo simulation, exchanging state/command packets with controllers over
the wire format defined in :mod:`wheeled_uav.protocol`.
"""

from .loop import run_headless_loop, run_simulation, run_viewer_loop
from .scene import (
    SimulationRequest,
    SimulationScene,
    load_simulation_scene,
    render_simulation_model,
    validate_model,
)
from .state_publisher import StatePayloadPublisher, build_state_payload

__all__ = [
    "SimulationRequest",
    "SimulationScene",
    "StatePayloadPublisher",
    "build_state_payload",
    "load_simulation_scene",
    "render_simulation_model",
    "run_headless_loop",
    "run_simulation",
    "run_viewer_loop",
    "validate_model",
]
