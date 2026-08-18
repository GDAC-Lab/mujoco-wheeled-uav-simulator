"""PX4 HITL bridge: run the MuJoCo plant against a real Pixhawk over MAVLink.

The flight controller runs its full stack (EKF2, commander, the on-board
controller) and receives simulated sensors via HIL_SENSOR / HIL_GPS, while
its HIL_ACTUATOR_CONTROLS drive this simulator's rotors. See
``python -m wheeled_uav.px4_hitl --help`` for the staged bring-up options
(--check-link, --arm, --takeoff).

Entry point: ``python -m wheeled_uav.px4_hitl --device COM5`` (requires the ``hil``
extra: ``uv sync --extra hil``).
"""

from .bridge import HilConfig, run_bridge, run_check_link

__all__ = ["HilConfig", "run_bridge", "run_check_link"]
