"""Reference controllers that talk to the simulator over UDP.

These are runnable samples, not project-specific research controllers — keep
project controllers in the repository that consumes this simulator. The
modules avoid importing MuJoCo so they can run on controller-only hosts
where MuJoCo may not be installed.
"""

from .hover import run_hover_controller

__all__ = ["run_hover_controller"]
