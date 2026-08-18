"""Public package API for the MuJoCo wheeled UAV simulator.

The stable package-level entry point is ``main``.
Other submodules are primarily internal implementation modules used by the
repository itself.
"""

from __future__ import annotations

__all__ = ["main"]


def main(argv: list[str] | None = None) -> int:
    from .cli import main as cli_main

    return cli_main(argv)
