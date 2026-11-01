"""Wall-clock time with sub-millisecond resolution on every platform.

Packets carry wall-clock stamps, and the HIL network model times its latencies
and packet ages with them. On Windows before Python 3.13, ``time.time_ns()``
advances in steps of the system timer tick (15.6 ms unless some process has
raised the timer resolution), which is too coarse for that. There the clock is
extrapolated from an anchor with the performance counter, and kept within the
tick the system clock reports, so it cannot drift away from wall time.
"""

from __future__ import annotations

import sys
import time

__all__ = ["wall_time_ns"]

# Longest step of the Windows system clock (the default timer tick).
_SYSTEM_TICK_NS = 15_625_000

if sys.platform == "win32" and sys.version_info < (3, 13):
    _anchor: list[int] = []

    def wall_time_ns() -> int:
        """Nanoseconds since the Unix epoch, with performance-counter resolution."""
        coarse = time.time_ns()
        counter = time.perf_counter_ns()
        if not _anchor:
            # Anchor at a step of the coarse clock, where it reads the true time.
            while (stepped := time.time_ns()) == coarse:
                pass
            coarse, counter = stepped, time.perf_counter_ns()
            _anchor[:] = [coarse, counter]
            return coarse
        estimate = _anchor[0] + (counter - _anchor[1])
        if estimate < coarse or estimate > coarse + _SYSTEM_TICK_NS:
            # The true time lies in the tick the system clock reports; re-anchor
            # if the counter has drifted outside it.
            _anchor[:] = [coarse, counter]
            return coarse
        return estimate

else:
    wall_time_ns = time.time_ns
