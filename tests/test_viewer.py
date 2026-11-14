from __future__ import annotations

import threading
import time
import unittest
from unittest.mock import patch

from wheeled_uav.runtime import visuals


class _FakeViewer:
    """Stands in for mujoco.viewer.launch_passive: draws on a daemon thread that keeps
    tearing its window down for a moment after close() returns, as the real one does."""

    def __init__(self) -> None:
        self.closed = threading.Event()
        self._stop = threading.Event()

    def launch_passive(self, model, data):
        def draw() -> None:
            self._stop.wait()
            time.sleep(0.2)
            self.closed.set()

        threading.Thread(target=draw, daemon=True).start()
        return self

    def close(self) -> None:
        self._stop.set()


class OpenPassiveViewerTest(unittest.TestCase):
    def test_leaving_waits_until_the_viewer_has_closed_its_window(self) -> None:
        fake = _FakeViewer()
        with patch.object(visuals.mujoco.viewer, "launch_passive", fake.launch_passive):
            with visuals.open_passive_viewer(None, None) as handle:
                self.assertIs(handle, fake)
                self.assertFalse(fake.closed.is_set())
            self.assertTrue(fake.closed.is_set())

    def test_an_error_in_the_loop_still_closes_and_waits(self) -> None:
        fake = _FakeViewer()
        with patch.object(visuals.mujoco.viewer, "launch_passive", fake.launch_passive):
            with self.assertRaises(RuntimeError):
                with visuals.open_passive_viewer(None, None):
                    raise RuntimeError("stop")
            self.assertTrue(fake.closed.is_set())


if __name__ == "__main__":
    unittest.main()
