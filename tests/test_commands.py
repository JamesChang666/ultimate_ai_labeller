import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import unittest

from ai_labeller.core.commands import HistoryManager


class HistoryManagerTests(unittest.TestCase):
    def test_undo_redo_snapshot(self):
        rects = [[0.0, 0.0, 5.0, 5.0, 0]]
        history = HistoryManager()

        history.push_snapshot(rects)
        rects.append([1.0, 1.0, 2.0, 2.0, 1])

        self.assertTrue(history.undo())
        self.assertEqual(len(rects), 1)

        self.assertTrue(history.redo())
        self.assertEqual(len(rects), 2)


if __name__ == "__main__":
    unittest.main()
