import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import unittest

from ai_labeller.core.geometry import calculate_iou, fuse_boxes


class GeometryTests(unittest.TestCase):
    def test_iou_identical_boxes(self):
        box = [0.0, 0.0, 10.0, 10.0, 0]
        self.assertAlmostEqual(calculate_iou(box, box), 1.0)

    def test_fuse_merges_overlapping_boxes(self):
        boxes = [
            [0.0, 0.0, 10.0, 10.0, 0],
            [1.0, 1.0, 11.0, 11.0, 0],
        ]
        fused = fuse_boxes(boxes, iou_thresh=0.1, dist_thresh=0)
        self.assertEqual(len(fused), 1)


if __name__ == "__main__":
    unittest.main()
