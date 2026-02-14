from __future__ import annotations

from typing import List

from .types import Rect


def calculate_iou(box1: Rect, box2: Rect) -> float:
    x1_1, y1_1, x2_1, y2_1 = box1[:4]
    x1_2, y1_2, x2_2, y2_2 = box2[:4]
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    if x2_i < x1_i or y2_i < y1_i:
        return 0.0
    inter = (x2_i - x1_i) * (y2_i - y1_i)
    union = (x2_1 - x1_1) * (y2_1 - y1_1) + (x2_2 - x1_2) * (y2_2 - y1_2) - inter
    return inter / union if union > 0 else 0.0


def fuse_boxes(boxes: List[Rect], iou_thresh: float, dist_thresh: int) -> List[Rect]:
    if len(boxes) <= 1:
        return boxes
    keep_fusing = True
    current = [box[:] for box in boxes]
    while keep_fusing:
        keep_fusing = False
        merged: List[Rect] = []
        used = [False] * len(current)
        for i, box in enumerate(current):
            if used[i]:
                continue
            curr = box[:]
            used[i] = True
            for j in range(i + 1, len(current)):
                if used[j]:
                    continue
                other = current[j]
                should_merge = calculate_iou(curr, other) > iou_thresh
                if not should_merge:
                    h_dist = max(0, max(curr[0], other[0]) - min(curr[2], other[2]))
                    v_overlap = min(curr[3], other[3]) - max(curr[1], other[1])
                    if v_overlap > 0 and h_dist <= dist_thresh:
                        should_merge = True
                if should_merge:
                    curr = [
                        min(curr[0], other[0]),
                        min(curr[1], other[1]),
                        max(curr[2], other[2]),
                        max(curr[3], other[3]),
                        curr[4],
                    ]
                    used[j] = True
                    keep_fusing = True
            merged.append(curr)
        current = merged
    return current
