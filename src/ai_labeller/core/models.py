from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from .types import Rect


@dataclass
class SessionState:
    project_root: str = ""
    split: str = "train"
    image_name: str = ""
    detection_model_mode: str = "Official YOLO26n.pt (Bundled)"
    detection_model_path: str = "yolo26n.pt"


@dataclass
class AppState:
    project_root: str = ""
    current_split: str = "train"
    image_files: List[str] = field(default_factory=list)
    current_idx: int = 0
    rects: List[Rect] = field(default_factory=list)
    class_names: List[str] = field(default_factory=lambda: ["text", "figure", "table"])
