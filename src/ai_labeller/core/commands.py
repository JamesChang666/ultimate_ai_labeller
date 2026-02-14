from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import List, Optional

from .types import Rect


@dataclass
class RevertToSnapshotCommand:
    target: List[Rect]
    before: List[Rect]
    after: Optional[List[Rect]] = None

    @classmethod
    def from_target(cls, target: List[Rect]) -> "RevertToSnapshotCommand":
        return cls(target=target, before=copy.deepcopy(target))

    def undo(self) -> None:
        if self.after is None:
            self.after = copy.deepcopy(self.target)
        self.target[:] = copy.deepcopy(self.before)

    def redo(self) -> None:
        if self.after is None:
            return
        self.target[:] = copy.deepcopy(self.after)


class HistoryManager:
    def __init__(self) -> None:
        self._undo_stack: List[RevertToSnapshotCommand] = []
        self._redo_stack: List[RevertToSnapshotCommand] = []

    def push_snapshot(self, target: List[Rect]) -> None:
        self._undo_stack.append(RevertToSnapshotCommand.from_target(target))
        self._redo_stack.clear()

    def clear(self) -> None:
        self._undo_stack.clear()
        self._redo_stack.clear()

    def undo(self) -> bool:
        if not self._undo_stack:
            return False
        cmd = self._undo_stack.pop()
        cmd.undo()
        self._redo_stack.append(cmd)
        return True

    def redo(self) -> bool:
        if not self._redo_stack:
            return False
        cmd = self._redo_stack.pop()
        cmd.redo()
        self._undo_stack.append(cmd)
        return True
