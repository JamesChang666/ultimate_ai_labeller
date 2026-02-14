from __future__ import annotations

import json
import os
import uuid
from typing import Any


def atomic_write_text(path: str, content: str, encoding: str = "utf-8") -> None:
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    temp_path = os.path.join(directory, f"tmp_{uuid.uuid4().hex}.tmp")
    try:
        with open(temp_path, "w", encoding=encoding, newline="\n") as handle:
            handle.write(content)
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


def atomic_write_json(path: str, payload: Any, encoding: str = "utf-8") -> None:
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False), encoding=encoding)
