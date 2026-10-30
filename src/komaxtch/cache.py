"""Cache em disco das detecções caras (nuvem), para reconverter sem gastar cota."""

from __future__ import annotations

import json
import os
from pathlib import Path

from komaxtch.geometry import BoundingBox

DEFAULT_CACHE_DIR = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "komaxtch"


class DetectionCache:
    def __init__(self, root: Path = DEFAULT_CACHE_DIR) -> None:
        self._root = root

    def get(self, namespace: str, key: str) -> list[BoundingBox] | None:
        path = self._path(namespace, key)
        if not path.exists():
            return None
        try:
            return [BoundingBox(*box) for box in json.loads(path.read_text())]
        except (ValueError, TypeError):
            return None

    def put(self, namespace: str, key: str, boxes: list[BoundingBox]) -> None:
        path = self._path(namespace, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps([box.as_tuple() for box in boxes]))

    def _path(self, namespace: str, key: str) -> Path:
        safe_namespace = "".join(c if c.isalnum() or c in "-_." else "_" for c in namespace)
        return self._root / safe_namespace / f"{key}.json"
