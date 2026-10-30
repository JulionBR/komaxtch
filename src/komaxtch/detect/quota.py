"""Registro em disco de quais modelos estão com a cota diária esgotada e até quando.

Evita gastar chamadas (e tempo de espera) em modelos bloqueados nas execuções seguintes."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable
from pathlib import Path

from komaxtch.cache import DEFAULT_CACHE_DIR

DEFAULT_LEDGER = DEFAULT_CACHE_DIR / "quota.json"
DEFAULT_BLOCK_SECONDS = 24 * 60 * 60  # sem prazo informado pela API, espera um dia


class QuotaLedger:
    def __init__(self, path: Path = DEFAULT_LEDGER, clock: Callable[[], float] = time.time) -> None:
        self._path = path
        self._clock = clock

    def available(self, model: str) -> bool:
        return self._clock() >= self._blocked_until().get(model, 0)

    def mark_exhausted(self, model: str, retry_after: float | None) -> None:
        blocked = self._blocked_until()
        blocked[model] = self._clock() + (retry_after or DEFAULT_BLOCK_SECONDS)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(blocked))

    def next_available(self, models: Iterable[str]) -> float | None:
        """Momento (epoch) em que o primeiro modelo bloqueado volta a ter cota."""
        blocked = self._blocked_until()
        times = [blocked[m] for m in models if m in blocked and blocked[m] > self._clock()]
        return min(times) if times else None

    def _blocked_until(self) -> dict[str, float]:
        try:
            return {str(k): float(v) for k, v in json.loads(self._path.read_text()).items()}
        except (OSError, ValueError, AttributeError):
            return {}
