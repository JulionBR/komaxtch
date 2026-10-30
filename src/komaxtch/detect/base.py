"""Contratos dos detectores de quadros e balões."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from komaxtch.geometry import BoundingBox
from komaxtch.models import Page

FULL_CONFIDENCE = 1.0
NO_CONFIDENCE = 0.0


@dataclass(frozen=True)
class PanelDetection:
    panels: tuple[BoundingBox, ...]
    confidence: float
    source: str

    @classmethod
    def whole_page(cls, page: Page) -> PanelDetection:
        return cls((page.bounds,), FULL_CONFIDENCE, "whole-page")


class PanelDetector(ABC):
    name: str

    @abstractmethod
    def detect(self, page: Page) -> PanelDetection: ...

    def prefetch(self, pages: list[Page]) -> None:
        """Analisa várias páginas de uma vez (detectores remotos agrupam em lotes)."""


class BubbleDetector(ABC):
    @abstractmethod
    def detect(self, page: Page) -> list[BoundingBox]: ...


class NoBubbles(BubbleDetector):
    def detect(self, page: Page) -> list[BoundingBox]:
        return []
