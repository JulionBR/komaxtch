"""Primitivas geométricas imutáveis usadas por detecção e layout."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BoundingBox:
    """Retângulo em pixels, com `right`/`bottom` exclusivos."""

    left: int
    top: int
    right: int
    bottom: int

    def __post_init__(self) -> None:
        if self.right < self.left or self.bottom < self.top:
            raise ValueError(f"Caixa inválida: {self}")

    @classmethod
    def from_size(cls, width: int, height: int) -> BoundingBox:
        return cls(0, 0, width, height)

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height if self.height else 0.0

    @property
    def center(self) -> tuple[float, float]:
        return (self.left + self.right) / 2, (self.top + self.bottom) / 2

    def intersection(self, other: BoundingBox) -> BoundingBox | None:
        left, top = max(self.left, other.left), max(self.top, other.top)
        right, bottom = min(self.right, other.right), min(self.bottom, other.bottom)
        if right <= left or bottom <= top:
            return None
        return BoundingBox(left, top, right, bottom)

    def union(self, other: BoundingBox) -> BoundingBox:
        return BoundingBox(
            min(self.left, other.left),
            min(self.top, other.top),
            max(self.right, other.right),
            max(self.bottom, other.bottom),
        )

    def iou(self, other: BoundingBox) -> float:
        overlap = self.intersection(other)
        if overlap is None:
            return 0.0
        return overlap.area / (self.area + other.area - overlap.area)

    def overlap_ratio(self, other: BoundingBox) -> float:
        """Fração desta caixa coberta por `other`."""
        overlap = self.intersection(other)
        return overlap.area / self.area if overlap and self.area else 0.0

    def contains(self, other: BoundingBox) -> bool:
        return (
            self.left <= other.left
            and self.top <= other.top
            and self.right >= other.right
            and self.bottom >= other.bottom
        )

    def expand(self, margin: int) -> BoundingBox:
        return BoundingBox(
            self.left - margin, self.top - margin, self.right + margin, self.bottom + margin
        )

    def clamp(self, bounds: BoundingBox) -> BoundingBox:
        return BoundingBox(
            max(self.left, bounds.left),
            max(self.top, bounds.top),
            min(self.right, bounds.right),
            min(self.bottom, bounds.bottom),
        )

    def as_tuple(self) -> tuple[int, int, int, int]:
        return self.left, self.top, self.right, self.bottom
