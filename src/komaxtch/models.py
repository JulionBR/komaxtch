"""Objetos de domínio compartilhados pelo pipeline."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from functools import cached_property

from PIL import Image

from komaxtch.geometry import BoundingBox


class ReadingDirection(Enum):
    LEFT_TO_RIGHT = "ltr"
    RIGHT_TO_LEFT = "rtl"


@dataclass
class Page:
    """Uma página de origem, já em tons de cinza."""

    number: int
    image: Image.Image

    @property
    def bounds(self) -> BoundingBox:
        return BoundingBox.from_size(*self.image.size)

    @cached_property
    def fingerprint(self) -> str:
        return hashlib.sha256(self.image.tobytes()).hexdigest()


@dataclass
class Book:
    title: str
    author: str
    direction: ReadingDirection | None
    pages: list[Page] = field(default_factory=list)


@dataclass(frozen=True)
class Panel:
    box: BoundingBox
    bubbles: tuple[BoundingBox, ...] = ()


@dataclass(frozen=True)
class Crop:
    """Região da página que vira uma tela."""

    page: Page
    box: BoundingBox
    kind: str  # "page", "panel" ou "bubble"
