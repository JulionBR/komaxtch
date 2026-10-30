"""Limpeza das páginas antes da detecção: corte de margens e divisão de páginas duplas."""

from __future__ import annotations

import numpy as np
from PIL import Image

from komaxtch.models import Page, ReadingDirection

WHITE_THRESHOLD = 235
TRIM_PADDING_PX = 4
MIN_CONTENT_FRACTION = 0.5  # não corta se sobrar menos que isso (página quase vazia)
SPREAD_ASPECT_RATIO = 1.1  # largura/altura acima disso indica página dupla


class PageNormalizer:
    def __init__(self, direction: ReadingDirection, split_spreads: bool = True) -> None:
        self._direction = direction
        self._split_spreads = split_spreads

    def normalize(self, pages: list[Page]) -> list[Page]:
        result: list[Page] = []
        for page in pages:
            for image in self._split(page.image):
                result.append(Page(len(result) + 1, self._trim(image)))
        return result

    def _split(self, image: Image.Image) -> list[Image.Image]:
        width, height = image.size
        if not self._split_spreads or width <= height * SPREAD_ASPECT_RATIO:
            return [image]
        middle = width // 2
        left = image.crop((0, 0, middle, height))
        right = image.crop((middle, 0, width, height))
        halves = [right, left] if self._direction is ReadingDirection.RIGHT_TO_LEFT else [left, right]
        return halves

    @staticmethod
    def _trim(image: Image.Image) -> Image.Image:
        content = np.asarray(image) < WHITE_THRESHOLD
        rows, cols = np.flatnonzero(content.any(axis=1)), np.flatnonzero(content.any(axis=0))
        if rows.size == 0 or cols.size == 0:
            return image
        width, height = image.size
        left = max(cols[0] - TRIM_PADDING_PX, 0)
        top = max(rows[0] - TRIM_PADDING_PX, 0)
        right = min(cols[-1] + 1 + TRIM_PADDING_PX, width)
        bottom = min(rows[-1] + 1 + TRIM_PADDING_PX, height)
        if (right - left) * (bottom - top) < MIN_CONTENT_FRACTION * width * height:
            return image
        return image.crop((left, top, right, bottom))
