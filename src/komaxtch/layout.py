"""Decide quais recortes de cada página viram telas e monta cada tela no tamanho do aparelho."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from enum import Enum

from PIL import Image

from komaxtch.detect.cleanup import PanelCleaner
from komaxtch.detect.base import BubbleDetector, PanelDetection, PanelDetector
from komaxtch.devices import DeviceProfile
from komaxtch.geometry import BoundingBox
from komaxtch.models import Crop, Page, ReadingDirection
from komaxtch.reading_order import ReadingOrderSorter

PANEL_MARGIN_RATIO = 0.01
BUBBLE_MARGIN_RATIO = 0.25
ROTATION_GAIN = 1.25  # só gira se o quadro ficar pelo menos 25% maior
MIN_BUBBLE_ON_SCREEN_PX = 170  # balão menor que isso na tela do quadro ganha uma tela de zoom
BUBBLE_GROUP_GAP_RATIO = 0.04
BACKGROUND = 255
AUTO_ZOOM_BELOW_SCALE = 0.6  # modo auto: zoom só quando o quadro aparece com menos de 60% do tamanho
ZOOM_EVERY_PANEL = math.inf


class LayoutMode(Enum):
    AUTO = "auto"
    PAGE = "page"
    PANEL = "panel"
    BUBBLE = "bubble"


class Rotation(Enum):
    CLOCKWISE = "cw"
    COUNTER_CLOCKWISE = "ccw"

    @property
    def degrees(self) -> int:
        return -90 if self is Rotation.CLOCKWISE else 90


class ScreenComposer:
    """Encaixa um recorte na tela, girando quadros largos quando isso os deixa maiores."""

    def __init__(self, device: DeviceProfile, rotation: Rotation = Rotation.CLOCKWISE) -> None:
        self._device = device
        self._rotation = rotation

    def compose(self, crop: Crop) -> Image.Image:
        image = crop.page.image.crop(crop.box.as_tuple())
        if self.should_rotate(crop.box):
            image = image.rotate(self._rotation.degrees, expand=True)
        return self._fit(image)

    def should_rotate(self, box: BoundingBox) -> bool:
        upright = self.fit_scale(box.width, box.height)
        rotated = self.fit_scale(box.height, box.width)
        return rotated >= upright * ROTATION_GAIN

    def display_scale(self, box: BoundingBox) -> float:
        if self.should_rotate(box):
            return self.fit_scale(box.height, box.width)
        return self.fit_scale(box.width, box.height)

    def fit_scale(self, width: int, height: int) -> float:
        return min(self._device.width / width, self._device.height / height)

    def _fit(self, image: Image.Image) -> Image.Image:
        scale = self.fit_scale(*image.size)
        size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
        resized = image.resize(size, Image.Resampling.LANCZOS)
        screen = Image.new("L", (self._device.width, self._device.height), BACKGROUND)
        offset = ((screen.width - resized.width) // 2, (screen.height - resized.height) // 2)
        screen.paste(resized, offset)
        return screen


class LayoutStrategy(ABC):
    last_detection: PanelDetection | None = None

    @abstractmethod
    def crops(self, page: Page) -> list[Crop]: ...

    def prefetch(self, pages: list[Page]) -> None:
        """Adianta, em lote, a detecção de todas as páginas do livro."""


class PageLayout(LayoutStrategy):
    def crops(self, page: Page) -> list[Crop]:
        return [Crop(page, page.bounds, "page")]


class PanelLayout(LayoutStrategy):
    def __init__(self, detector: PanelDetector, sorter: ReadingOrderSorter) -> None:
        self._detector = detector
        self._sorter = sorter
        self.last_detection: PanelDetection | None = None

    def crops(self, page: Page) -> list[Crop]:
        return [Crop(page, box, "panel") for box in self.panel_boxes(page)]

    def prefetch(self, pages: list[Page]) -> None:
        self._detector.prefetch(pages)

    def panel_boxes(self, page: Page) -> list[BoundingBox]:
        detection = self._detector.detect(page)
        self.last_detection = detection
        cleaned = PanelCleaner(page.bounds).clean(list(detection.panels))
        boxes = self._sorter.sort(cleaned)
        margin = round(PANEL_MARGIN_RATIO * max(page.image.size))
        return [box.expand(margin).clamp(page.bounds) for box in boxes]


class BubbleLayout(LayoutStrategy):
    """Cada quadro e, logo depois, telas de zoom nos balões que ficariam pequenos demais.

    Com `zoom_below_scale`, o zoom só acontece em quadros mostrados muito reduzidos (página
    inteira, splash); os demais seguem um por tela. Os balões só são procurados nessas páginas."""

    def __init__(
        self,
        panels: PanelLayout,
        bubbles: BubbleDetector,
        composer: ScreenComposer,
        direction: ReadingDirection,
        zoom_below_scale: float = ZOOM_EVERY_PANEL,
    ) -> None:
        self._panels = panels
        self._bubbles = bubbles
        self._composer = composer
        self._direction = direction
        self._zoom_below_scale = zoom_below_scale

    @property
    def last_detection(self) -> PanelDetection | None:
        return self._panels.last_detection

    def prefetch(self, pages: list[Page]) -> None:
        self._panels.prefetch(pages)

    def crops(self, page: Page) -> list[Crop]:
        panels = self._panels.panel_boxes(page)
        shrunken = [p for p in panels if self._composer.display_scale(p) < self._zoom_below_scale]
        bubbles = self._bubbles.detect(page) if shrunken else []
        crops: list[Crop] = []
        for panel in panels:
            crops.append(Crop(page, panel, "panel"))
            if panel in shrunken:
                crops.extend(self._zoom_crops(page, panel, bubbles))
        return crops

    def _zoom_crops(self, page: Page, panel: BoundingBox, bubbles: list[BoundingBox]) -> list[Crop]:
        inside = [b for b in bubbles if b.overlap_ratio(panel) > 0.5]
        scale = self._composer.display_scale(panel)
        small = [b for b in inside if min(b.width, b.height) * scale < MIN_BUBBLE_ON_SCREEN_PX]
        groups = self._group(small, gap=round(BUBBLE_GROUP_GAP_RATIO * max(page.image.size)))
        ordered = self._reading_order(groups)
        return [Crop(page, self._with_margin(box, page), "bubble") for box in ordered]

    @staticmethod
    def _group(boxes: list[BoundingBox], gap: int) -> list[BoundingBox]:
        """Junta balões vizinhos numa só tela para não multiplicar o número de telas."""
        groups: list[BoundingBox] = []
        for box in boxes:
            for neighbour in [g for g in groups if g.expand(gap).intersection(box) is not None]:
                groups.remove(neighbour)
                box = box.union(neighbour)
            groups.append(box)
        return groups

    def _reading_order(self, boxes: list[BoundingBox]) -> list[BoundingBox]:
        rtl = self._direction is ReadingDirection.RIGHT_TO_LEFT
        return sorted(boxes, key=lambda b: (b.top, -b.right if rtl else b.left))

    @staticmethod
    def _with_margin(box: BoundingBox, page: Page) -> BoundingBox:
        margin = round(BUBBLE_MARGIN_RATIO * min(box.width, box.height))
        return box.expand(margin).clamp(page.bounds)
