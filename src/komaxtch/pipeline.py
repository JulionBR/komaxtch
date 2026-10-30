"""Orquestra a conversão: lê o livro, gera as telas e grava o XTCH."""

from __future__ import annotations

import os
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from komaxtch.cache import DetectionCache
from komaxtch.detect.base import BubbleDetector, PanelDetector
from komaxtch.detect.chain import FallbackPanelDetector
from komaxtch.detect import gemini, ollama
from komaxtch.detect.vision import Throttle, VisionPanelDetector
from komaxtch.detect.panels import OpenCvPanelDetector
from komaxtch.devices import DeviceProfile
from komaxtch.ingest import SourceFactory
from komaxtch.layout import (
    AUTO_ZOOM_BELOW_SCALE,
    ZOOM_EVERY_PANEL,
    BubbleLayout,
    LayoutMode,
    LayoutStrategy,
    PageLayout,
    PanelLayout,
    Rotation,
    ScreenComposer,
)
from komaxtch.models import Book, Crop, Page, ReadingDirection
from komaxtch.normalize import PageNormalizer
from komaxtch.reading_order import ReadingOrderSorter
from komaxtch.render import Dithering, EInkRenderer
from komaxtch.xtch.writer import BookMetadata, XtchWriter

GEMINI_KEY_VARIABLE = "GEMINI_API_KEY"
GEMINI_MODELS_VARIABLE = "GEMINI_MODELS"  # lista separada por vírgulas, em ordem de preferência
OLLAMA_MODEL_VARIABLE = "OLLAMA_MODEL"
OLLAMA_HOST_VARIABLE = "OLLAMA_HOST"
FALLBACK_DIRECTION = ReadingDirection.LEFT_TO_RIGHT

ProgressCallback = Callable[[int, int], None]
StatusCallback = Callable[[str], None]


@dataclass(frozen=True)
class ConversionSettings:
    device: DeviceProfile
    mode: LayoutMode = LayoutMode.AUTO
    direction: ReadingDirection | None = None  # None = usar o metadado do livro
    dithering: Dithering = Dithering.FLOYD
    rotation: Rotation = Rotation.NONE  # girar quadros largos obriga a virar o aparelho
    use_local_ai: bool = False  # Ollama pesa ~4 GB de RAM; só quando pedido
    use_cloud: bool = True
    split_spreads: bool = True


@dataclass
class ConversionReport:
    title: str
    direction: ReadingDirection
    direction_detected: bool
    pages: int = 0
    screens: int = 0
    detections: Counter[str] = field(default_factory=Counter)
    ai_pending_pages: int = 0  # páginas difíceis que ficaram para depois (cota esgotada)
    ai_resume_hint: str | None = None


class LayoutFactory:
    """Monta a estratégia de layout para um livro, já que a direção de leitura muda por livro."""

    def __init__(self, settings: ConversionSettings, cache: DetectionCache | None = None) -> None:
        self._settings = settings
        self._cache = cache or DetectionCache()
        self._bubbles: BubbleDetector | None = None
        self.cloud_enabled = settings.use_cloud and bool(os.environ.get(GEMINI_KEY_VARIABLE))
        self._ollama = self._ollama_client() if settings.use_local_ai else None
        self.vision_detectors: list[VisionPanelDetector] = []

    @property
    def local_ai_enabled(self) -> bool:
        return self._ollama is not None

    def build(self, direction: ReadingDirection, composer: ScreenComposer) -> LayoutStrategy:
        mode = self._settings.mode
        if mode is LayoutMode.PAGE:
            return PageLayout()
        panels = PanelLayout(self._panel_detector(direction), ReadingOrderSorter(direction))
        if mode is LayoutMode.PANEL:
            return panels
        zoom_below = AUTO_ZOOM_BELOW_SCALE if mode is LayoutMode.AUTO else ZOOM_EVERY_PANEL
        return BubbleLayout(panels, self._bubble_detector(), composer, direction, zoom_below)

    def _panel_detector(self, direction: ReadingDirection) -> PanelDetector:
        vision: list[VisionPanelDetector] = []
        if self.cloud_enabled:
            throttle = Throttle(gemini.MIN_SECONDS_BETWEEN_CALLS)
            vision.append(VisionPanelDetector(self._gemini_client(), direction, self._cache, throttle))
        if self._ollama is not None:
            vision.append(VisionPanelDetector(self._ollama, direction, self._cache))
        self.vision_detectors = vision
        local = OpenCvPanelDetector()
        return FallbackPanelDetector([local, *vision], refiner=local)

    @staticmethod
    def _gemini_client() -> gemini.GeminiModelRotation:
        key = os.environ[GEMINI_KEY_VARIABLE]
        configured = os.environ.get(GEMINI_MODELS_VARIABLE, "")
        models = [m.strip() for m in configured.split(",") if m.strip()] or list(gemini.DEFAULT_MODELS)
        return gemini.GeminiModelRotation([gemini.GoogleGenAiClient(key, model) for model in models])

    @staticmethod
    def _ollama_client() -> ollama.OllamaClient | None:
        """Usa o Ollama só se ele estiver rodando com o modelo já baixado."""
        client = ollama.OllamaClient(
            os.environ.get(OLLAMA_MODEL_VARIABLE, ollama.DEFAULT_MODEL),
            os.environ.get(OLLAMA_HOST_VARIABLE, ollama.DEFAULT_HOST),
        )
        return client if client.is_ready() else None

    def _bubble_detector(self) -> BubbleDetector:
        if self._bubbles is None:
            from komaxtch.detect.bubbles import YoloBubbleDetector

            self._bubbles = YoloBubbleDetector()
        return self._bubbles


class ConversionPipeline:
    def __init__(
        self,
        settings: ConversionSettings,
        layouts: LayoutFactory | None = None,
        sources: SourceFactory | None = None,
    ) -> None:
        self._settings = settings
        self._layouts = layouts or LayoutFactory(settings)
        self._sources = sources or SourceFactory()
        self._composer = ScreenComposer(settings.device, settings.rotation)
        self._renderer = EInkRenderer(settings.dithering)

    @property
    def composer(self) -> ScreenComposer:
        return self._composer

    @property
    def renderer(self) -> EInkRenderer:
        return self._renderer

    def convert(
        self,
        source: Path,
        output: Path,
        progress: ProgressCallback | None = None,
        status: StatusCallback | None = None,
    ) -> ConversionReport:
        book = self._sources.create(source).read()
        report, pages, layout = self.prepare(book)
        if status and self._layouts.vision_detectors:
            status("Analisando as páginas difíceis com a IA (em lotes)...")
        layout.prefetch(pages)
        with XtchWriter(output, BookMetadata(book.title, book.author)) as writer:
            for index, page in enumerate(pages, start=1):
                for crop in layout.crops(page):
                    writer.add_page(self._renderer.render(self._composer.compose(crop)))
                    report.screens += 1
                self._record_detection(report, layout)
                if progress:
                    progress(index, len(pages))
        self._record_ai_pending(report)
        return report

    def prepare(self, book: Book) -> tuple[ConversionReport, list[Page], LayoutStrategy]:
        direction = self._settings.direction or book.direction or FALLBACK_DIRECTION
        detected = self._settings.direction is None and book.direction is not None
        pages = PageNormalizer(direction, self._settings.split_spreads).normalize(book.pages)
        report = ConversionReport(book.title, direction, detected, pages=len(pages))
        return report, pages, self._layouts.build(direction, self._composer)

    def screens(self, layout: LayoutStrategy, page: Page) -> Iterator[tuple[Crop, Image.Image]]:
        """Telas de uma página já como aparecem no aparelho (usado por prévia e depuração)."""
        for crop in layout.crops(page):
            levels = self._renderer.render(self._composer.compose(crop))
            yield crop, self._renderer.to_image(levels)

    def _record_ai_pending(self, report: ConversionReport) -> None:
        for detector in self._layouts.vision_detectors:
            report.ai_pending_pages += detector.skipped_pages
            report.ai_resume_hint = report.ai_resume_hint or detector.resume_hint()

    @staticmethod
    def _record_detection(report: ConversionReport, layout: LayoutStrategy) -> None:
        detection = layout.last_detection
        report.detections["page" if detection is None else detection.source] += 1
