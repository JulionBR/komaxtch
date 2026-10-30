"""Cadeia de fallback: tenta cada detector em ordem; se nenhum for confiável, usa o melhor
palpite razoável e, sem nenhum, a página inteira."""

from __future__ import annotations

from dataclasses import replace

from komaxtch.detect.base import PanelDetection, PanelDetector
from komaxtch.detect.panels import OpenCvPanelDetector
from komaxtch.models import Page

DEFAULT_MIN_CONFIDENCE = 0.6
BEST_EFFORT_CONFIDENCE = 0.35  # abaixo disso a página inteira é melhor que um palpite


class FallbackPanelDetector(PanelDetector):
    name = "auto"

    def __init__(
        self,
        detectors: list[PanelDetector],
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
        refiner: OpenCvPanelDetector | None = None,
    ) -> None:
        self._detectors = detectors
        self._min_confidence = min_confidence
        self._refiner = refiner

    def prefetch(self, pages: list[Page]) -> None:
        """Cada detector recebe, de uma vez, só as páginas em que os anteriores não foram confiáveis."""
        pending = list(pages)
        for detector in self._detectors:
            if not pending:
                return
            detector.prefetch(pending)
            pending = [page for page in pending if not self._is_confident(detector.detect(page))]

    def detect(self, page: Page) -> PanelDetection:
        best: PanelDetection | None = None
        for detector in self._detectors:
            detection = detector.detect(page)
            if not detection.panels:
                continue
            if self._is_confident(detection):
                return self._refined(page, detection, detector)
            if best is None or detection.confidence > best.confidence:
                best = detection
        if best is not None and best.confidence >= BEST_EFFORT_CONFIDENCE:
            return best
        return PanelDetection.whole_page(page)

    def _refined(self, page: Page, detection: PanelDetection, detector: PanelDetector) -> PanelDetection:
        if self._refiner is None or detector is self._refiner:
            return detection
        return replace(detection, panels=self._refiner.refine(page, detection.panels))

    def _is_confident(self, detection: PanelDetection) -> bool:
        return bool(detection.panels) and detection.confidence >= self._min_confidence
