"""Detecção local de balões de fala com um modelo YOLOv8 treinado em mangá, manhwa e HQ."""

from __future__ import annotations

import logging

from komaxtch.detect.base import BubbleDetector
from komaxtch.geometry import BoundingBox
from komaxtch.models import Page

logger = logging.getLogger(__name__)

MODEL_REPO = "ogkalu/comic-speech-bubble-detector-yolov8m"  # Apache-2.0
MODEL_FILE = "comic-speech-bubble-detector.pt"
MIN_CONFIDENCE = 0.35


class YoloBubbleDetector(BubbleDetector):
    """Baixa os pesos do Hugging Face na primeira execução e reutiliza o cache depois."""

    def __init__(self, min_confidence: float = MIN_CONFIDENCE) -> None:
        self._min_confidence = min_confidence
        self._model = None

    def detect(self, page: Page) -> list[BoundingBox]:
        result = self._load()(page.image.convert("RGB"), conf=self._min_confidence, verbose=False)[0]
        return [
            BoundingBox(*(round(v) for v in xyxy)).clamp(page.bounds)
            for xyxy in result.boxes.xyxy.tolist()
        ]

    def _load(self):
        if self._model is None:
            from huggingface_hub import hf_hub_download
            from huggingface_hub.utils import logging as hub_logging
            from ultralytics import YOLO

            hub_logging.set_verbosity_error()
            logger.info("Carregando o detector de balões (%s)...", MODEL_REPO)
            self._model = YOLO(hf_hub_download(MODEL_REPO, MODEL_FILE))
        return self._model
