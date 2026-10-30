"""Detector de quadros baseado em modelo de visão (na nuvem via Gemini ou local via Ollama).

O detector é o mesmo para qualquer modelo: manda as páginas difíceis em lotes (uma requisição
para várias páginas, o que poupa a cota diária), guarda as respostas em cache, respeita o
limite por minuto e, quando a cota acaba, se desliga até o fim da execução sem interromper
a conversão. Rodar de novo depois retoma só as páginas que faltaram."""

from __future__ import annotations

import io
import json
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator

from PIL import Image

from komaxtch.cache import DetectionCache
from komaxtch.detect.base import FULL_CONFIDENCE, NO_CONFIDENCE, PanelDetection, PanelDetector
from komaxtch.geometry import BoundingBox
from komaxtch.models import Page, ReadingDirection

logger = logging.getLogger(__name__)

RATE_LIMIT_BACKOFF_SECONDS = (20.0, 60.0)
TRANSIENT_BACKOFF_SECONDS = (5.0, 15.0)
UPLOAD_MAX_SIDE_PX = 1024
UPLOAD_JPEG_QUALITY = 85
NORMALIZED_SCALE = 1000  # os modelos devolvem coordenadas relativas de 0 a 1000

PROMPT = """You will receive {count} page(s) of a {kind} (read {order}), numbered 1 to {count} in the order given.
For each page, detect every comic panel. A panel is a framed region of artwork; if a page is a
single illustration without separate panels, return one box covering the whole illustration.
Answer only with JSON: {{"pages": [{{"page": 1, "panels": [[x1, y1, x2, y2], ...]}}, ...]}}, one entry
per page, with coordinates normalized to 0-1000 relative to that page."""

# Schema da resposta (Gemini: response_json_schema; Ollama: format). Caixas como listas simples
# deixam a resposta ~3x mais curta do que objetos {"bbox_2d": ...}.
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "pages": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "page": {"type": "integer"},
                    "panels": {"type": "array", "items": {"type": "array", "items": {"type": "integer"}}},
                },
                "required": ["page", "panels"],
            },
        }
    },
    "required": ["pages"],
}


class RateLimitedError(Exception):
    """Limite por minuto excedido (HTTP 429); passa depois de `retry_after` segundos."""

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class QuotaExhaustedError(Exception):
    """Cota diária esgotada; só volta depois de `retry_after` segundos."""

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class TransientError(Exception):
    """Falha passageira do serviço (ex.: HTTP 503 por excesso de demanda); vale tentar de novo."""


class ServiceUnavailableError(Exception):
    """O serviço não responde (ex.: Ollama desligado); não adianta tentar de novo."""


class VisionClient(ABC):
    name: str
    batch_size: int = 1  # páginas por requisição

    @abstractmethod
    def generate_json(self, images_jpeg: list[bytes], prompt: str) -> str: ...

    def resume_hint(self) -> str | None:
        """Quando vale rodar de novo, se o serviço ficou indisponível."""
        return None


class Throttle:
    def __init__(
        self,
        min_interval: float,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._min_interval = min_interval
        self._clock = clock
        self.sleep = sleep
        self._last_call: float | None = None

    def wait(self) -> None:
        if self._last_call is not None:
            remaining = self._min_interval - (self._clock() - self._last_call)
            if remaining > 0:
                self.sleep(remaining)
        self._last_call = self._clock()


class VisionPanelDetector(PanelDetector):
    def __init__(
        self,
        client: VisionClient,
        direction: ReadingDirection,
        cache: DetectionCache,
        throttle: Throttle | None = None,
    ) -> None:
        self.name = client.name
        self._client = client
        self._direction = direction
        self._cache = cache
        self._throttle = throttle or Throttle(0)
        self._namespace = f"{client.name}-{direction.value}"
        self._disabled = False
        self._skipped: set[str] = set()

    @property
    def skipped_pages(self) -> int:
        """Páginas que ficaram sem análise porque o serviço ficou indisponível."""
        return len(self._skipped)

    def resume_hint(self) -> str | None:
        return self._client.resume_hint()

    def detect(self, page: Page) -> PanelDetection:
        boxes = self._cached(page)
        if boxes is None:
            self.prefetch([page])
            boxes = self._cached(page)
        return self._failure() if boxes is None else self._detection(boxes)

    def prefetch(self, pages: list[Page]) -> None:
        pending = list({p.fingerprint: p for p in pages if self._cached(p) is None}.values())
        for batch in _chunks(pending, self._client.batch_size):
            if self._disabled:
                self._skipped.update(p.fingerprint for p in batch)
                continue
            try:
                results = self._ask(batch)
            except (QuotaExhaustedError, ServiceUnavailableError) as error:
                logger.warning("%s indisponível (%s); páginas difíceis seguem com o palpite local.", self.name, error)
                self._disabled = True
                self._skipped.update(p.fingerprint for p in batch)
                continue
            except Exception as error:  # resposta inválida: segue sem o modelo neste lote
                logger.warning("%s falhou nas páginas %s: %s", self.name, [p.number for p in batch], error)
                continue
            for page, boxes in zip(batch, results):
                if boxes:
                    self._cache.put(self._namespace, page.fingerprint, boxes)

    def _cached(self, page: Page) -> list[BoundingBox] | None:
        return self._cache.get(self._namespace, page.fingerprint)

    def _ask(self, batch: list[Page]) -> list[list[BoundingBox]]:
        images = [_jpeg_for_upload(page.image) for page in batch]
        prompt = self._prompt(len(batch))
        rate_limit_waits = iter(RATE_LIMIT_BACKOFF_SECONDS)
        transient_waits = iter(TRANSIENT_BACKOFF_SECONDS)
        while True:
            try:
                self._throttle.wait()
                return parse_batch(self._client.generate_json(images, prompt), batch)
            except RateLimitedError as error:
                wait = next(rate_limit_waits, None)
                if wait is None:
                    raise QuotaExhaustedError(str(error), error.retry_after) from error
                wait = max(wait, error.retry_after or 0)
            except TransientError:
                wait = next(transient_waits, None)
                if wait is None:
                    raise  # segue sem a IA só neste lote
            self._throttle.sleep(wait)

    def _prompt(self, count: int) -> str:
        if self._direction is ReadingDirection.RIGHT_TO_LEFT:
            return PROMPT.format(count=count, kind="Japanese manga", order="right to left")
        return PROMPT.format(count=count, kind="comic book", order="left to right")

    def _detection(self, boxes: list[BoundingBox]) -> PanelDetection:
        # A ordem devolvida pelos modelos não é confiável; o ReadingOrderSorter decide.
        confidence = FULL_CONFIDENCE if boxes else NO_CONFIDENCE
        return PanelDetection(tuple(boxes), confidence, self.name)

    def _failure(self) -> PanelDetection:
        return PanelDetection((), NO_CONFIDENCE, self.name)


def parse_batch(answer: str, pages: list[Page]) -> list[list[BoundingBox]]:
    """Caixas de cada página, na ordem de `pages`. Aceita {"pages": [...]} e, para uma
    página só, os formatos simples aceitos por `parse_boxes`."""
    data = json.loads(_strip_code_fence(answer))
    if not (isinstance(data, dict) and "pages" in data):
        if len(pages) != 1:
            raise ValueError("Resposta sem a lista de páginas")
        return [parse_boxes(answer, pages[0])]
    panels_by_page = {
        int(entry.get("page", position)): entry.get("panels", [])
        for position, entry in enumerate(data["pages"], start=1)
    }
    return [_to_boxes(panels_by_page.get(number, []), page) for number, page in enumerate(pages, start=1)]


def parse_boxes(answer: str, page: Page) -> list[BoundingBox]:
    """Aceita listas [x1, y1, x2, y2], `bbox_2d` [x1, y1, x2, y2] e `box_2d` [ymin, xmin, ymax, xmax]."""
    items = json.loads(_strip_code_fence(answer))
    if isinstance(items, dict):
        items = next((v for v in items.values() if isinstance(v, list)), [items])
    return _to_boxes(items, page)


def _to_boxes(items: list, page: Page) -> list[BoundingBox]:
    width, height = page.image.size
    boxes = []
    for item in items:
        x1, y1, x2, y2 = _corners(item)
        box = BoundingBox(
            round(min(x1, x2) * width / NORMALIZED_SCALE),
            round(min(y1, y2) * height / NORMALIZED_SCALE),
            round(max(x1, x2) * width / NORMALIZED_SCALE),
            round(max(y1, y2) * height / NORMALIZED_SCALE),
        ).clamp(page.bounds)
        if box.area:
            boxes.append(box)
    return boxes


def _corners(item: dict | list) -> tuple[float, float, float, float]:
    if isinstance(item, list):
        x1, y1, x2, y2 = (float(v) for v in item)
    elif "bbox_2d" in item:
        x1, y1, x2, y2 = (float(v) for v in item["bbox_2d"])
    else:
        y1, x1, y2, x2 = (float(v) for v in item["box_2d"])
    return x1, y1, x2, y2


def _chunks(pages: list[Page], size: int) -> Iterator[list[Page]]:
    for start in range(0, len(pages), size):
        yield pages[start : start + size]


def _strip_code_fence(answer: str) -> str:
    text = answer.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
    return text


def _jpeg_for_upload(image: Image.Image) -> bytes:
    preview = image.copy()
    preview.thumbnail((UPLOAD_MAX_SIDE_PX, UPLOAD_MAX_SIDE_PX))
    buffer = io.BytesIO()
    preview.save(buffer, format="JPEG", quality=UPLOAD_JPEG_QUALITY)
    return buffer.getvalue()
