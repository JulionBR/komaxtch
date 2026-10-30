"""Cliente da Gemini API (camada gratuita) para o detector de visão.

Na camada gratuita a cota é de poucas requisições por dia e por modelo (20, em out/2026).
Por isso cada requisição leva várias páginas, o raciocínio do modelo fica desligado (metade
dos tokens) e, quando um modelo esgota a cota do dia, a rotação passa para o próximo."""

from __future__ import annotations

import logging
import re
from datetime import datetime

from komaxtch.detect.quota import QuotaLedger
from komaxtch.detect.vision import (
    RESPONSE_SCHEMA,
    QuotaExhaustedError,
    RateLimitedError,
    TransientError,
    VisionClient,
)

logger = logging.getLogger(__name__)

# Em ordem de preferência: o 3.5-flash detecta melhor; o 3.1-flash-lite tem cota própria.
DEFAULT_MODELS = ("gemini-3.5-flash", "gemini-3.1-flash-lite")
PAGES_PER_REQUEST = 8
MIN_SECONDS_BETWEEN_CALLS = 6.5  # abaixo do limite gratuito por minuto
DAILY_QUOTA_MARKER = "PerDay"
_RETRY_DELAY = re.compile(r"([\d.]+)s")


class GoogleGenAiClient(VisionClient):
    """Um modelo Gemini específico."""

    name = "gemini"
    batch_size = PAGES_PER_REQUEST

    def __init__(self, api_key: str, model: str) -> None:
        from google import genai  # import tardio: só carrega o SDK quando a nuvem é usada

        self._client = genai.Client(api_key=api_key)
        self.model = model

    def generate_json(self, images_jpeg: list[bytes], prompt: str) -> str:
        from google.genai import errors, types

        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_json_schema=RESPONSE_SCHEMA,
            temperature=0,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        )
        contents = [types.Part.from_bytes(data=image, mime_type="image/jpeg") for image in images_jpeg]
        try:
            response = self._client.models.generate_content(
                model=self.model, contents=[*contents, prompt], config=config
            )
        except errors.ClientError as error:
            if error.code == 429:
                raise _quota_error(error) from error
            raise
        except errors.ServerError as error:
            raise TransientError(str(error)) from error
        return response.text or '{"pages": []}'


class GeminiModelRotation(VisionClient):
    """Usa o primeiro modelo com cota; quando a cota diária dele acaba, passa para o próximo."""

    name = "gemini"
    batch_size = PAGES_PER_REQUEST

    def __init__(self, clients: list[GoogleGenAiClient], ledger: QuotaLedger | None = None) -> None:
        self._clients = clients
        self._ledger = ledger or QuotaLedger()

    def generate_json(self, images_jpeg: list[bytes], prompt: str) -> str:
        for client in self._clients:
            if not self._ledger.available(client.model):
                continue
            try:
                return client.generate_json(images_jpeg, prompt)
            except QuotaExhaustedError as error:
                logger.info("Cota diária do %s esgotada; tentando o próximo modelo.", client.model)
                self._ledger.mark_exhausted(client.model, error.retry_after)
        raise QuotaExhaustedError("cota diária gratuita esgotada em todos os modelos")

    def resume_hint(self) -> str | None:
        when = self._ledger.next_available(c.model for c in self._clients)
        if when is None:
            return None
        return datetime.fromtimestamp(when).strftime("%d/%m às %H:%M")


def _quota_error(error) -> Exception:
    details = (getattr(error, "details", None) or {}).get("error", {}).get("details", [])
    violations = [v for d in details for v in d.get("violations", [])]
    daily = any(DAILY_QUOTA_MARKER in v.get("quotaId", "") for v in violations)
    retry_after = next((_seconds(d["retryDelay"]) for d in details if "retryDelay" in d), None)
    if daily:
        return QuotaExhaustedError(str(error), retry_after)
    return RateLimitedError(str(error), retry_after)


def _seconds(delay: str) -> float | None:
    match = _RETRY_DELAY.fullmatch(delay.strip())
    return float(match.group(1)) if match else None
