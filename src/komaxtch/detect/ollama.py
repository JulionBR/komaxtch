"""Cliente do Ollama: modelo de visão rodando no próprio Mac, grátis e sem limite de uso."""

from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request

from komaxtch.detect.vision import RESPONSE_SCHEMA, ServiceUnavailableError, VisionClient

DEFAULT_HOST = "http://localhost:11434"
DEFAULT_MODEL = "qwen3-vl:4b-instruct"
REQUEST_TIMEOUT_SECONDS = 300  # o primeiro pedido inclui carregar o modelo na memória
PROBE_TIMEOUT_SECONDS = 2
CONTEXT_TOKENS = 2048  # 1 imagem (~1.200 tokens) + resposta; contexto menor cabe melhor na GPU


class OllamaClient(VisionClient):
    name = "ollama"
    batch_size = 1  # várias imagens não cabem no contexto reduzido nem na RAM de 8 GB

    def __init__(self, model: str = DEFAULT_MODEL, host: str = DEFAULT_HOST) -> None:
        self.model = model
        self._host = host.rstrip("/")

    def is_ready(self) -> bool:
        """O servidor responde e o modelo já foi baixado?"""
        try:
            with urllib.request.urlopen(f"{self._host}/api/tags", timeout=PROBE_TIMEOUT_SECONDS) as response:
                models = json.load(response).get("models", [])
        except (urllib.error.URLError, OSError, ValueError):
            return False
        return any(model.get("name") == self.model for model in models)

    def generate_json(self, images_jpeg: list[bytes], prompt: str) -> str:
        images = [base64.b64encode(image).decode() for image in images_jpeg]
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt, "images": images}],
            "format": RESPONSE_SCHEMA,
            "stream": False,
            "options": {"temperature": 0, "num_ctx": CONTEXT_TOKENS},
        }
        request = urllib.request.Request(
            f"{self._host}/api/chat",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                return json.load(response)["message"]["content"]
        except urllib.error.URLError as error:
            raise ServiceUnavailableError(str(error.reason)) from error
