import json

from PIL import Image

from komaxtch.cache import DetectionCache
from komaxtch.detect.base import PanelDetection, PanelDetector
from komaxtch.detect.chain import FallbackPanelDetector
from komaxtch.detect.gemini import GeminiModelRotation
from komaxtch.detect.quota import QuotaLedger
from komaxtch.detect.vision import (
    QuotaExhaustedError,
    RateLimitedError,
    ServiceUnavailableError,
    Throttle,
    TransientError,
    VisionClient,
    VisionPanelDetector,
)
from komaxtch.geometry import BoundingBox
from komaxtch.models import Page, ReadingDirection

WHOLE = [[0, 0, 1000, 1000]]


def page(number: int) -> Page:
    return Page(number, Image.new("L", (800, 1200), number % 256))


def batch_answer(*panels_per_page) -> str:
    return json.dumps({"pages": [{"page": i, "panels": p} for i, p in enumerate(panels_per_page, 1)]})


class FakeVision(VisionClient):
    name = "fake"

    def __init__(self, answers, batch_size=1, model="fake-model"):
        self._answers = list(answers)
        self.batch_size = batch_size
        self.model = model
        self.calls: list[int] = []  # páginas por chamada

    def generate_json(self, images_jpeg, prompt):
        self.calls.append(len(images_jpeg))
        answer = self._answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


def detector(client, tmp_path, sleeps=None):
    throttle = Throttle(0, sleep=(sleeps.append if sleeps is not None else lambda _: None))
    return VisionPanelDetector(client, ReadingDirection.RIGHT_TO_LEFT, DetectionCache(tmp_path), throttle)


def test_parses_every_box_format_and_caches(tmp_path):
    expected = (BoundingBox(400, 0, 800, 600),)
    answers = [
        batch_answer([[500, 0, 1000, 500]]),
        '```json\n{"panels": [{"bbox_2d": [500, 0, 1000, 500]}]}\n```',
        '[{"box_2d": [0, 500, 500, 1000]}]',
    ]
    for index, answer in enumerate(answers):
        client = FakeVision([answer])
        assert detector(client, tmp_path / str(index)).detect(page(1)).panels == expected
        again = detector(client, tmp_path / str(index)).detect(page(1))  # nova execução: cache
        assert again.panels == expected and len(client.calls) == 1


def test_prefetch_sends_pages_in_batches(tmp_path):
    pages = [page(n) for n in range(1, 11)]
    client = FakeVision([batch_answer(*[WHOLE] * 8), batch_answer(*[WHOLE] * 2)], batch_size=8)
    vision = detector(client, tmp_path)
    vision.prefetch(pages)
    assert client.calls == [8, 2]  # 10 páginas em 2 requisições
    assert all(vision.detect(p).confidence == 1.0 for p in pages)
    assert len(client.calls) == 2  # detect() usa o que o lote trouxe


def test_rate_limit_waits_the_time_the_api_asks(tmp_path):
    sleeps = []
    client = FakeVision([RateLimitedError("429", retry_after=37), batch_answer(WHOLE)])
    assert detector(client, tmp_path, sleeps).detect(page(1)).confidence == 1.0
    assert sleeps == [37]


def test_daily_quota_stops_calls_and_reports_pending_pages(tmp_path):
    client = FakeVision([QuotaExhaustedError("cota")], batch_size=2)
    vision = detector(client, tmp_path)
    vision.prefetch([page(1), page(2), page(3), page(4)])
    assert client.calls == [2]  # não insiste depois da cota diária
    assert vision.skipped_pages == 4
    assert vision.detect(page(5)).confidence == 0.0 and len(client.calls) == 1


def test_transient_errors_are_retried_without_giving_up_for_good(tmp_path):
    sleeps = []
    failures = [TransientError("503")] * 3
    client = FakeVision([TransientError("503"), batch_answer(WHOLE), *failures, batch_answer(WHOLE)])
    vision = detector(client, tmp_path, sleeps)
    assert vision.detect(page(1)).confidence == 1.0  # 2ª tentativa deu certo
    assert vision.detect(page(2)).confidence == 0.0  # 3 falhas: desiste deste lote
    assert vision.detect(page(3)).confidence == 1.0  # mas continua nas próximas
    assert len(sleeps) == 3 and vision.skipped_pages == 0


def test_bad_answer_and_service_down(tmp_path):
    assert detector(FakeVision(["isso não é json"]), tmp_path / "a").detect(page(1)).confidence == 0.0
    down = FakeVision([ServiceUnavailableError("connection refused")])
    vision = detector(down, tmp_path / "b")
    vision.detect(page(1))
    vision.detect(page(2))
    assert len(down.calls) == 1


class Clock:
    def __init__(self):
        self.now = 1_000.0

    def __call__(self):
        return self.now


def test_rotation_moves_to_next_model_and_remembers_until_quota_resets(tmp_path):
    clock = Clock()
    ledger = QuotaLedger(tmp_path / "quota.json", clock)
    best = FakeVision([QuotaExhaustedError("cota", retry_after=3600)], model="melhor")
    backup = FakeVision([batch_answer(WHOLE), batch_answer(WHOLE)], model="reserva")
    rotation = GeminiModelRotation([best, backup], ledger)

    rotation.generate_json([b"img"], "prompt")
    rotation.generate_json([b"img"], "prompt")
    assert best.calls == [1] and backup.calls == [1, 1]  # o bloqueado não é chamado de novo

    clock.now += 3601
    assert QuotaLedger(tmp_path / "quota.json", clock).available("melhor")  # vale entre execuções


def test_rotation_reports_when_every_model_is_exhausted(tmp_path):
    ledger = QuotaLedger(tmp_path / "quota.json", Clock())
    only = FakeVision([QuotaExhaustedError("cota", retry_after=60)], model="unico")
    rotation = GeminiModelRotation([only], ledger)
    vision = detector(rotation, tmp_path)
    assert vision.detect(page(1)).confidence == 0.0
    assert vision.skipped_pages == 1 and vision.resume_hint() is not None


class CountingDetector(PanelDetector):
    def __init__(self, name, confident_pages):
        self.name = name
        self._confident = confident_pages
        self.prefetched: list[int] = []

    def prefetch(self, pages):
        self.prefetched = [p.number for p in pages]

    def detect(self, p):
        confidence = 1.0 if p.number in self._confident else 0.2
        return PanelDetection((BoundingBox(0, 0, 10, 10),), confidence, self.name)


def test_chain_sends_only_unsure_pages_to_the_ai():
    local = CountingDetector("local", confident_pages={1, 3})
    cloud = CountingDetector("cloud", confident_pages={2, 4})
    FallbackPanelDetector([local, cloud]).prefetch([page(n) for n in (1, 2, 3, 4)])
    assert cloud.prefetched == [2, 4]
