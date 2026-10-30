from conftest import LTR_ORDER, MANGA_PANELS, RTL_ORDER, draw_manga_page

from komaxtch.detect.base import PanelDetection, PanelDetector
from komaxtch.detect.chain import FallbackPanelDetector
from komaxtch.detect.panels import OpenCvPanelDetector
from komaxtch.geometry import BoundingBox
from komaxtch.models import Page, ReadingDirection
from komaxtch.reading_order import ReadingOrderSorter

from PIL import Image, ImageDraw


def names_in_order(boxes, reference=MANGA_PANELS):
    return [max(reference, key=lambda name: box.iou(reference[name])) for box in boxes]


def test_opencv_finds_every_panel(manga_page):
    detection = OpenCvPanelDetector().detect(manga_page)
    assert len(detection.panels) == len(MANGA_PANELS)
    for box in detection.panels:
        assert max(box.iou(expected) for expected in MANGA_PANELS.values()) > 0.9
    assert detection.confidence >= 0.9


def test_borderless_art_has_low_confidence():
    image = Image.new("L", (800, 1200), 255)
    image.paste(0, (0, 0, 800, 1200))  # arte sangrada ocupando a página toda
    detection = OpenCvPanelDetector().detect(Page(1, image))
    assert detection.confidence <= 0.5


def test_reading_order_rtl_and_ltr():
    boxes = list(MANGA_PANELS.values())[::-1]
    rtl = ReadingOrderSorter(ReadingDirection.RIGHT_TO_LEFT).sort(boxes)
    ltr = ReadingOrderSorter(ReadingDirection.LEFT_TO_RIGHT).sort(boxes)
    assert names_in_order(rtl) == RTL_ORDER
    assert names_in_order(ltr) == LTR_ORDER


class FixedDetector(PanelDetector):
    def __init__(self, name, confidence, boxes=(BoundingBox(0, 0, 10, 10),)):
        self.name = name
        self._result = PanelDetection(tuple(boxes), confidence, name)
        self.calls = 0

    def detect(self, page):
        self.calls += 1
        return self._result


def test_fallback_chain_uses_cloud_only_when_local_is_unsure(manga_page):
    confident, unsure, cloud = FixedDetector("local", 0.9), FixedDetector("local", 0.2), FixedDetector("cloud", 1.0)
    assert FallbackPanelDetector([confident, cloud]).detect(manga_page).source == "local"
    assert cloud.calls == 0
    assert FallbackPanelDetector([unsure, cloud]).detect(manga_page).source == "cloud"


def test_fallback_chain_ends_with_whole_page(manga_page):
    detection = FallbackPanelDetector([FixedDetector("local", 0.1)]).detect(manga_page)
    assert detection.source == "whole-page" and detection.panels == (manga_page.bounds,)


def test_bubble_across_gutter_is_split_at_the_gutter():
    image = draw_manga_page()
    ImageDraw.Draw(image).ellipse((300, 280, 520, 420), fill=255, outline=0, width=3)
    detection = OpenCvPanelDetector().detect(Page(1, image))
    assert len(detection.panels) == len(MANGA_PANELS)
    for box in detection.panels:
        assert max(box.iou(expected) for expected in MANGA_PANELS.values()) > 0.85
    assert detection.confidence >= 0.6  # resolvido sem IA


def test_fallback_chain_keeps_reasonable_guess_when_ai_is_off(manga_page):
    merged = FixedDetector("local", 0.4)
    assert FallbackPanelDetector([merged]).detect(manga_page).source == "local"


def test_coarse_ai_boxes_are_refined_at_the_gutters(manga_page):
    left_column = MANGA_PANELS["left_upper"].union(MANGA_PANELS["left_lower"])
    coarse = FixedDetector("cloud", 1.0, boxes=(left_column,))
    detection = FallbackPanelDetector([FixedDetector("local", 0.1), coarse], refiner=OpenCvPanelDetector()).detect(manga_page)
    assert detection.source == "cloud"
    assert sorted(names_in_order(detection.panels)) == ["left_lower", "left_upper"]
