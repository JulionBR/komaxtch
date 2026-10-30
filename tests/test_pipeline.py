import re
from dataclasses import replace

import numpy as np
from PIL import Image

from komaxtch.cli import Cli
from komaxtch.detect.base import BubbleDetector
from komaxtch.detect.chain import FallbackPanelDetector
from komaxtch.detect.panels import OpenCvPanelDetector
from komaxtch.devices import get_device
from komaxtch.geometry import BoundingBox
from komaxtch.ingest import CbzSource, EpubSource
from komaxtch.layout import BubbleLayout, LayoutMode, PanelLayout, Rotation, ScreenComposer
from komaxtch.models import Crop, Page, ReadingDirection
from komaxtch.normalize import PageNormalizer
from komaxtch.pipeline import ConversionPipeline, ConversionSettings
from komaxtch.reading_order import ReadingOrderSorter
from komaxtch.xtch.reader import XtchReader

X3 = get_device("x3")
SETTINGS = ConversionSettings(device=X3, mode=LayoutMode.PANEL, use_local_ai=False, use_cloud=False)


def test_epub_follows_spine_and_reads_direction(manga_epub):
    book = EpubSource(manga_epub).read()
    assert (book.title, book.author) == ("Meu Mangá", "Autora")
    assert book.direction is ReadingDirection.RIGHT_TO_LEFT
    assert [p.image.size for p in book.pages] == [(800, 1200), (600, 900)]


def test_cbz_uses_natural_order_and_comic_info(comic_cbz):
    book = CbzSource(comic_cbz).read()
    assert [p.image.getpixel((0, 0)) for p in book.pages] == [100, 200, 30]
    assert book.direction is ReadingDirection.LEFT_TO_RIGHT and book.title == "HQ"


def test_spreads_split_in_reading_order():
    spread = Image.new("L", (1600, 1200), 255)
    spread.paste(0, (0, 0, 800, 1200))  # metade esquerda preta
    pages = PageNormalizer(ReadingDirection.RIGHT_TO_LEFT).normalize([Page(1, spread)])
    assert len(pages) == 2
    assert np.asarray(pages[0].image).mean() > 200  # RTL: metade direita (branca) primeiro


def test_wide_panel_is_rotated_and_screen_has_device_size(manga_page):
    composer = ScreenComposer(X3, Rotation.CLOCKWISE)
    assert not ScreenComposer(X3).should_rotate(BoundingBox(0, 0, 800, 300))  # padrão: não gira
    wide = BoundingBox(0, 0, 800, 300)
    tall = BoundingBox(0, 0, 300, 800)
    assert composer.should_rotate(wide) and not composer.should_rotate(tall)
    for box in (wide, tall, manga_page.bounds):
        assert composer.compose(Crop(manga_page, box, "panel")).size == (X3.width, X3.height)


class FakeBubbles(BubbleDetector):
    def __init__(self, boxes):
        self._boxes = boxes

    def detect(self, page):
        return self._boxes


def test_bubble_layout_zooms_only_small_bubbles(manga_page):
    direction = ReadingDirection.RIGHT_TO_LEFT
    panels = PanelLayout(FallbackPanelDetector([OpenCvPanelDetector()]), ReadingOrderSorter(direction))
    small = BoundingBox(70, 70, 130, 110)  # dentro do quadro de cima
    huge = BoundingBox(60, 380, 380, 590)  # ocupa quase o quadro esquerdo de cima
    layout = BubbleLayout(panels, FakeBubbles([small, huge]), ScreenComposer(X3), direction)
    kinds = [crop.kind for crop in layout.crops(manga_page)]
    assert kinds.count("panel") == 5
    assert kinds.count("bubble") == 1
    assert kinds[:2] == ["panel", "bubble"]  # o zoom vem logo depois do seu quadro


def test_auto_layout_zooms_bubbles_only_on_shrunken_panels():
    from komaxtch.detect.base import PanelDetection, PanelDetector

    class WholePage(PanelDetector):
        name = "whole"

        def detect(self, page):
            return PanelDetection((page.bounds,), 1.0, self.name)

    class CountingBubbles(FakeBubbles):
        calls = 0

        def detect(self, page):
            self.calls += 1
            return super().detect(page)

    direction = ReadingDirection.RIGHT_TO_LEFT
    bubble = BoundingBox(100, 100, 220, 200)
    bubbles = CountingBubbles([bubble])
    layout = BubbleLayout(
        PanelLayout(WholePage(), ReadingOrderSorter(direction)), bubbles, ScreenComposer(X3), direction, 0.6
    )
    splash = Page(1, Image.new("L", (1107, 1680), 200))  # página inteira: aparece a 47%
    assert [c.kind for c in layout.crops(splash)] == ["panel", "bubble"]
    small = Page(2, Image.new("L", (500, 700), 200))  # cabe quase sem reduzir
    assert [c.kind for c in layout.crops(small)] == ["panel"]
    assert bubbles.calls == 1  # nem procura balões onde não vai dar zoom


def test_convert_epub_end_to_end(manga_epub, tmp_path):
    output = tmp_path / "out.xtch"
    report = ConversionPipeline(SETTINGS).convert(manga_epub, output)
    reader = XtchReader(output)
    assert report.direction is ReadingDirection.RIGHT_TO_LEFT and report.direction_detected
    assert report.screens == len(reader.book.pages) == 5 + 1  # 5 quadros + página lisa inteira
    assert {(p.width, p.height) for p in reader.book.pages} == {(X3.width, X3.height)}
    assert reader.book.title == "Meu Mangá"


def test_page_mode_gives_one_screen_per_page(comic_cbz, tmp_path):
    output = tmp_path / "hq.xtch"
    report = ConversionPipeline(replace(SETTINGS, mode=LayoutMode.PAGE)).convert(comic_cbz, output)
    assert report.screens == 3 == len(XtchReader(output).book.pages)


def test_cli_convert_and_inspect(manga_epub, tmp_path, capsys):
    assert Cli().run(["convert", str(manga_epub), "--no-cloud"]) == 0
    output = manga_epub.with_suffix(".xtch")
    extract = tmp_path / "telas"
    pdf = tmp_path / "previa.pdf"
    assert Cli().run(["inspect", str(output), "--extract", str(extract), "--pdf", str(pdf)]) == 0
    assert "arquivo válido" in capsys.readouterr().out
    assert len(list(extract.glob("*.png"))) == 6
    assert len(re.findall(rb"/Type\s*/Page\b", pdf.read_bytes())) == 6  # uma página do PDF por tela
