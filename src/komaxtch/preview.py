"""Prévia em PNG das telas (simulando os 4 tons) e sobreposição de depuração das detecções."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from PIL import Image, ImageDraw

from komaxtch.ingest import SourceFactory
from komaxtch.layout import LayoutMode
from komaxtch.models import Page
from komaxtch.pipeline import ConversionPipeline, ConversionSettings, LayoutFactory

THUMBNAIL_SCALE = 0.5
SHEET_GAP_PX = 12
LABEL_HEIGHT_PX = 24
SHEET_BACKGROUND = 200
PANEL_COLOR = (220, 30, 30)
BUBBLE_COLOR = (30, 90, 220)
OUTLINE_WIDTH = 4


def parse_page_selection(selection: str) -> set[int]:
    """'1-3,7' vira {1, 2, 3, 7}."""
    pages: set[int] = set()
    for part in filter(None, (p.strip() for p in selection.split(","))):
        start, _, end = part.partition("-")
        pages.update(range(int(start), int(end or start) + 1))
    return pages


class PreviewBuilder:
    """Gera, para as páginas escolhidas, as telas de cada modo lado a lado."""

    def __init__(self, settings: ConversionSettings, sources: SourceFactory | None = None) -> None:
        self._settings = settings
        self._sources = sources or SourceFactory()

    def build(self, source: Path, selection: set[int], modes: list[LayoutMode], output: Path) -> list[Path]:
        book = self._sources.create(source).read()
        sheets: dict[int, list[tuple[str, list[Image.Image]]]] = {}
        for mode in modes:
            pipeline = ConversionPipeline(replace(self._settings, mode=mode), LayoutFactory(replace(self._settings, mode=mode)))
            _, pages, layout = pipeline.prepare(book)
            for page in (p for p in pages if p.number in selection):
                screens = [image for _, image in pipeline.screens(layout, page)]
                self._save_screens(output / mode.value, page, screens)
                sheets.setdefault(page.number, []).append((mode.value, screens))
        return [self._save_sheet(output, number, rows) for number, rows in sorted(sheets.items())]

    @staticmethod
    def _save_screens(folder: Path, page: Page, screens: list[Image.Image]) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        for index, screen in enumerate(screens, start=1):
            screen.save(folder / f"p{page.number:03d}_{index:02d}.png")

    @staticmethod
    def _save_sheet(output: Path, page_number: int, rows: list[tuple[str, list[Image.Image]]]) -> Path:
        thumbs = [
            (label, [s.resize((round(s.width * THUMBNAIL_SCALE), round(s.height * THUMBNAIL_SCALE))) for s in screens])
            for label, screens in rows
        ]
        thumb_w, thumb_h = thumbs[0][1][0].size
        columns = max(len(screens) for _, screens in thumbs)
        width = SHEET_GAP_PX + columns * (thumb_w + SHEET_GAP_PX)
        row_height = LABEL_HEIGHT_PX + thumb_h + SHEET_GAP_PX
        sheet = Image.new("L", (width, SHEET_GAP_PX + len(thumbs) * row_height), SHEET_BACKGROUND)
        draw = ImageDraw.Draw(sheet)
        for row, (label, screens) in enumerate(thumbs):
            y = SHEET_GAP_PX + row * row_height
            draw.text((SHEET_GAP_PX, y), f"{label}: {len(screens)} telas", fill=0)
            for column, thumb in enumerate(screens):
                sheet.paste(thumb, (SHEET_GAP_PX + column * (thumb_w + SHEET_GAP_PX), y + LABEL_HEIGHT_PX))
        path = output / f"comparacao_p{page_number:03d}.png"
        output.mkdir(parents=True, exist_ok=True)
        sheet.save(path)
        return path


class DebugOverlay:
    """Desenha os quadros (numerados na ordem de leitura) e os balões detectados sobre a página."""

    def __init__(self, settings: ConversionSettings, sources: SourceFactory | None = None) -> None:
        self._settings = replace(settings, mode=LayoutMode.BUBBLE)
        self._sources = sources or SourceFactory()

    def build(self, source: Path, selection: set[int], output: Path) -> list[Path]:
        book = self._sources.create(source).read()
        pipeline = ConversionPipeline(self._settings)
        _, pages, layout = pipeline.prepare(book)
        output.mkdir(parents=True, exist_ok=True)
        paths = []
        for page in (p for p in pages if p.number in selection):
            crops = layout.crops(page)
            source_name = layout.last_detection.source if layout.last_detection else "?"
            image = page.image.convert("RGB")
            draw = ImageDraw.Draw(image)
            panels = [c.box for c in crops if c.kind == "panel"]
            for order, box in enumerate(panels, start=1):
                draw.rectangle(box.as_tuple(), outline=PANEL_COLOR, width=OUTLINE_WIDTH)
                draw.text((box.left + 8, box.top + 8), str(order), fill=PANEL_COLOR)
            for crop in (c for c in crops if c.kind == "bubble"):
                draw.rectangle(crop.box.as_tuple(), outline=BUBBLE_COLOR, width=OUTLINE_WIDTH)
            path = output / f"debug_p{page.number:03d}_{source_name}.png"
            image.save(path)
            paths.append(path)
        return paths
