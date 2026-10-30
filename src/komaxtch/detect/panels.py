"""Detecção local de quadros com OpenCV, no estilo do Kumiko: contornos das áreas com tinta
separadas pelas calhas brancas, corte de quadros grudados nas calhas internas e uma nota de
confiança para acionar a IA só nas páginas difíceis."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from komaxtch.detect.base import NO_CONFIDENCE, PanelDetection, PanelDetector
from komaxtch.geometry import BoundingBox
from komaxtch.models import Page

INK_THRESHOLD = 220
CLOSING_KERNEL_PX = 3
MIN_PANEL_SIDE_RATIO = 0.08
MIN_PANEL_AREA_RATIO = 0.015
CONTAINED_RATIO = 0.9
SPLASH_AREA_RATIO = 0.75
SPLASH_CONFIDENCE = 0.5  # bloco enorme: splash ou vários quadros sem calha branca
MAX_PANELS = 16
CROWDED_CONFIDENCE = 0.3
MAX_INK_OUTSIDE_RATIO = 0.2  # tinta fora dos quadros indica arte sem borda
MAX_SPLIT_DEPTH = 5
MERGED_CONFIDENCE = 0.4  # sobrou calha que não deu para cortar (ex.: balão cobrindo a borda)

INNER_BORDER_LENGTH_RATIO = 0.3  # borda interna cobre ao menos 30% do lado da caixa
INNER_BORDER_EDGE_MARGIN_RATIO = 0.05  # ignora as bordas da própria caixa
MIN_BORDER_THICKNESS_PX = 3  # traços mais finos são desenho, não borda de quadro
MAX_BORDER_THICKNESS_PX = 14  # mais grosso que isso é área preta, não borda
MIN_GUTTER_PX = 4
MAX_GUTTER_RATIO = 0.06  # calha mais larga que 6% da altura da página não é calha
MAX_GUTTER_INK_RATIO = 0.25  # a calha é quase toda branca (no máximo contornos de balão)
MIN_CUT_BORDER_COVERAGE = 0.3  # bordas dos dois lados da calha cobrem parte boa da caixa
CROSSING_INK_RATIO = 0.95  # coluna quase toda preta através da calha = borda que a cruza
MIN_WHITE_BAND_RATIO = 0.008  # faixa branca sem moldura (arte sangrada): ~1% da altura da página


@dataclass(frozen=True)
class _Gutter:
    """Faixa branca entre duas bordas, em coordenadas relativas à caixa."""

    horizontal: bool
    start: int
    end: int


class OpenCvPanelDetector(PanelDetector):
    name = "opencv"

    def __init__(self) -> None:
        self._results: dict[str, PanelDetection] = {}

    def detect(self, page: Page) -> PanelDetection:
        if page.fingerprint not in self._results:
            self._results[page.fingerprint] = self._analyse(page)
        return self._results[page.fingerprint]

    def refine(self, page: Page, boxes: tuple[BoundingBox, ...]) -> tuple[BoundingBox, ...]:
        """Corta nas calhas internas as caixas grosseiras vindas de outro detector (ex.: uma
        faixa inteira devolvida pela IA que ainda contém vários quadros lado a lado)."""
        ink = np.asarray(page.image) < INK_THRESHOLD
        refined = []
        for box in boxes:
            tight = _ink_bounds(box, ink)
            refined.extend(self._split_merged(tight, ink, MAX_SPLIT_DEPTH) if tight else [box])
        return tuple(refined)

    def _analyse(self, page: Page) -> PanelDetection:
        ink = np.asarray(page.image) < INK_THRESHOLD
        candidates = self._remove_contained(self._candidate_boxes(ink))
        boxes = [part for box in candidates for part in self._split_merged(box, ink, MAX_SPLIT_DEPTH)]
        return PanelDetection(tuple(boxes), self._confidence(boxes, ink), self.name)

    @staticmethod
    def _candidate_boxes(ink: np.ndarray) -> list[BoundingBox]:
        height, width = ink.shape
        kernel = np.ones((CLOSING_KERNEL_PX, CLOSING_KERNEL_PX), np.uint8)
        mask = cv2.morphologyEx(ink.astype(np.uint8) * 255, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            box = BoundingBox(x, y, x + w, y + h)
            if _big_enough(box, width, height):
                boxes.append(box)
        return boxes

    @staticmethod
    def _remove_contained(boxes: list[BoundingBox]) -> list[BoundingBox]:
        by_size = sorted(boxes, key=lambda b: b.area, reverse=True)
        kept: list[BoundingBox] = []
        for box in by_size:
            if not any(box.overlap_ratio(bigger) >= CONTAINED_RATIO for bigger in kept):
                kept.append(box)
        return kept

    def _split_merged(self, box: BoundingBox, ink: np.ndarray, depth: int) -> list[BoundingBox]:
        """Quadros grudados (ex.: balão atravessando a calha) são cortados na calha interna."""
        gutter = _find_inner_gutter(box, ink) if depth else None
        if gutter is None:
            return [box]
        parts = []
        for side in _sides_of(box, gutter):
            tight = _ink_bounds(side, ink)
            if tight is not None and _big_enough(tight, ink.shape[1], ink.shape[0]):
                parts.extend(self._split_merged(tight, ink, depth - 1))
        return parts or [box]

    @staticmethod
    def _confidence(boxes: list[BoundingBox], ink: np.ndarray) -> float:
        if not boxes:
            return NO_CONFIDENCE
        if len(boxes) > MAX_PANELS:
            return CROWDED_CONFIDENCE
        covered = np.zeros_like(ink)
        for box in boxes:
            covered[box.top : box.bottom, box.left : box.right] = True
        total_ink = int(ink.sum()) or 1
        ink_outside = int((ink & ~covered).sum()) / total_ink
        confidence = max(NO_CONFIDENCE, 1.0 - ink_outside / MAX_INK_OUTSIDE_RATIO)
        page_area = ink.shape[0] * ink.shape[1]
        if any(box.area >= SPLASH_AREA_RATIO * page_area for box in boxes):
            confidence = min(confidence, SPLASH_CONFIDENCE)
        if any(_is_unsplit_merge(box, ink) for box in boxes):
            confidence = min(confidence, MERGED_CONFIDENCE)
        return confidence


def _big_enough(box: BoundingBox, page_width: int, page_height: int) -> bool:
    return (
        box.width >= MIN_PANEL_SIDE_RATIO * page_width
        and box.height >= MIN_PANEL_SIDE_RATIO * page_height
        and box.area >= MIN_PANEL_AREA_RATIO * page_width * page_height
    )


def _is_unsplit_merge(box: BoundingBox, ink: np.ndarray) -> bool:
    """Sobrou uma calha parcial separando dois pedaços com tamanho de quadro?"""
    gutter = _find_inner_gutter(box, ink, min_border_coverage=0.0, clean_cut=False)
    if gutter is None:
        return False
    page_height, page_width = ink.shape
    sides = (_ink_bounds(side, ink) for side in _sides_of(box, gutter))
    return all(side is not None and _big_enough(side, page_width, page_height) for side in sides)


def _sides_of(box: BoundingBox, gutter: _Gutter) -> tuple[BoundingBox, BoundingBox]:
    if gutter.horizontal:
        return (
            BoundingBox(box.left, box.top, box.right, box.top + gutter.start),
            BoundingBox(box.left, box.top + gutter.end, box.right, box.bottom),
        )
    return (
        BoundingBox(box.left, box.top, box.left + gutter.start, box.bottom),
        BoundingBox(box.left + gutter.end, box.top, box.right, box.bottom),
    )


def _ink_bounds(box: BoundingBox, ink: np.ndarray) -> BoundingBox | None:
    region = ink[box.top : box.bottom, box.left : box.right]
    rows, cols = np.flatnonzero(region.any(axis=1)), np.flatnonzero(region.any(axis=0))
    if rows.size == 0:
        return None
    return BoundingBox(
        box.left + int(cols[0]), box.top + int(rows[0]), box.left + int(cols[-1]) + 1, box.top + int(rows[-1]) + 1
    )


def _find_inner_gutter(
    box: BoundingBox,
    ink: np.ndarray,
    min_border_coverage: float = MIN_CUT_BORDER_COVERAGE,
    clean_cut: bool = True,
) -> _Gutter | None:
    """Uma calha dentro da caixa indica que ela juntou vários quadros. A calha é uma faixa branca
    entre duas bordas paralelas ou, na arte sangrada sem moldura, uma faixa branca larga.
    Com `clean_cut`, só vale a calha que dá para cortar de ponta a ponta."""
    region = ink[box.top : box.bottom, box.left : box.right].astype(np.uint8)
    page_height = ink.shape[0]
    for horizontal in (True, False):
        oriented = region if horizontal else region.T
        length = box.width if horizontal else box.height
        kernel = (max(1, int(length * INNER_BORDER_LENGTH_RATIO)), 1) if horizontal else (1, max(1, int(length * INNER_BORDER_LENGTH_RATIO)))
        lines = _border_lines(region, kernel, across_rows=horizontal)
        lines = lines if horizontal else lines.T
        gap = _gap_between_borders(lines, oriented, page_height, min_border_coverage, clean_cut)
        if gap is None and clean_cut:
            gap = _white_band(oriented, page_height)
        if gap is not None:
            return _Gutter(horizontal, *gap)
    return None


def _border_lines(region: np.ndarray, length_kernel: tuple[int, int], across_rows: bool) -> np.ndarray:
    """Linhas longas com espessura de borda: traços finos do desenho e áreas pretas sólidas ficam de fora."""

    def across(thickness: int) -> tuple[int, int]:
        return (1, thickness) if across_rows else (thickness, 1)

    too_thick = _open(region, across(MAX_BORDER_THICKNESS_PX))
    border_like = _open(region & ~too_thick, across(MIN_BORDER_THICKNESS_PX))
    return _open(border_like, length_kernel)


def _gap_between_borders(
    lines: np.ndarray, region: np.ndarray, page_height: int, min_border_coverage: float, clean_cut: bool
) -> tuple[int, int] | None:
    """Procura, ao longo do eixo 0, uma faixa branca entre duas bordas paralelas. Com `clean_cut`,
    nenhuma borda de outro quadro pode atravessar a faixa."""
    max_gap = max(MIN_GUTTER_PX, int(page_height * MAX_GUTTER_RATIO))
    has_border = lines.any(axis=1)
    ink_profile = region.mean(axis=1)
    margin = int(len(has_border) * INNER_BORDER_EDGE_MARGIN_RATIO)
    runs = [(start, end) for start, end in _runs(has_border) if margin <= start and end <= len(has_border) - margin]
    for (before_start, gap_start), (gap_end, after_end) in zip(runs, runs[1:]):
        width = gap_end - gap_start
        is_gutter = (
            MIN_GUTTER_PX <= width <= max_gap
            and ink_profile[gap_start:gap_end].mean() <= MAX_GUTTER_INK_RATIO
            and _coverage(lines[before_start:gap_start]) >= min_border_coverage
            and _coverage(lines[gap_end:after_end]) >= min_border_coverage
            and not (clean_cut and _crossed_by_border(region[gap_start:gap_end]))
        )
        if is_gutter:
            return int(gap_start), int(gap_end)
    return None


def _white_band(region: np.ndarray, page_height: int) -> tuple[int, int] | None:
    """Faixa quase toda branca atravessando a caixa, sem borda cruzando: calha de arte sangrada."""
    white = region.mean(axis=1) <= MAX_GUTTER_INK_RATIO
    margin = int(len(white) * INNER_BORDER_EDGE_MARGIN_RATIO)
    min_width = max(MIN_GUTTER_PX, int(page_height * MIN_WHITE_BAND_RATIO))
    for start, end in _runs(white):
        inside = margin <= start and end <= len(white) - margin
        if inside and end - start >= min_width and not _crossed_by_border(region[start:end]):
            return int(start), int(end)
    return None


def _crossed_by_border(band: np.ndarray) -> bool:
    """Uma borda reta atravessando a faixa aparece como colunas pretas de ponta a ponta."""
    crossing = band.mean(axis=0) >= CROSSING_INK_RATIO
    return any(end - start >= MIN_BORDER_THICKNESS_PX for start, end in _runs(crossing))


def _coverage(border_rows: np.ndarray) -> float:
    """Fração do comprimento da caixa ocupada por uma borda."""
    return float(border_rows.any(axis=0).mean())


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Trechos contíguos verdadeiros como pares (início, fim exclusivo)."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def _open(region: np.ndarray, kernel_size: tuple[int, int]) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    return cv2.morphologyEx(region, cv2.MORPH_OPEN, kernel)
