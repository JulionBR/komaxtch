"""Ordem de leitura dos quadros por cortes recursivos nas calhas (XY-cut)."""

from __future__ import annotations

from komaxtch.geometry import BoundingBox
from komaxtch.models import ReadingDirection

GUTTER_TOLERANCE_PX = 8  # quadros podem invadir a calha um pouco sem impedir o corte


class ReadingOrderSorter:
    """Corta a página primeiro em faixas horizontais (de cima para baixo) e depois em
    colunas (da direita para a esquerda no mangá), recursivamente."""

    def __init__(self, direction: ReadingDirection) -> None:
        self._rtl = direction is ReadingDirection.RIGHT_TO_LEFT

    def sort(self, boxes: list[BoundingBox]) -> list[BoundingBox]:
        if len(boxes) <= 1:
            return list(boxes)
        rows = self._split(boxes, horizontal=True)
        if len(rows) > 1:
            return [box for row in rows for box in self.sort(row)]
        columns = self._split(boxes, horizontal=False)
        if len(columns) > 1:
            if self._rtl:
                columns.reverse()
            return [box for column in columns for box in self.sort(column)]
        return self._fallback(boxes)

    @staticmethod
    def _split(boxes: list[BoundingBox], horizontal: bool) -> list[list[BoundingBox]]:
        """Agrupa caixas cujas projeções no eixo se sobrepõem; cada grupo fica de um lado de uma calha."""

        def span(box: BoundingBox) -> tuple[int, int]:
            return (box.top, box.bottom) if horizontal else (box.left, box.right)

        groups: list[list[BoundingBox]] = []
        group_end = None
        for box in sorted(boxes, key=lambda b: span(b)[0]):
            start, end = span(box)
            if group_end is None or start >= group_end - GUTTER_TOLERANCE_PX:
                groups.append([box])
                group_end = end
            else:
                groups[-1].append(box)
                group_end = max(group_end, end)
        return groups

    def _fallback(self, boxes: list[BoundingBox]) -> list[BoundingBox]:
        """Sem calha limpa (quadros entrelaçados): ordena pelo centro."""
        def key(box: BoundingBox) -> tuple[float, float]:
            x, y = box.center
            return y, -x if self._rtl else x

        return sorted(boxes, key=key)
