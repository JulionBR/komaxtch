"""Limpeza das caixas de quadros: junta as que se sobrepõem e absorve tiras finas."""

from __future__ import annotations

from komaxtch.geometry import BoundingBox

OVERLAP_MERGE_RATIO = 0.3  # caixa com mais de 30% dentro de outra vira uma só
MIN_SIDE_RATIO = 0.1  # lado menor que 10% da página é tira, não quadro


class PanelCleaner:
    def __init__(self, page_bounds: BoundingBox) -> None:
        self._page = page_bounds

    def clean(self, boxes: list[BoundingBox]) -> list[BoundingBox]:
        return self._absorb_thin(self._merge_overlapping(boxes))

    @staticmethod
    def _merge_overlapping(boxes: list[BoundingBox]) -> list[BoundingBox]:
        merged = list(boxes)
        changed = True
        while changed:
            changed = False
            for i, a in enumerate(merged):
                for j in range(i + 1, len(merged)):
                    b = merged[j]
                    if max(a.overlap_ratio(b), b.overlap_ratio(a)) > OVERLAP_MERGE_RATIO:
                        merged[i] = a.union(b)
                        del merged[j]
                        changed = True
                        break
                if changed:
                    break
        return merged

    def _absorb_thin(self, boxes: list[BoundingBox]) -> list[BoundingBox]:
        thin = [b for b in boxes if self._is_thin(b)]
        solid = [b for b in boxes if not self._is_thin(b)]
        if not solid:
            return boxes
        for strip in thin:
            nearest = min(range(len(solid)), key=lambda i: _distance(strip, solid[i]))
            solid[nearest] = solid[nearest].union(strip)
        return solid

    def _is_thin(self, box: BoundingBox) -> bool:
        return (
            box.width < MIN_SIDE_RATIO * self._page.width
            or box.height < MIN_SIDE_RATIO * self._page.height
        )


def _distance(a: BoundingBox, b: BoundingBox) -> float:
    (ax, ay), (bx, by) = a.center, b.center
    return (ax - bx) ** 2 + (ay - by) ** 2
