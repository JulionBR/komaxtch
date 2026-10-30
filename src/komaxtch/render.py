"""Otimização de uma tela para e-ink de 4 tons."""

from __future__ import annotations

from enum import Enum

import numpy as np
from PIL import Image, ImageFilter, ImageOps

# Valor de cinza (0-255) de cada nível, do branco (0) ao preto (3).
GRAY_LEVELS = (255, 170, 85, 0)
PALETTE_SIZE = 256
AUTOCONTRAST_CUTOFF_PERCENT = 1
SHARPEN = ImageFilter.UnsharpMask(radius=1.2, percent=80, threshold=2)


class Dithering(Enum):
    FLOYD = "floyd"
    NONE = "none"

    @property
    def pillow_mode(self) -> Image.Dither:
        return Image.Dither.FLOYDSTEINBERG if self is Dithering.FLOYD else Image.Dither.NONE


class EInkRenderer:
    """Transforma uma imagem já no tamanho da tela em níveis de cinza 0..3."""

    def __init__(self, dithering: Dithering = Dithering.FLOYD) -> None:
        self._dithering = dithering
        self._palette = self._build_palette()

    def render(self, screen: Image.Image) -> np.ndarray:
        enhanced = self._enhance(screen)
        quantized = enhanced.convert("RGB").quantize(
            palette=self._palette, dither=self._dithering.pillow_mode
        )
        # Entradas além do 4º nível da paleta são preto; o clip as reúne no nível 3.
        return np.minimum(np.asarray(quantized, dtype=np.uint8), len(GRAY_LEVELS) - 1)

    @staticmethod
    def to_image(levels: np.ndarray) -> Image.Image:
        """Simula no computador como a tela aparece no aparelho."""
        lookup = np.array(GRAY_LEVELS, dtype=np.uint8)
        return Image.fromarray(lookup[levels], mode="L")

    @staticmethod
    def _enhance(screen: Image.Image) -> Image.Image:
        gray = ImageOps.autocontrast(screen.convert("L"), cutoff=AUTOCONTRAST_CUTOFF_PERCENT)
        return gray.filter(SHARPEN)

    @staticmethod
    def _build_palette() -> Image.Image:
        palette = Image.new("P", (1, 1))
        colors = [channel for gray in GRAY_LEVELS for channel in (gray, gray, gray)]
        padding = [0] * (PALETTE_SIZE * 3 - len(colors))
        palette.putpalette(colors + padding)
        return palette
