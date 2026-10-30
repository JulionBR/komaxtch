"""Codificação de uma tela em 4 tons para o bitmap XTH (2 bit-planes, colunas da direita para a esquerda)."""

from __future__ import annotations

import numpy as np

from komaxtch.render import GRAY_LEVELS

BITS_PER_BYTE = 8

# O firmware usa 0=branco, 1=cinza escuro, 2=cinza claro, 3=preto.
# Nossos níveis são lineares (0=branco ... 3=preto), então 1 e 2 trocam de lugar.
_LEVEL_TO_CODE = np.array([0, 2, 1, 3], dtype=np.uint8)
_CODE_TO_LEVEL = np.argsort(_LEVEL_TO_CODE).astype(np.uint8)


class XthCodec:
    """Converte matrizes de níveis (altura × largura, valores 0..3) em bytes XTH e vice-versa."""

    @staticmethod
    def bitmap_size(width: int, height: int) -> int:
        return width * column_bytes(height) * 2

    def encode(self, levels: np.ndarray) -> bytes:
        self._validate(levels)
        codes = _LEVEL_TO_CODE[levels]
        columns = codes[:, ::-1].T  # linha i = coluna (largura - 1 - i)
        high_plane = self._pack((columns >> 1) & 1)
        low_plane = self._pack(columns & 1)
        return high_plane + low_plane

    def decode(self, data: bytes, width: int, height: int) -> np.ndarray:
        plane_size = width * column_bytes(height)
        if len(data) != plane_size * 2:
            raise ValueError(f"Bitmap XTH com {len(data)} bytes; esperado {plane_size * 2}")
        high = self._unpack(data[:plane_size], width, height)
        low = self._unpack(data[plane_size:], width, height)
        codes = ((high << 1) | low).T[:, ::-1]
        return _CODE_TO_LEVEL[codes]

    @staticmethod
    def _pack(bits: np.ndarray) -> bytes:
        return np.packbits(bits.astype(np.uint8), axis=1).tobytes()

    @staticmethod
    def _unpack(plane: bytes, width: int, height: int) -> np.ndarray:
        packed = np.frombuffer(plane, dtype=np.uint8).reshape(width, column_bytes(height))
        return np.unpackbits(packed, axis=1)[:, :height]

    @staticmethod
    def _validate(levels: np.ndarray) -> None:
        if levels.ndim != 2:
            raise ValueError("A tela deve ser uma matriz 2D de níveis de cinza")
        if levels.size and levels.max() >= len(GRAY_LEVELS):
            raise ValueError("Níveis de cinza devem estar entre 0 e 3")


def column_bytes(height: int) -> int:
    return (height + BITS_PER_BYTE - 1) // BITS_PER_BYTE
