"""Leitura e validação de arquivos XTCH com as mesmas checagens do firmware."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from komaxtch.xtch import format as fmt
from komaxtch.xtch.codec import XthCodec

METADATA_TEXT_FIELDS = (fmt.TITLE_SIZE, fmt.AUTHOR_SIZE)


class XtchFormatError(ValueError):
    pass


@dataclass(frozen=True)
class PageInfo:
    offset: int
    size: int
    width: int
    height: int


@dataclass(frozen=True)
class XtchBook:
    title: str
    author: str
    version: tuple[int, int]
    pages: list[PageInfo]


class XtchReader:
    def __init__(self, path: Path, codec: XthCodec | None = None) -> None:
        self._data = path.read_bytes()
        self._codec = codec or XthCodec()
        self.book = self._parse()

    def page_levels(self, index: int) -> np.ndarray:
        page = self.book.pages[index]
        header = self._data[page.offset : page.offset + fmt.PAGE_HEADER.size]
        magic, width, height, _, _, data_size, _ = fmt.PAGE_HEADER.unpack(header)
        if magic != fmt.XTH_MAGIC:
            raise XtchFormatError(f"Página {index}: magic inválido {magic:#x}")
        expected = XthCodec.bitmap_size(width, height)
        if data_size != expected:
            raise XtchFormatError(f"Página {index}: tamanho {data_size}, esperado {expected}")
        start = page.offset + fmt.PAGE_HEADER.size
        return self._codec.decode(self._data[start : start + data_size], width, height)

    def _parse(self) -> XtchBook:
        if len(self._data) < fmt.HEADER.size:
            raise XtchFormatError("Arquivo menor que o cabeçalho")
        fields = fmt.HEADER.unpack_from(self._data)
        magic, major, minor, page_count = fields[0], fields[1], fields[2], fields[3]
        has_metadata, table_offset = fields[5], fields[10]
        if magic != fmt.XTCH_MAGIC:
            raise XtchFormatError(f"Magic inválido {magic:#x}; esperado XTCH")
        if (major, minor) != (fmt.VERSION_MAJOR, fmt.VERSION_MINOR):
            raise XtchFormatError(f"Versão não suportada {major}.{minor}")
        pages = self._page_table(table_offset, page_count)
        title, author = self._title_author() if has_metadata else ("", "")
        return XtchBook(title, author, (major, minor), pages)

    def _page_table(self, table_offset: int, page_count: int) -> list[PageInfo]:
        table_end = table_offset + page_count * fmt.PAGE_TABLE_ENTRY.size
        if table_offset < fmt.HEADER.size or table_end > len(self._data):
            raise XtchFormatError("Índice de páginas fora dos limites do arquivo")
        pages = [
            PageInfo(*fmt.PAGE_TABLE_ENTRY.unpack_from(self._data, table_offset + i * fmt.PAGE_TABLE_ENTRY.size))
            for i in range(page_count)
        ]
        for index, page in enumerate(pages):
            if page.offset + page.size > len(self._data):
                raise XtchFormatError(f"Página {index} ultrapassa o fim do arquivo")
        return pages

    def _title_author(self) -> tuple[str, str]:
        # O firmware lê o título fixo em 0x38 e o autor em 0xB8.
        title_offset = fmt.HEADER.size
        author_offset = title_offset + fmt.TITLE_SIZE
        return (
            _read_c_string(self._data, title_offset, fmt.TITLE_SIZE),
            _read_c_string(self._data, author_offset, fmt.AUTHOR_SIZE),
        )


def _read_c_string(data: bytes, offset: int, size: int) -> str:
    return data[offset : offset + size].split(b"\0", 1)[0].decode("utf-8", errors="replace")
