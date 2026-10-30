"""Gravação de arquivos XTCH em fluxo: as páginas vão para um arquivo temporário até o fechamento."""

from __future__ import annotations

import hashlib
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType

import numpy as np

from komaxtch.xtch import format as fmt
from komaxtch.xtch.codec import XthCodec

MD5_PREFIX_BYTES = 8
COLOR_MODE_DEFAULT = 0
NO_COMPRESSION = 0


@dataclass(frozen=True)
class BookMetadata:
    title: str = ""
    author: str = ""
    publisher: str = ""
    language: str = ""


@dataclass(frozen=True)
class _PageEntry:
    relative_offset: int
    size: int
    width: int
    height: int


class XtchWriter:
    """Uso: `with XtchWriter(caminho, metadados) as writer: writer.add_page(níveis)`."""

    def __init__(self, path: Path, metadata: BookMetadata, codec: XthCodec | None = None) -> None:
        self._path = path
        self._metadata = metadata
        self._codec = codec or XthCodec()
        self._pages: list[_PageEntry] = []
        self._data = tempfile.TemporaryFile()
        self._data_size = 0

    @property
    def page_count(self) -> int:
        return len(self._pages)

    def add_page(self, levels: np.ndarray) -> None:
        height, width = levels.shape
        page = self._encode_page(levels, width, height)
        self._data.write(page)
        self._pages.append(_PageEntry(self._data_size, len(page), width, height))
        self._data_size += len(page)

    def close(self) -> None:
        if self._data.closed:
            return
        try:
            if not self._pages:
                raise ValueError("Nenhuma página para gravar")
            self._write_file()
        finally:
            self._data.close()

    def __enter__(self) -> XtchWriter:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if exc_type is None:
            self.close()
        else:
            self._data.close()

    def _encode_page(self, levels: np.ndarray, width: int, height: int) -> bytes:
        bitmap = self._codec.encode(levels)
        md5 = int.from_bytes(hashlib.md5(bitmap).digest()[:MD5_PREFIX_BYTES], "little")
        header = fmt.PAGE_HEADER.pack(
            fmt.XTH_MAGIC, width, height, COLOR_MODE_DEFAULT, NO_COMPRESSION, len(bitmap), md5
        )
        return header + bitmap

    def _write_file(self) -> None:
        metadata_offset = fmt.HEADER.size
        table_offset = metadata_offset + fmt.METADATA_SIZE
        data_offset = table_offset + fmt.PAGE_TABLE_ENTRY.size * len(self._pages)

        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("wb") as output:
            output.write(self._header(metadata_offset, table_offset, data_offset))
            output.write(self._metadata_block())
            output.write(self._page_table(data_offset))
            self._data.seek(0)
            shutil.copyfileobj(self._data, output)

    def _header(self, metadata_offset: int, table_offset: int, data_offset: int) -> bytes:
        has_metadata, has_thumbnails, has_chapters = 1, 0, 0
        current_page, thumb_offset, chapter_offset, padding = 0, 0, 0, 0
        return fmt.HEADER.pack(
            fmt.XTCH_MAGIC,
            fmt.VERSION_MAJOR,
            fmt.VERSION_MINOR,
            len(self._pages),
            fmt.READ_LEFT_TO_RIGHT,  # as telas já saem na ordem de leitura
            has_metadata,
            has_thumbnails,
            has_chapters,
            current_page,
            metadata_offset,
            table_offset,
            data_offset,
            thumb_offset,
            chapter_offset,
            padding,
        )

    def _metadata_block(self) -> bytes:
        meta = self._metadata
        tail = fmt.METADATA_TAIL.pack(int(time.time()), fmt.NO_COVER_PAGE, 0)
        return (
            _fixed_utf8(meta.title, fmt.TITLE_SIZE)
            + _fixed_utf8(meta.author, fmt.AUTHOR_SIZE)
            + _fixed_utf8(meta.publisher, fmt.PUBLISHER_SIZE)
            + _fixed_utf8(meta.language, fmt.LANGUAGE_SIZE)
            + tail
        )

    def _page_table(self, data_offset: int) -> bytes:
        return b"".join(
            fmt.PAGE_TABLE_ENTRY.pack(
                data_offset + page.relative_offset, page.size, page.width, page.height
            )
            for page in self._pages
        )


def _fixed_utf8(text: str, size: int) -> bytes:
    """String UTF-8 terminada em nulo, cortada sem quebrar caracteres multibyte."""
    encoded = text.encode("utf-8")[: size - 1]
    encoded = encoded.decode("utf-8", errors="ignore").encode("utf-8")
    return encoded.ljust(size, b"\0")
