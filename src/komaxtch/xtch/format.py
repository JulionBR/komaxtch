"""Layout binário do XTCH, conforme o parser do firmware CrossPoint (lib/Xtc/Xtc/XtcTypes.h)."""

from __future__ import annotations

import struct

XTCH_MAGIC = 0x48435458  # "XTCH"
XTH_MAGIC = 0x00485458  # "XTH\0"
VERSION_MAJOR = 1
VERSION_MINOR = 0

# <IBBHBBBBIQQQQII : magic, versão (maior, menor), páginas, direção, flags x3,
# página atual, offsets (metadados, índice, dados, miniaturas), offset de capítulos, padding
HEADER = struct.Struct("<IBBHBBBBIQQQQII")
PAGE_TABLE_ENTRY = struct.Struct("<QIHH")  # offset, tamanho, largura, altura
PAGE_HEADER = struct.Struct("<IHHBBIQ")  # magic, largura, altura, cor, compressão, tamanho, md5[8]

METADATA_SIZE = 256
TITLE_SIZE = 128
AUTHOR_SIZE = 64
PUBLISHER_SIZE = 32
LANGUAGE_SIZE = 16
METADATA_TAIL = struct.Struct("<IHH8x")  # criação, capa, nº de capítulos, reservado
NO_COVER_PAGE = 0xFFFF

READ_LEFT_TO_RIGHT = 0

assert HEADER.size == 56
assert PAGE_TABLE_ENTRY.size == 16
assert PAGE_HEADER.size == 22
assert TITLE_SIZE + AUTHOR_SIZE + PUBLISHER_SIZE + LANGUAGE_SIZE + METADATA_TAIL.size == METADATA_SIZE
