import struct

import numpy as np
import pytest

from komaxtch.xtch import format as fmt
from komaxtch.xtch.codec import XthCodec
from komaxtch.xtch.reader import XtchFormatError, XtchReader
from komaxtch.xtch.writer import BookMetadata, XtchWriter

X3_WIDTH, X3_HEIGHT = 528, 792


def random_levels(width: int, height: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 4, size=(height, width), dtype=np.uint8)


def firmware_pixel(data: bytes, width: int, height: int, x: int, y: int) -> int:
    """Transcrição de getPixelValue() do XtcReaderActivity.cpp do CrossPoint."""
    col_bytes = (height + 7) // 8
    plane_size = width * col_bytes
    offset = (width - 1 - x) * col_bytes + y // 8
    bit = 7 - (y % 8)
    bit1 = (data[offset] >> bit) & 1
    bit2 = (data[plane_size + offset] >> bit) & 1
    return (bit1 << 1) | bit2


@pytest.mark.parametrize("size", [(X3_WIDTH, X3_HEIGHT), (480, 800), (13, 21)])
def test_codec_round_trip(size):
    width, height = size
    levels = random_levels(width, height)
    codec = XthCodec()
    data = codec.encode(levels)
    assert len(data) == XthCodec.bitmap_size(width, height)
    np.testing.assert_array_equal(codec.decode(data, width, height), levels)


def test_codec_matches_firmware_pixel_reader():
    width, height = 13, 21
    levels = random_levels(width, height, seed=3)
    data = XthCodec().encode(levels)
    firmware_code_for_level = {0: 0, 1: 2, 2: 1, 3: 3}  # 1=cinza escuro, 2=cinza claro
    for y in range(height):
        for x in range(width):
            assert firmware_pixel(data, width, height, x, y) == firmware_code_for_level[levels[y, x]]


def test_writer_layout_matches_firmware(tmp_path):
    path = tmp_path / "book.xtch"
    pages = [random_levels(X3_WIDTH, X3_HEIGHT, seed) for seed in range(3)]
    with XtchWriter(path, BookMetadata(title="Naruto 1", author="Kishimoto")) as writer:
        for levels in pages:
            writer.add_page(levels)

    data = path.read_bytes()
    magic, major, minor, count = struct.unpack_from("<IBBH", data)
    assert (magic, major, minor, count) == (fmt.XTCH_MAGIC, 1, 0, 3)
    assert data[0x38:0x40] == b"Naruto 1"  # título lido pelo firmware em 0x38
    assert data[0xB8:0xC1] == b"Kishimoto"  # autor em 0xB8

    reader = XtchReader(path)
    assert reader.book.title == "Naruto 1"
    assert [(p.width, p.height) for p in reader.book.pages] == [(X3_WIDTH, X3_HEIGHT)] * 3
    for index, levels in enumerate(pages):
        np.testing.assert_array_equal(reader.page_levels(index), levels)


def test_long_utf8_title_is_truncated_safely(tmp_path):
    path = tmp_path / "book.xtch"
    with XtchWriter(path, BookMetadata(title="ç" * 200)) as writer:
        writer.add_page(random_levels(8, 8))
    title = XtchReader(path).book.title
    assert set(title) == {"ç"} and len(title.encode()) < fmt.TITLE_SIZE


def test_writer_refuses_empty_book(tmp_path):
    with pytest.raises(ValueError):
        with XtchWriter(tmp_path / "empty.xtch", BookMetadata()):
            pass


def test_reader_rejects_other_formats(tmp_path):
    path = tmp_path / "bad.xtch"
    path.write_bytes(b"XTC\0" + bytes(60))
    with pytest.raises(XtchFormatError):
        XtchReader(path)
