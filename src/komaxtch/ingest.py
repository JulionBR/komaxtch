"""Leitura de EPUB, CBZ e pastas de imagens como uma sequência ordenada de páginas."""

from __future__ import annotations

import io
import posixpath
import re
import zipfile
from abc import ABC, abstractmethod
from pathlib import Path
from urllib.parse import unquote
from xml.etree import ElementTree

from PIL import Image

from komaxtch.models import Book, Page, ReadingDirection

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
BOOK_EXTENSIONS = {".epub", ".cbz"}

_IMAGE_REFERENCE = re.compile(
    r"<(?:img|image)\b[^>]*?\b(?:src|xlink:href|href)\s*=\s*[\"']([^\"']+)[\"']", re.IGNORECASE
)
_NATURAL_CHUNK = re.compile(r"(\d+)")
_NS = {
    "container": "urn:oasis:names:tc:opendocument:xmlns:container",
    "opf": "http://www.idpf.org/2007/opf",
    "dc": "http://purl.org/dc/elements/1.1/",
}


def natural_key(name: str) -> list[int | str]:
    """Ordena 'p2' antes de 'p10'."""
    return [int(chunk) if chunk.isdigit() else chunk.lower() for chunk in _NATURAL_CHUNK.split(name)]


def is_image_name(name: str) -> bool:
    return Path(name).suffix.lower() in IMAGE_EXTENSIONS


def load_image(data: bytes) -> Image.Image:
    with Image.open(io.BytesIO(data)) as image:
        return image.convert("L")


class PageSource(ABC):
    def __init__(self, path: Path) -> None:
        self.path = path

    @abstractmethod
    def read(self) -> Book: ...

    @staticmethod
    def _book(title: str, author: str, direction: ReadingDirection | None, images: list[bytes]) -> Book:
        pages = [Page(number, load_image(data)) for number, data in enumerate(images, start=1)]
        return Book(title, author, direction, pages)


class EpubSource(PageSource):
    def read(self) -> Book:
        with zipfile.ZipFile(self.path) as archive:
            opf_path = self._opf_path(archive)
            opf = ElementTree.fromstring(archive.read(opf_path))
            names = self._image_names(archive, opf, posixpath.dirname(opf_path))
            images = [archive.read(name) for name in names]
        return self._book(
            self._text(opf, "dc:title") or self.path.stem,
            self._text(opf, "dc:creator"),
            self._direction(opf),
            images,
        )

    @staticmethod
    def _opf_path(archive: zipfile.ZipFile) -> str:
        container = ElementTree.fromstring(archive.read("META-INF/container.xml"))
        rootfile = container.find(".//container:rootfile", _NS)
        if rootfile is None:
            raise ValueError("EPUB sem rootfile em META-INF/container.xml")
        return rootfile.attrib["full-path"]

    def _image_names(self, archive: zipfile.ZipFile, opf: ElementTree.Element, base: str) -> list[str]:
        manifest = {
            item.attrib["id"]: item.attrib
            for item in opf.iterfind("opf:manifest/opf:item", _NS)
        }
        names: list[str] = []
        for itemref in opf.iterfind("opf:spine/opf:itemref", _NS):
            item = manifest.get(itemref.attrib.get("idref", ""))
            if item is None:
                continue
            href = _join(base, item["href"])
            if is_image_name(href):
                names.append(href)
            else:
                names.extend(self._images_in_document(archive, href))
        names = _without_duplicates(name for name in names if name in archive.NameToInfo)
        return names or sorted(
            (name for name in archive.namelist() if is_image_name(name)), key=natural_key
        )

    @staticmethod
    def _images_in_document(archive: zipfile.ZipFile, href: str) -> list[str]:
        try:
            markup = archive.read(href).decode("utf-8", errors="ignore")
        except KeyError:
            return []
        base = posixpath.dirname(href)
        return [_join(base, src) for src in _IMAGE_REFERENCE.findall(markup) if is_image_name(src)]

    @staticmethod
    def _text(opf: ElementTree.Element, tag: str) -> str:
        element = opf.find(f"opf:metadata/{tag}", _NS)
        return (element.text or "").strip() if element is not None else ""

    @staticmethod
    def _direction(opf: ElementTree.Element) -> ReadingDirection | None:
        spine = opf.find("opf:spine", _NS)
        if spine is not None and spine.attrib.get("page-progression-direction") == "rtl":
            return ReadingDirection.RIGHT_TO_LEFT
        for meta in opf.iterfind("opf:metadata/opf:meta", _NS):
            if meta.attrib.get("name") == "primary-writing-mode":
                rtl = meta.attrib.get("content", "").endswith("-rl")
                return ReadingDirection.RIGHT_TO_LEFT if rtl else ReadingDirection.LEFT_TO_RIGHT
        if spine is not None and spine.attrib.get("page-progression-direction") == "ltr":
            return ReadingDirection.LEFT_TO_RIGHT
        return None


class CbzSource(PageSource):
    COMIC_INFO = "ComicInfo.xml"

    def read(self) -> Book:
        with zipfile.ZipFile(self.path) as archive:
            names = sorted((n for n in archive.namelist() if is_image_name(n)), key=natural_key)
            images = [archive.read(name) for name in names]
            info = self._comic_info(archive)
        title = info.findtext("Series") or info.findtext("Title") or self.path.stem
        return self._book(title, info.findtext("Writer") or "", self._direction(info), images)

    def _comic_info(self, archive: zipfile.ZipFile) -> ElementTree.Element:
        name = next((n for n in archive.namelist() if n.endswith(self.COMIC_INFO)), None)
        return ElementTree.fromstring(archive.read(name)) if name else ElementTree.Element("ComicInfo")

    @staticmethod
    def _direction(info: ElementTree.Element) -> ReadingDirection | None:
        manga = info.findtext("Manga")
        if manga == "YesAndRightToLeft":
            return ReadingDirection.RIGHT_TO_LEFT
        return ReadingDirection.LEFT_TO_RIGHT if manga == "No" else None


class FolderSource(PageSource):
    def read(self) -> Book:
        files = sorted(
            (f for f in self.path.iterdir() if f.is_file() and is_image_name(f.name)),
            key=lambda f: natural_key(f.name),
        )
        return self._book(self.path.name, "", None, [f.read_bytes() for f in files])


class SourceFactory:
    """Escolhe o leitor certo para cada caminho e expande pastas de livros."""

    _BY_EXTENSION: dict[str, type[PageSource]] = {".epub": EpubSource, ".cbz": CbzSource}

    def create(self, path: Path) -> PageSource:
        if path.is_dir():
            return FolderSource(path)
        source = self._BY_EXTENSION.get(path.suffix.lower())
        if source is None:
            raise ValueError(f"Formato não suportado: {path.name} (use EPUB, CBZ ou pasta de imagens)")
        return source(path)

    def expand(self, paths: list[Path]) -> list[Path]:
        """Uma pasta que contém livros vira a lista desses livros; senão, é uma pasta de imagens."""
        expanded: list[Path] = []
        for path in paths:
            books = self._books_in(path) if path.is_dir() else []
            expanded.extend(books or [path])
        return expanded

    @staticmethod
    def _books_in(folder: Path) -> list[Path]:
        return sorted(
            (f for f in folder.iterdir() if f.suffix.lower() in BOOK_EXTENSIONS),
            key=lambda f: natural_key(f.name),
        )


def _join(base: str, href: str) -> str:
    return posixpath.normpath(posixpath.join(base, unquote(href.split("#", 1)[0])))


def _without_duplicates(names) -> list[str]:
    return list(dict.fromkeys(names))
