import io
import zipfile

import pytest
from PIL import Image, ImageDraw

from komaxtch.geometry import BoundingBox
from komaxtch.models import Page

PAGE_SIZE = (800, 1200)
BORDER_PX = 4

# Layout clássico de mangá: faixa de cima larga, depois um quadro alto à direita
# ao lado de dois empilhados à esquerda, e uma faixa final.
# Ordem RTL esperada: topo, alto (direita), esquerda-cima, esquerda-baixo, rodapé.
MANGA_PANELS = {
    "top": BoundingBox(40, 40, 760, 330),
    "tall_right": BoundingBox(420, 360, 760, 860),
    "left_upper": BoundingBox(40, 360, 390, 600),
    "left_lower": BoundingBox(40, 620, 390, 860),
    "bottom": BoundingBox(40, 890, 760, 1160),
}
RTL_ORDER = ["top", "tall_right", "left_upper", "left_lower", "bottom"]
LTR_ORDER = ["top", "left_upper", "left_lower", "tall_right", "bottom"]


def draw_manga_page(panels=MANGA_PANELS) -> Image.Image:
    image = Image.new("L", PAGE_SIZE, 255)
    draw = ImageDraw.Draw(image)
    for box in panels.values():
        draw.rectangle(box.as_tuple(), outline=0, width=BORDER_PX, fill=235)
        for offset in range(box.left + 20, box.right - 20, 37):  # "arte" interna
            draw.line((offset, box.top + 20, offset + 15, box.bottom - 20), fill=60, width=2)
        bubble = (box.left + 30, box.top + 30, box.left + 130, box.top + 100)
        draw.ellipse(bubble, fill=255, outline=0, width=2)
    return image


@pytest.fixture
def manga_page() -> Page:
    return Page(1, draw_manga_page())


def png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def manga_epub(tmp_path):
    """EPUB de imagens no estilo do KCC, com páginas RTL e a página 2 referenciada via SVG."""
    path = tmp_path / "manga.epub"
    opf = """<?xml version="1.0"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:title>Meu Mangá</dc:title><dc:creator>Autora</dc:creator>
  </metadata>
  <manifest>
    <item id="p1" href="Text/p1.xhtml" media-type="application/xhtml+xml"/>
    <item id="p2" href="Text/p2.xhtml" media-type="application/xhtml+xml"/>
    <item id="i1" href="Images/p1.png" media-type="image/png"/>
    <item id="i2" href="Images/p2.png" media-type="image/png"/>
  </manifest>
  <spine page-progression-direction="rtl"><itemref idref="p1"/><itemref idref="p2"/></spine>
</package>"""
    container = """<?xml version="1.0"?>
<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container" version="1.0">
  <rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/></rootfiles>
</container>"""
    with zipfile.ZipFile(path, "w") as epub:
        epub.writestr("mimetype", "application/epub+zip")
        epub.writestr("META-INF/container.xml", container)
        epub.writestr("OEBPS/content.opf", opf)
        epub.writestr("OEBPS/Text/p1.xhtml", '<html><body><img src="../Images/p1.png"/></body></html>')
        epub.writestr(
            "OEBPS/Text/p2.xhtml",
            '<html><body><svg><image xlink:href="../Images/p2.png"/></svg></body></html>',
        )
        epub.writestr("OEBPS/Images/p1.png", png_bytes(draw_manga_page()))
        epub.writestr("OEBPS/Images/p2.png", png_bytes(Image.new("L", (600, 900), 128)))
    return path


@pytest.fixture
def comic_cbz(tmp_path):
    path = tmp_path / "hq.cbz"
    with zipfile.ZipFile(path, "w") as cbz:
        for name, shade in [("p10.png", 30), ("p2.png", 200), ("p1.png", 100)]:
            cbz.writestr(name, png_bytes(Image.new("L", (600, 900), shade)))
        cbz.writestr("ComicInfo.xml", "<ComicInfo><Series>HQ</Series><Manga>No</Manga></ComicInfo>")
    return path
