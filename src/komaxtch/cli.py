"""Linha de comando: `komaxtch convert livro.epub` e utilitários de prévia, depuração e inspeção."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections.abc import Iterator
from pathlib import Path

from PIL import Image

from dotenv import load_dotenv

from komaxtch.devices import DEFAULT_DEVICE, DEVICES, device_for_size, get_device
from komaxtch.ingest import SourceFactory
from komaxtch.layout import LayoutMode, Rotation
from komaxtch.models import ReadingDirection
from komaxtch.pipeline import (
    GEMINI_KEY_VARIABLE,
    ConversionPipeline,
    ConversionReport,
    ConversionSettings,
    LayoutFactory,
)
from komaxtch.preview import DebugOverlay, PreviewBuilder, parse_page_selection
from komaxtch.render import Dithering, EInkRenderer
from komaxtch.xtch.reader import XtchFormatError, XtchReader

OUTPUT_SUFFIX = ".xtch"
USER_ENV_FILE = Path.home() / ".config" / "komaxtch" / ".env"
AUTO = "auto"
DETECTION_LABELS = {
    "opencv": "detecção local (OpenCV)",
    "ollama": "IA local (Ollama)",
    "gemini": "Gemini (nuvem grátis)",
    "whole-page": "página inteira",
    "page": "página inteira (modo page)",
}


class Cli:
    def __init__(self) -> None:
        self._parser = self._build_parser()

    def run(self, argv: list[str] | None = None) -> int:
        load_dotenv(Path.cwd() / ".env")
        load_dotenv(USER_ENV_FILE)  # vale de qualquer pasta; o .env local tem prioridade
        logging.basicConfig(level=logging.WARNING, format="aviso: %(message)s")
        args = self._parser.parse_args(argv)
        try:
            return args.handler(args)
        except (ValueError, XtchFormatError, FileNotFoundError) as error:
            print(f"erro: {error}", file=sys.stderr)
            return 1

    # --- comandos -------------------------------------------------------

    def _convert(self, args: argparse.Namespace) -> int:
        settings = self._settings(args)
        layouts = LayoutFactory(settings)
        if settings.mode is not LayoutMode.PAGE:
            self._print_ai_status(layouts)
        books = SourceFactory().expand(args.inputs)
        pipeline = ConversionPipeline(settings, layouts)
        for book in books:
            output = self._output_path(book, args.output, multiple=len(books) > 1)
            started = time.monotonic()
            report = pipeline.convert(book, output, progress=_print_progress, status=print)
            self._print_report(report, output, time.monotonic() - started)
        return 0

    def _preview(self, args: argparse.Namespace) -> int:
        modes = [LayoutMode(mode) for mode in args.modes.split(",")]
        sheets = PreviewBuilder(self._settings(args)).build(
            args.input, parse_page_selection(args.pages), modes, args.output
        )
        for sheet in sheets:
            print(f"comparação: {sheet}")
        return 0

    def _debug(self, args: argparse.Namespace) -> int:
        for path in DebugOverlay(self._settings(args)).build(args.input, parse_page_selection(args.pages), args.output):
            print(f"depuração: {path}")
        return 0

    def _inspect(self, args: argparse.Namespace) -> int:
        reader = XtchReader(args.file)
        book = reader.book
        sizes = {(p.width, p.height) for p in book.pages}
        print(f"título: {book.title or '-'}  autor: {book.author or '-'}  versão: {book.version[0]}.{book.version[1]}")
        print(f"telas: {len(book.pages)}  tamanhos: {', '.join(f'{w}x{h}' for w, h in sorted(sizes))}")
        if args.extract:
            args.extract.mkdir(parents=True, exist_ok=True)
        for number, screen in enumerate(_screens(reader), start=1):  # decodificar tudo valida o arquivo
            if args.extract:
                screen.save(args.extract / f"tela_{number:04d}.png")
        print("arquivo válido")
        if args.pdf:
            self._save_pdf(reader, args.pdf)
            print(f"prévia em PDF: {args.pdf} (use 'Tamanho Real' para ver no tamanho da tela)")
        return 0

    @staticmethod
    def _save_pdf(reader: XtchReader, path: Path) -> None:
        first_page = reader.book.pages[0]
        ppi = device_for_size(first_page.width, first_page.height).ppi
        screens = _screens(reader)
        path.parent.mkdir(parents=True, exist_ok=True)
        next(screens).save(path, save_all=True, append_images=screens, resolution=ppi)

    # --- apoio ----------------------------------------------------------

    @staticmethod
    def _settings(args: argparse.Namespace) -> ConversionSettings:
        return ConversionSettings(
            device=get_device(args.device),
            mode=LayoutMode(getattr(args, "mode", LayoutMode.PANEL.value)),
            direction=None if args.direction == AUTO else ReadingDirection(args.direction),
            dithering=Dithering(args.dither),
            rotation=Rotation(args.rotate),
            use_local_ai=args.local_ai,
            use_cloud=args.cloud,
            split_spreads=args.split_spreads,
        )

    @staticmethod
    def _output_path(book: Path, output: Path | None, multiple: bool) -> Path:
        if output is None:
            return book.with_suffix(OUTPUT_SUFFIX) if book.is_file() else book.parent / f"{book.name}{OUTPUT_SUFFIX}"
        if multiple or output.is_dir():
            return output / f"{book.stem}{OUTPUT_SUFFIX}"
        return output

    @staticmethod
    def _print_ai_status(layouts: LayoutFactory) -> None:
        cloud = "Gemini ligada" if layouts.cloud_enabled else f"Gemini desligada (defina {GEMINI_KEY_VARIABLE} em {USER_ENV_FILE})"
        local = "; Ollama ligado" if layouts.local_ai_enabled else ""
        print(f"IA para páginas difíceis: {cloud}{local}")

    @staticmethod
    def _print_report(report: ConversionReport, output: Path, seconds: float) -> None:
        origin = "do livro" if report.direction_detected else "padrão; use --direction rtl para mangá"
        print(f"\n✔ {report.title} → {output}")
        print(f"  {report.pages} páginas → {report.screens} telas em {seconds:.0f}s")
        print(f"  direção: {report.direction.value} ({origin})")
        for source, count in report.detections.most_common():
            print(f"  {DETECTION_LABELS.get(source, source)}: {count} páginas")
        if report.ai_pending_pages:
            when = f" a partir de {report.ai_resume_hint}" if report.ai_resume_hint else " mais tarde"
            print(
                f"  ⚠ cota grátis da IA esgotada: {report.ai_pending_pages} páginas difíceis ficaram com o"
                f" palpite local. Rode o mesmo comando{when} para completar"
                " (o que já foi analisado fica guardado)."
            )

    def _build_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            prog="komaxtch", description="Mangá e HQ quadro a quadro para o Xteink (CrossPoint)."
        )
        commands = parser.add_subparsers(required=True, metavar="comando")

        convert = commands.add_parser("convert", help="converte EPUB/CBZ/pasta em .xtch")
        convert.add_argument("inputs", nargs="+", type=Path, help="arquivos ou pastas")
        convert.add_argument("-o", "--output", type=Path, help="arquivo ou pasta de saída")
        self._add_mode(convert)
        self._add_common(convert)
        convert.set_defaults(handler=self._convert)

        preview = commands.add_parser("preview", help="compara os modos em algumas páginas (PNG)")
        preview.add_argument("input", type=Path)
        preview.add_argument("--pages", default="1-5", help="ex.: 1-5,12")
        preview.add_argument("--modes", default="page,panel,bubble")
        preview.add_argument("-o", "--output", type=Path, default=Path("preview"))
        self._add_common(preview)
        preview.set_defaults(handler=self._preview)

        debug = commands.add_parser("debug", help="mostra quadros e balões detectados (PNG)")
        debug.add_argument("input", type=Path)
        debug.add_argument("--pages", default="1-5")
        debug.add_argument("-o", "--output", type=Path, default=Path("debug"))
        self._add_common(debug)
        debug.set_defaults(handler=self._debug)

        inspect = commands.add_parser("inspect", help="valida um .xtch")
        inspect.add_argument("file", type=Path)
        inspect.add_argument("--extract", type=Path, help="salva cada tela como PNG nesta pasta")
        inspect.add_argument("--pdf", type=Path, help="gera um PDF com as telas, para ver no computador")
        inspect.set_defaults(handler=self._inspect)
        return parser

    @staticmethod
    def _add_mode(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--mode", choices=[m.value for m in LayoutMode], default=LayoutMode.PANEL.value,
            help="page = página inteira, panel = um quadro por tela, bubble = quadro + zoom nos balões",
        )

    @staticmethod
    def _add_common(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--device", choices=list(DEVICES), default=DEFAULT_DEVICE)
        parser.add_argument("--direction", choices=[AUTO, "rtl", "ltr"], default=AUTO,
                            help="rtl = mangá, ltr = HQ, auto = metadado do livro")
        parser.add_argument("--dither", choices=[d.value for d in Dithering], default=Dithering.FLOYD.value)
        parser.add_argument("--rotate", choices=[r.value for r in Rotation], default=Rotation.CLOCKWISE.value,
                            help="sentido do giro dos quadros largos")
        parser.add_argument("--local-ai", dest="local_ai", action="store_true",
                            help="usar também o Ollama (pesado: ~4 GB de RAM)")
        parser.add_argument("--no-cloud", dest="cloud", action="store_false", help="não usar a Gemini")
        parser.add_argument("--no-split-spreads", dest="split_spreads", action="store_false",
                            help="não dividir páginas duplas")


def _screens(reader: XtchReader) -> Iterator[Image.Image]:
    for index in range(len(reader.book.pages)):
        yield EInkRenderer.to_image(reader.page_levels(index))


def _print_progress(done: int, total: int) -> None:
    print(f"\r  página {done}/{total}", end="" if done < total else "\n", flush=True)


def main() -> None:
    sys.exit(Cli().run())
