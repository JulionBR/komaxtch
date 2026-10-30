"""Perfis dos leitores Xteink suportados."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DeviceProfile:
    name: str
    width: int
    height: int
    ppi: int  # densidade da tela, para prévias em tamanho real


DEVICES: dict[str, DeviceProfile] = {
    "x3": DeviceProfile("x3", 528, 792, 259),
    "x4": DeviceProfile("x4", 480, 800, 219),
}

DEFAULT_DEVICE = "x3"


def get_device(name: str) -> DeviceProfile:
    try:
        return DEVICES[name.lower()]
    except KeyError:
        known = ", ".join(DEVICES)
        raise ValueError(f"Aparelho desconhecido '{name}'. Opções: {known}") from None


def device_for_size(width: int, height: int) -> DeviceProfile:
    """Aparelho com essa resolução de tela; o padrão quando nenhum bate."""
    return next((d for d in DEVICES.values() if (d.width, d.height) == (width, height)), DEVICES[DEFAULT_DEVICE])
