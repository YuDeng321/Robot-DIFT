"""Utilities for keeping Robocasa environment names consistent."""

from __future__ import annotations

from typing import Iterable, Sequence

# More descriptive task names (used in configs) must be mapped to how the
# datasets / robosuite register the environments (PnP*).
PICK_PLACE_ALIASES = {
    "PickPlaceCounterToCabinet": "PnPCounterToCab",
    "PickPlaceCabinetToCounter": "PnPCabToCounter",
    "PickPlaceCounterToSink": "PnPCounterToSink",
    "PickPlaceSinkToCounter": "PnPSinkToCounter",
    "PickPlaceCounterToMicrowave": "PnPCounterToMicrowave",
    "PickPlaceMicrowaveToCounter": "PnPMicrowaveToCounter",
    "PickPlaceCounterToStove": "PnPCounterToStove",
    "PickPlaceStoveToCounter": "PnPStoveToCounter",
}


def normalize_env_name(name: str) -> str:
    """Return the robosuite-registered name for the friendly alias."""
    return PICK_PLACE_ALIASES.get(name, name)


def normalize_env_names(names: Sequence[str] | Iterable[str]) -> list[str]:
    """Vectorized helper that normalizes each provided env alias."""
    return [normalize_env_name(name) for name in names]
