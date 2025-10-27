"""Wrapper exposing the historical Demucs separator through the backend registry."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch as th

from ..api import Separator as DemucsSeparator
from .base import ensure_canonical_stems
from . import register_backend

LOGGER = logging.getLogger(__name__)


@register_backend(name="demucs", description="Classic Demucs separator")
class DemucsBackend:
    """Thin wrapper around :class:`demucs.api.Separator`."""

    def __init__(self, *args, **kwargs) -> None:
        self._separator = DemucsSeparator(*args, **kwargs)

    def separate_tensor(
        self, wav: th.Tensor, sr: Optional[int] = None
    ) -> Tuple[th.Tensor, Dict[str, th.Tensor]]:
        mixture, stems = self._separator.separate_tensor(wav, sr)
        canonical = ensure_canonical_stems(mixture, stems, logger=LOGGER)
        return canonical.mixture, canonical.stems

    def separate_audio_file(self, file: Path):
        mixture, stems = self._separator.separate_audio_file(file)
        canonical = ensure_canonical_stems(mixture, stems, logger=LOGGER)
        return canonical.mixture, canonical.stems

    @property
    def samplerate(self):
        return self._separator.samplerate

    @property
    def audio_channels(self):
        return self._separator.audio_channels

    @property
    def model(self):
        return self._separator.model
