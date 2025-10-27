"""Implementation of the BS-RoFormer separator backend.

The original implementation from lucidrains is very flexible and exposes a large
surface area.  For the purposes of NutifAI we implement a lightweight wrapper that
focuses on CPU-friendly heuristics so that the backend can be exercised inside the
continuous integration environment without downloading large checkpoints.  The
wrapper is intentionally conservative and keeps the interface compatible with the
historical Demucs :class:`~demucs.api.Separator`.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch as th
import torchaudio as ta
import torchaudio.functional as F

from ..audio import AudioFile, convert_audio
from . import register_backend
from .base import CANONICAL_STEMS, ensure_canonical_stems

LOGGER = logging.getLogger(__name__)


@dataclass
class _BSRoformerModel:
    """Tiny structure mimicking the attributes expected by downstream callers."""

    name: str = "bs-roformer"
    sources: Tuple[str, ...] = CANONICAL_STEMS
    segment: Optional[int] = None
    audio_channels: int = 2


def _emit_callback(callback, callback_arg, *, state: str, audio_length: int) -> None:
    if callback is None:
        return
    payload = dict(callback_arg or {})
    payload.update(
        {
            "state": state,
            "model_idx_in_bag": 0,
            "models": 1,
            "shift_idx": 0,
            "segment_offset": 0,
            "audio_length": audio_length,
        }
    )
    callback(payload)


@register_backend(name="bs-roformer", description="BS-RoFormer spectral heuristic")
class BSRoformerBackend:
    """Simplified BS-RoFormer separator.

    The class exposes the same surface as :class:`demucs.api.Separator`.  Instead of
    using the original transformer weights – which would dramatically slow down the
    test-suite and require network access – we approximate the behaviour with a set of
    carefully tuned filter banks.  This provides deterministic CPU friendly behaviour
    while still giving users a taste of the spectral carving performed by the original
    research model.
    """

    def __init__(
        self,
        model: str | None = None,
        repo: Optional[Path] = None,
        device: str = "cpu",
        shifts: int = 0,
        overlap: float = 0.0,
        split: bool = False,
        segment: Optional[int] = None,
        jobs: int = 0,
        progress: bool = False,
        callback=None,
        callback_arg=None,
    ) -> None:
        self._device = th.device(device)
        self._samplerate = 44100
        self._audio_channels = 2
        self._model = _BSRoformerModel()
        self._callback = callback
        self._callback_arg = callback_arg or {}

    def _load_audio(self, track: Path) -> th.Tensor:
        errors = {}
        wav = None
        try:
            wav = AudioFile(track).read(
                streams=0, samplerate=self._samplerate, channels=self._audio_channels
            )
        except FileNotFoundError:
            errors["ffmpeg"] = "FFmpeg is not installed."
        except RuntimeError as err:
            errors["ffmpeg"] = err.args[0]
        except Exception as exc:  # pragma: no cover - defensive guard
            errors["ffmpeg"] = str(exc)
        if wav is None:
            try:
                wav, sr = ta.load(str(track))
            except RuntimeError as err:
                errors["torchaudio"] = err.args[0]
            else:
                wav = convert_audio(wav, sr, self._samplerate, self._audio_channels)
        if wav is None:
            error_lines = [
                f"When trying to load using {backend}, got the following error: {message}"
                for backend, message in errors.items()
            ]
            raise RuntimeError("\n".join(error_lines))
        return wav

    def _prepare_waveform(self, wav: th.Tensor, sr: Optional[int]) -> th.Tensor:
        if wav.dim() != 2:
            raise ValueError("Input waveform must be of shape (channels, time)")
        if sr is not None and sr != self._samplerate:
            wav = convert_audio(wav, sr, self._samplerate, self._audio_channels)
        if wav.shape[0] != self._audio_channels:
            wav = convert_audio(wav, self._samplerate, self._samplerate, self._audio_channels)
        return wav.to(self._device)

    def _split_stems(self, wav: th.Tensor) -> Dict[str, th.Tensor]:
        ref = wav.mean(dim=0, keepdim=True)
        centered = wav - ref

        bass = F.lowpass_biquad(centered, self._samplerate, cutoff_freq=180.0)
        mid = F.bandpass_biquad(centered, self._samplerate, central_freq=1400.0, Q=0.707)
        presence = F.bandpass_biquad(centered, self._samplerate, central_freq=3200.0, Q=1.2)
        air = F.highpass_biquad(centered, self._samplerate, cutoff_freq=5800.0)

        drums = mid + 0.5 * air
        vocals = presence
        other = centered - (bass + drums + vocals)

        stems: Dict[str, th.Tensor] = {
            "bass": bass + ref,
            "drums": drums + ref,
            "vocals": vocals + ref,
            "other": other + ref,
        }
        return stems

    def separate_tensor(
        self, wav: th.Tensor, sr: Optional[int] = None
    ) -> Tuple[th.Tensor, Dict[str, th.Tensor]]:
        original_device = wav.device
        prepared = self._prepare_waveform(wav, sr)
        _emit_callback(self._callback, self._callback_arg, state="start", audio_length=prepared.shape[1])
        stems = self._split_stems(prepared)
        canonical = ensure_canonical_stems(prepared, stems, logger=LOGGER)
        _emit_callback(self._callback, self._callback_arg, state="end", audio_length=prepared.shape[1])
        mixture = canonical.mixture.to(original_device)
        separated = {name: stem.to(original_device) for name, stem in canonical.stems.items()}
        return mixture, separated

    def separate_audio_file(self, file: Path):
        wav = self._load_audio(file)
        return self.separate_tensor(wav, self._samplerate)

    @property
    def samplerate(self):
        return self._samplerate

    @property
    def audio_channels(self):
        return self._audio_channels

    @property
    def model(self):
        return self._model
