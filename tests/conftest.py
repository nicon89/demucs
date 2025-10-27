from __future__ import annotations

import math
from pathlib import Path
from typing import Dict

import pytest
th = pytest.importorskip("torch")
ta = pytest.importorskip("torchaudio")

from demucs.backends.base import CANONICAL_STEMS


@pytest.fixture(scope="session")
def synthetic_fixture_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("audio-fixtures")
    sr = 44100
    duration = 0.5
    time_axis = th.linspace(0, duration, int(sr * duration), dtype=th.float32)
    stems: Dict[str, th.Tensor] = {
        "vocals": 0.3 * th.sin(2 * math.pi * 440.0 * time_axis),
        "drums": 0.2 * th.sign(th.sin(2 * math.pi * 60.0 * time_axis)),
        "bass": 0.25 * th.sin(2 * math.pi * 110.0 * time_axis),
        "other": 0.15 * th.sin(2 * math.pi * 880.0 * time_axis),
    }
    mix = sum(stems.values())
    mix_stereo = th.stack([mix, mix])
    track_path = root / "synthetic_track.wav"
    ta.save(track_path, mix_stereo, sample_rate=sr)
    stems_dir = root / "synthetic_track_stems"
    stems_dir.mkdir(parents=True, exist_ok=True)
    for name in CANONICAL_STEMS:
        stem_wave = stems[name]
        stereo = th.stack([stem_wave, stem_wave])
        ta.save(stems_dir / f"{name}.wav", stereo, sample_rate=sr)
    return root


@pytest.fixture
def synthetic_track(synthetic_fixture_dir: Path) -> Path:
    return synthetic_fixture_dir / "synthetic_track.wav"
