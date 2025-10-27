"""Utility to generate deterministic smoke-test audio fixtures without binary blobs."""
from __future__ import annotations

import math
import random
from array import array
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

STEMS: List[str] = [
    "vocals",
    "drums",
    "bass",
    "guitar",
    "piano",
    "keys",
    "strings",
    "other",
]

SAMPLE_RATE = 44100
DURATION = 0.25  # seconds
NUM_COMPONENTS = 3

StereoSignal = Tuple[List[float], List[float]]


def _write_wave(path: Path, audio: StereoSignal) -> None:
    import wave

    left, right = audio
    if len(left) != len(right):  # pragma: no cover - sanity guard
        raise ValueError("Stereo channels must have identical lengths")
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = array("h")
    for l, r in zip(left, right):
        pcm.append(int(max(-1.0, min(1.0, l)) * 32767))
        pcm.append(int(max(-1.0, min(1.0, r)) * 32767))
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(pcm.tobytes())


def _constant_power_pan(sample: float, pan: float) -> Tuple[float, float]:
    left_gain = math.cos((pan + 1.0) * math.pi / 4.0)
    right_gain = math.sin((pan + 1.0) * math.pi / 4.0)
    return sample * left_gain, sample * right_gain


def _generate_stem(seed: int, num_samples: int) -> StereoSignal:
    rng = random.Random(seed)
    components = [
        (
            rng.uniform(80.0, 1800.0),
            rng.uniform(0.0, 2.0 * math.pi),
            rng.uniform(0.05, 0.25),
            rng.uniform(-0.75, 0.75),
        )
        for _ in range(NUM_COMPONENTS)
    ]
    left = [0.0] * num_samples
    right = [0.0] * num_samples
    if num_samples <= 1:
        return left, right
    for idx in range(num_samples):
        t = idx / SAMPLE_RATE
        envelope = 1.0 - 0.8 * (idx / (num_samples - 1))
        for freq, phase, amp, pan in components:
            sample = amp * math.sin(2.0 * math.pi * freq * t + phase) * envelope
            l_val, r_val = _constant_power_pan(sample, pan)
            left[idx] += l_val
            right[idx] += r_val
    return left, right


def _mix_inplace(accumulator: StereoSignal, addition: StereoSignal) -> None:
    left_acc, right_acc = accumulator
    left_add, right_add = addition
    for idx in range(len(left_acc)):
        left_acc[idx] += left_add[idx]
        right_acc[idx] += right_add[idx]


def _write_clip(clip_dir: Path, seeds: Sequence[int], num_samples: int) -> None:
    mix: StereoSignal = ([0.0] * num_samples, [0.0] * num_samples)
    for stem_name, stem_seed in zip(STEMS, seeds):
        audio = _generate_stem(int(stem_seed), num_samples)
        _write_wave(clip_dir / f"{stem_name}.wav", audio)
        _mix_inplace(mix, audio)
    _write_wave(clip_dir / "mixture.wav", mix)


def ensure_test_small_dataset(root: Path, clips: int = 6) -> Path:
    """Ensure that the synthetic `test_small` dataset exists under ``root``."""
    subset_root = root / "test_small"
    subset_root.mkdir(parents=True, exist_ok=True)
    expected_mixture = subset_root / "clip01" / "mixture.wav"
    if expected_mixture.exists():
        return subset_root

    num_samples = int(SAMPLE_RATE * DURATION)
    for clip_idx in range(clips):
        clip_dir = subset_root / f"clip{clip_idx + 1:02d}"
        seeds = [clip_idx * len(STEMS) + i + 1 for i in range(len(STEMS))]
        _write_clip(clip_dir, seeds, num_samples)

    for clip_idx in range(clips):
        clip_dir = subset_root / f"clip{clip_idx + 1:02d}"
        for stem in STEMS + ["mixture"]:
            path = clip_dir / f"{stem}.wav"
            if not path.exists():
                raise RuntimeError(f"Failed to create {path}")
    return subset_root


__all__ = ["ensure_test_small_dataset", "STEMS", "SAMPLE_RATE", "DURATION"]
