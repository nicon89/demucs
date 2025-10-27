from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import torchaudio as ta

from demucs.api import (
    available_separator_backends,
    create_separator_backend,
)
from demucs.backends.base import CANONICAL_STEMS


def test_available_backends_exposes_bs_roformer():
    backends = available_separator_backends()
    assert "bs-roformer" in backends
    assert "demucs" in backends


def test_bs_roformer_separator_produces_canonical_stems(synthetic_track: Path):
    waveform, sr = ta.load(str(synthetic_track))
    separator = create_separator_backend("bs-roformer", device="cpu")
    mixture, stems = separator.separate_tensor(waveform, sr)
    assert set(stems) == set(CANONICAL_STEMS)
    for stem in CANONICAL_STEMS:
        assert stems[stem].shape == mixture.shape
        assert stems[stem].dtype == mixture.dtype


def test_cli_separation_with_bs_roformer(tmp_path: Path, synthetic_track: Path):
    output_dir = tmp_path / "separated"
    cmd = [
        sys.executable,
        "-m",
        "demucs.separate",
        str(synthetic_track),
        "--device",
        "cpu",
        "--separator",
        "bs-roformer",
        "--no-split",
        "--jobs",
        "0",
        "--out",
        str(output_dir),
    ]
    subprocess.run(cmd, check=True)
    produced = sorted(output_dir.rglob("*.wav"))
    assert produced, "Expected separated stems to be generated"
    stem_names = {path.stem for path in produced}
    for stem in CANONICAL_STEMS:
        assert any(stem in name for name in stem_names)


def test_benchmark_script_runs(tmp_path: Path, synthetic_track: Path):
    output_dir = tmp_path / "bench"
    cmd = [
        sys.executable,
        "tools/benchmark_separators.py",
        str(synthetic_track),
        "--backends",
        "bs-roformer",
        "--output-dir",
        str(output_dir),
    ]
    subprocess.run(cmd, check=True)
    csv_path = output_dir / "benchmark_results.csv"
    summary_path = output_dir / "benchmark_summary.md"
    assert csv_path.exists()
    assert summary_path.exists()
    assert "bs-roformer" in summary_path.read_text()
