import json
import subprocess
import sys
from pathlib import Path

import pytest

from tests.fixtures.audio import ensure_test_small_dataset


try:
    import torch
except ImportError:  # pragma: no cover - cpu only CI
    torch = None  # type: ignore


@pytest.fixture(scope="module")
def cpu_device():
    if torch is None:
        pytest.skip("torch not available")
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def tiny_audio():
    if torch is None:
        pytest.skip("torch not available")
    torch.manual_seed(0)
    return torch.randn(2, 4096)


@pytest.mark.skipif(torch is None, reason="torch not available")
def test_htdemucs_8s_outputs(cpu_device, tiny_audio):
    from demucs.api import Separator

    separator = Separator(model="htdemucs_8s", device=cpu_device, split=False, progress=False,
                          tta_flips=True, tta_aggregate="median")
    _, stems = separator.separate_tensor(tiny_audio, separator.samplerate)
    assert "strings" in stems and "keys" in stems
    energy = {name: value.abs().sum().item() for name, value in stems.items()}
    assert any(v > 0 for v in energy.values())


@pytest.mark.skipif(torch is None, reason="torch not available")
def test_htdemucs_8s_small_segment_chunk(cpu_device, tiny_audio):
    from demucs.api import Separator

    separator = Separator(
        model="htdemucs_8s",
        device=cpu_device,
        split=True,
        progress=False,
        segment=7,
        shifts=0,
    )
    _, stems = separator.separate_tensor(tiny_audio, separator.samplerate)
    assert stems["strings"].shape[-1] == tiny_audio.shape[-1]


@pytest.mark.skipif(torch is None, reason="torch not available")
def test_deterministic_separator(tmp_path, cpu_device, tiny_audio):
    from demucs.api import Separator

    separator = Separator(model="htdemucs_6s", device=cpu_device, split=False, progress=False,
                          shifts=0)
    torch.manual_seed(123)
    first = separator.separate_tensor(tiny_audio.clone(), separator.samplerate)[1]
    torch.manual_seed(123)
    second = separator.separate_tensor(tiny_audio.clone(), separator.samplerate)[1]
    for stem in separator.model.sources:
        torch.testing.assert_close(first[stem], second[stem], atol=1e-6, rtol=1e-4)


@pytest.mark.parametrize("tracks", [1])
def test_eval_baseline_script(tmp_path, tracks):
    ensure_test_small_dataset(Path("tests/fixtures/audio"))
    out_dir = tmp_path / "baseline"
    out_dir.mkdir()
    cmd = [
        sys.executable,
        "tools/eval_baseline.py",
        "--model",
        "htdemucs_6s",
        "--subset",
        "test_small",
        "--dataset-root",
        "tests/fixtures/audio",
        "--tracks",
        str(tracks),
        "--out",
        str(out_dir),
        "--offline",
    ]
    subprocess.run(cmd, check=True)
    metrics = out_dir / "metrics.csv"
    summary = out_dir / "summary.md"
    assert metrics.exists()
    assert summary.exists()
    content = summary.read_text()
    assert "Baseline Evaluation" in content


def test_benchmark_script(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    csv_path = run_dir / "metrics.csv"
    csv_path.write_text("track,stem,metric,value\nclip,vocals,sdr,1.0\nclip,bass,sdr,0.5\n")
    output = tmp_path / "table.md"
    cmd = [
        sys.executable,
        "tools/benchmark.py",
        str(run_dir),
        "--names",
        "demo",
        "--output",
        str(output),
    ]
    subprocess.run(cmd, check=True)
    text = output.read_text()
    assert "demo" in text
    assert "vocals SDR" in text


@pytest.mark.skipif(torch is None, reason="torch not available")
def test_cli_two_stems(tmp_path):
    ensure_test_small_dataset(Path("tests/fixtures/audio"))
    mixture = Path("tests/fixtures/audio/test_small/clip01/mixture.wav")
    out_dir = tmp_path / "sep"
    cmd = [
        sys.executable,
        "-m",
        "demucs.separate",
        str(mixture),
        "--name",
        "htdemucs_6s",
        "--two-stems",
        "vocals",
        "--segment",
        "1",
        "--out",
        str(out_dir),
        "--tta-flip",
    ]
    subprocess.run(cmd, check=True)
    output_dir = out_dir / "htdemucs_6s"
    stems = list(output_dir.rglob("*.wav"))
    assert len(stems) == 2
