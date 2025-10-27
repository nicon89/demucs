"""Benchmark Demucs against the BS-RoFormer backend."""
from __future__ import annotations

import argparse
import csv
import math
import resource
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, List, Mapping, Sequence
try:  # pragma: no cover - import guard for optional dependencies
    import torch as th
except ModuleNotFoundError as exc:  # pragma: no cover - user friendly error
    raise SystemExit(
        "PyTorch is required to run the separator benchmark. Install torch first."
    ) from exc

try:  # pragma: no cover - import guard for optional dependencies
    import torchaudio as ta
except ModuleNotFoundError as exc:  # pragma: no cover - user friendly error
    raise SystemExit(
        "Torchaudio is required to run the separator benchmark. Install torchaudio first."
    ) from exc

from museval.metrics import bss_eval_sources

from demucs.audio import convert_audio
from demucs.backends import available_backends, create_separator
from demucs.backends.base import CANONICAL_STEMS


@dataclass
class StemMetrics:
    track: str
    backend: str
    stem: str
    sdr: float
    sir: float
    sar: float
    runtime_s: float
    rt_factor: float
    peak_memory_mb: float


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Audio files to benchmark. References are expected next to the audio files,"
        " inside a '<track>_stems' folder.",
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=list(available_backends()),
        default=["demucs", "bs-roformer"],
        help="Separator backends to evaluate.",
    )
    parser.add_argument(
        "--demucs-model",
        default="htdemucs",
        help="Pretrained Demucs model to use when the demucs backend is selected.",
    )
    parser.add_argument("--segment", type=int, default=None, help="Override segment duration.")
    parser.add_argument("--shifts", type=int, default=1, help="Number of shifts to average.")
    parser.add_argument("--overlap", type=float, default=0.25, help="Segment overlap.")
    parser.add_argument("--jobs", type=int, default=0, help="Number of worker jobs.")
    parser.add_argument(
        "--device",
        default="cuda" if th.cuda.is_available() else "cpu",
        help="Torch device to run the benchmark on.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_results"),
        help="Directory used to write the CSV and markdown summary.",
    )
    parser.add_argument(
        "--references-root",
        type=Path,
        default=None,
        help="Optional root directory containing reference stems organised as '<track>/<stem>.wav'.",
    )
    return parser.parse_args(argv)


def _reference_path(track: Path, stem: str, *, references_root: Path | None) -> Path:
    if references_root is not None:
        return references_root / track.stem / f"{stem}.wav"
    return track.parent / f"{track.stem}_stems" / f"{stem}.wav"


def _load_audio(path: Path) -> tuple[th.Tensor, int]:
    wav, sr = ta.load(str(path))
    return wav, sr


def _mono(tensor: th.Tensor) -> th.Tensor:
    if tensor.dim() != 2:
        raise ValueError("waveforms must be 2D")
    return tensor.mean(dim=0, keepdim=True)


def _compute_bss_metrics(reference: th.Tensor, estimate: th.Tensor) -> tuple[float, float, float]:
    ref = _mono(reference).cpu().numpy()
    est = _mono(estimate).cpu().numpy()
    sdr, sir, sar, _ = bss_eval_sources(ref, est)
    return float(sdr.mean()), float(sir.mean()), float(sar.mean())


def _current_peak_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        usage /= 1024 * 1024
    else:
        usage /= 1024
    return float(usage)


def _ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _instantiate_backend(name: str, args: argparse.Namespace):
    kwargs = dict(
        device=args.device,
        segment=args.segment,
        shifts=args.shifts,
        overlap=args.overlap,
        jobs=args.jobs,
        progress=False,
    )
    if name == "demucs":
        kwargs["model"] = args.demucs_model
    return create_separator(name, **kwargs)


def _benchmark_backend(
    backend: str,
    args: argparse.Namespace,
    tracks: Sequence[Path],
    baseline_memory: float,
) -> List[StemMetrics]:
    separator = _instantiate_backend(backend, args)
    results: List[StemMetrics] = []
    backend_peak = baseline_memory
    for track in tracks:
        waveform, sr = _load_audio(track)
        waveform = convert_audio(waveform, sr, separator.samplerate, separator.audio_channels)
        duration = waveform.shape[-1] / separator.samplerate
        start = time.perf_counter()
        _, stems = separator.separate_tensor(waveform, separator.samplerate)
        runtime = time.perf_counter() - start
        backend_peak = max(backend_peak, _current_peak_mb())
        rt_factor = duration / runtime if runtime > 0 else math.inf
        for stem_name in CANONICAL_STEMS:
            reference_path = _reference_path(track, stem_name, references_root=args.references_root)
            reference_audio, ref_sr = _load_audio(reference_path)
            reference_audio = convert_audio(
                reference_audio, ref_sr, separator.samplerate, separator.audio_channels
            )
            est_audio = stems[stem_name]
            sdr, sir, sar = _compute_bss_metrics(reference_audio, est_audio)
            results.append(
                StemMetrics(
                    track=track.stem,
                    backend=backend,
                    stem=stem_name,
                    sdr=sdr,
                    sir=sir,
                    sar=sar,
                    runtime_s=runtime,
                    rt_factor=rt_factor,
                    peak_memory_mb=max(0.0, backend_peak - baseline_memory),
                )
            )
    return results


def _write_csv(path: Path, rows: Sequence[StemMetrics]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "track",
                "backend",
                "stem",
                "sdr",
                "sir",
                "sar",
                "runtime_s",
                "rt_factor",
                "peak_memory_mb",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.track,
                    row.backend,
                    row.stem,
                    f"{row.sdr:.4f}",
                    f"{row.sir:.4f}",
                    f"{row.sar:.4f}",
                    f"{row.runtime_s:.4f}",
                    f"{row.rt_factor:.4f}",
                    f"{row.peak_memory_mb:.2f}",
                ]
            )


def _summarise(rows: Sequence[StemMetrics]) -> Mapping[str, Mapping[str, float]]:
    per_backend: Dict[str, List[StemMetrics]] = {}
    for row in rows:
        per_backend.setdefault(row.backend, []).append(row)
    summary: Dict[str, Dict[str, float]] = {}
    for backend, backend_rows in per_backend.items():
        sdr_values = [row.sdr for row in backend_rows]
        sir_values = [row.sir for row in backend_rows]
        sar_values = [row.sar for row in backend_rows]
        rt_values = [row.rt_factor for row in backend_rows]
        memory_values = [row.peak_memory_mb for row in backend_rows]
        summary[backend] = {
            "vocals_sdr": mean(row.sdr for row in backend_rows if row.stem == "vocals"),
            "drums_sdr": mean(row.sdr for row in backend_rows if row.stem == "drums"),
            "bass_sdr": mean(row.sdr for row in backend_rows if row.stem == "bass"),
            "other_sdr": mean(row.sdr for row in backend_rows if row.stem == "other"),
            "avg_sdr": mean(sdr_values),
            "avg_sir": mean(sir_values),
            "avg_sar": mean(sar_values),
            "rt_x": mean(rt_values),
            "peak_memory_mb": max(memory_values) if memory_values else 0.0,
        }
    return summary


def _write_markdown(path: Path, summary: Mapping[str, Mapping[str, float]]) -> None:
    headers = [
        "backend",
        "vocals SDR",
        "drums SDR",
        "bass SDR",
        "other SDR",
        "avg SDR",
        "avg SIR",
        "avg SAR",
        "rt_x",
        "peak MB",
    ]
    with path.open("w") as handle:
        handle.write("| " + " | ".join(headers) + " |\n")
        handle.write("|" + "---|" * len(headers) + "\n")
        for backend in sorted(summary):
            stats = summary[backend]
            handle.write(
                "| {backend} | {vocals:.2f} | {drums:.2f} | {bass:.2f} | {other:.2f} | {avg:.2f} | {sir:.2f} | {sar:.2f} | {rt:.2f} | {mem:.1f} |\n".format(
                    backend=backend,
                    vocals=stats["vocals_sdr"],
                    drums=stats["drums_sdr"],
                    bass=stats["bass_sdr"],
                    other=stats["other_sdr"],
                    avg=stats["avg_sdr"],
                    sir=stats["avg_sir"],
                    sar=stats["avg_sar"],
                    rt=stats["rt_x"],
                    mem=stats["peak_memory_mb"],
                )
            )


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv or sys.argv[1:])
    tracks = [path for path in args.inputs if path.exists()]
    if not tracks:
        raise SystemExit("No valid input tracks found.")
    baseline_memory = _current_peak_mb()
    all_rows: List[StemMetrics] = []
    for backend in args.backends:
        rows = _benchmark_backend(backend, args, tracks, baseline_memory)
        all_rows.extend(rows)
    _ensure_directory(args.output_dir)
    _write_csv(args.output_dir / "benchmark_results.csv", all_rows)
    _write_markdown(args.output_dir / "benchmark_summary.md", _summarise(all_rows))


if __name__ == "__main__":  # pragma: no cover - manual usage entrypoint
    main()
