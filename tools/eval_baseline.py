#!/usr/bin/env python
"""Reproducible baseline evaluation with optional offline mode."""

from __future__ import annotations

import argparse
import csv
import json
import math
import struct
import wave
from pathlib import Path
from typing import Dict, Iterable, List, Optional

try:  # Heavy deps are optional so the script still runs in minimal environments.
    import torch as th
    from demucs.api import Separator
    from demucs.evaluate import new_sdr
except ImportError:  # pragma: no cover - offline mode
    th = None  # type: ignore
    Separator = None  # type: ignore
    new_sdr = None  # type: ignore

try:  # pragma: no cover - optional for real separation
    import torchaudio as ta  # type: ignore
except ImportError:
    ta = None  # type: ignore

try:  # pragma: no cover - optional if museval available
    import numpy as np  # type: ignore
    import museval  # type: ignore
except ImportError:
    np = None  # type: ignore
    museval = None  # type: ignore

try:  # pragma: no cover - optional synthetic fixtures
    from tests.fixtures.audio import ensure_test_small_dataset  # type: ignore
except ImportError:  # pragma: no cover - installed package case
    ensure_test_small_dataset = None  # type: ignore


def _load_audio_torch(path: Path, target_sr: int) -> th.Tensor:
    if ta is None:
        raise RuntimeError("torchaudio is required for model-based evaluation")
    wav, sr = ta.load(str(path))
    if sr != target_sr:
        wav = ta.functional.resample(wav, sr, target_sr)
    return wav


def _read_wave_mono(path: Path) -> List[float]:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        frames = wav.getnframes()
        data = wav.readframes(frames)
        samples = struct.unpack("<" + "h" * frames * channels, data)
        mono: List[float] = []
        for idx in range(frames):
            frame = samples[idx * channels : (idx + 1) * channels]
            mono.append(sum(frame) / (channels * 32768.0))
        return mono


def compute_bss_metrics(references: th.Tensor, estimates: th.Tensor) -> Dict[str, List[float]]:
    eps = 1e-8
    scores = {"sdr": [], "sir": [], "sar": [], "isr": []}
    num_sources = references.shape[0]
    for idx in range(num_sources):
        target = references[idx]
        estimate = estimates[idx]
        target_energy = (target ** 2).sum()
        proj_target = (estimate * target).sum() / (target_energy + eps) * target
        interference = th.zeros_like(estimate)
        for j in range(num_sources):
            if j == idx:
                continue
            ref_j = references[j]
            ref_energy = (ref_j ** 2).sum()
            interference += (estimate * ref_j).sum() / (ref_energy + eps) * ref_j
        artifact = estimate - proj_target - interference
        target_power = (proj_target ** 2).sum()
        interf_power = (interference ** 2).sum()
        artifact_power = (artifact ** 2).sum()
        scores["sdr"].append(float(10 * th.log10((target_power + eps) / (interf_power + artifact_power + eps))))
        scores["sir"].append(float(10 * th.log10((target_power + eps) / (interf_power + eps))))
        scores["sar"].append(float(10 * th.log10(((target_power + interf_power) + eps) / (artifact_power + eps))))
        scores["isr"].append(scores["sdr"][-1])
    return scores


def evaluate_track(separator: Separator, track_dir: Path, metrics: Iterable[str]) -> Dict[str, Dict[str, float]]:
    if th is None:
        raise RuntimeError("torch is required for model-based evaluation")
    mixture_path = track_dir / "mixture.wav"
    mix = _load_audio_torch(mixture_path, separator.samplerate)
    try:
        device = next(separator.model.parameters()).device
    except StopIteration:
        device = th.device(getattr(separator, "_device", "cpu"))
    mix = mix.to(device=device, dtype=th.float32)
    _, predictions = separator.separate_tensor(mix, separator.samplerate)

    references: List[th.Tensor] = []
    estimates: List[th.Tensor] = []
    stems: List[str] = []

    for stem in separator.model.sources:
        target_path = track_dir / f"{stem}.wav"
        if not target_path.exists():
            continue
        ref = _load_audio_torch(target_path, separator.samplerate).to(mix)
        references.append(ref.cpu())
        estimates.append(predictions[stem].cpu())
        stems.append(stem)

    refs_t = th.stack(references).to(th.float32)
    ests_t = th.stack(estimates).to(th.float32)

    results: Dict[str, Dict[str, float]] = {stem: {} for stem in stems}
    nsdr = new_sdr(refs_t.unsqueeze(0), ests_t.unsqueeze(0))[0]
    for stem, value in zip(stems, nsdr.tolist()):
        results[stem]["nsdr"] = value

    if metrics:
        if museval is not None and np is not None:
            references_np = refs_t.cpu().numpy().transpose(0, 2, 1)
            estimates_np = ests_t.cpu().numpy().transpose(0, 2, 1)
            sdr, isr, sir, sar, *_ = museval.metrics.bss_eval(
                references_np, estimates_np, compute_permutation=False, window=None, hop=None,
                framewise_filters=False, bsseval_sources_version=False
            )
            metric_map = {"sdr": sdr, "sir": sir, "sar": sar, "isr": isr}
            for metric in metrics:
                if metric not in metric_map:
                    continue
                for idx, stem in enumerate(stems):
                    values = metric_map[metric][idx]
                    results[stem][metric] = float(np.nanmean(values))
        else:
            fallback = compute_bss_metrics(refs_t, ests_t)
            for metric in metrics:
                if metric not in fallback:
                    continue
                for stem, value in zip(stems, fallback[metric]):
                    results[stem][metric] = value
    return results


def _snr(reference: List[float], estimate: List[float]) -> float:
    eps = 1e-10
    error = [(e - r) for e, r in zip(estimate, reference)]
    signal = sum(r * r for r in reference) + eps
    noise = sum(e * e for e in error) + eps
    return 10 * math.log10(signal / noise)


def evaluate_offline(track_dir: Path, metrics: Iterable[str]) -> Dict[str, Dict[str, float]]:
    mixture = _read_wave_mono(track_dir / "mixture.wav")
    stems = sorted(p.stem for p in track_dir.glob("*.wav") if p.stem != "mixture")
    results: Dict[str, Dict[str, float]] = {}
    for stem in stems:
        ref = _read_wave_mono(track_dir / f"{stem}.wav")
        length = min(len(mixture), len(ref))
        ref = ref[:length]
        estimate = ref[:]  # identity prediction for offline smoke tests
        sdr = _snr(ref, estimate)
        scores = {metric: sdr for metric in metrics}
        scores["nsdr"] = sdr
        results[stem] = scores
    return results


def summarise(all_results: Dict[str, Dict[str, Dict[str, float]]], metrics: Iterable[str]) -> Dict[str, Dict[str, float]]:
    metrics = list(metrics) + ["nsdr"]
    summary: Dict[str, Dict[str, float]] = {}
    for metric in metrics:
        per_metric: Dict[str, List[float]] = {}
        for track in all_results.values():
            for stem, values in track.items():
                if metric not in values:
                    continue
                per_metric.setdefault(stem, []).append(values[metric])
        if not per_metric:
            continue
        summary[metric] = {stem: sum(vals) / len(vals) for stem, vals in per_metric.items()}
        summary[metric + "_std"] = {
            stem: math.sqrt(sum((v - summary[metric][stem]) ** 2 for v in vals) / len(vals))
            for stem, vals in per_metric.items()
        }
        means = list(summary[metric].values())
        stds = list(summary[metric + "_std"].values())
        summary[metric]["avg"] = sum(means) / len(means)
        summary[metric + "_std"]["avg"] = sum(stds) / len(stds)
    return summary


def write_csv(path: Path, all_results: Dict[str, Dict[str, Dict[str, float]]]) -> None:
    rows = []
    for track, stems in all_results.items():
        for stem, metrics in stems.items():
            for metric, value in metrics.items():
                rows.append({"track": track, "stem": stem, "metric": metric, "value": value})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["track", "stem", "metric", "value"])
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, summary: Dict[str, Dict[str, float]], config: Dict[str, object]) -> None:
    lines = ["# Baseline Evaluation", ""]
    lines.append("```json")
    lines.append(json.dumps(config, indent=2))
    lines.append("```")
    lines.append("")
    for metric, values in summary.items():
        if metric.endswith("_std"):
            continue
        stds = summary.get(metric + "_std", {})
        stems = sorted(values.keys())
        lines.append(f"## {metric.upper()}")
        lines.append("| stem | mean | std |")
        lines.append("|---|---|---|")
        for stem in stems:
            if stem == "avg":
                continue
            std = stds.get(stem, float("nan"))
            lines.append(f"| {stem} | {values[stem]:.4f} | {std:.4f} |")
        if "avg" in values:
            avg_std = stds.get("avg", float("nan"))
            lines.append(f"| avg | {values['avg']:.4f} | {avg_std:.4f} |")
        lines.append("")
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline evaluation harness")
    parser.add_argument("--model", default="htdemucs", help="Model name to evaluate")
    parser.add_argument("--subset", default="test_small", help="Subset folder inside dataset root")
    parser.add_argument("--dataset-root", default="tests/fixtures/audio", help="Root dataset directory")
    parser.add_argument("--out", type=Path, required=True, help="Output directory for results")
    parser.add_argument("--metrics", nargs="*", default=["sdr", "sir", "sar", "isr"],
                        help="Metrics to compute (sdr/sir/sar/isr)")
    parser.add_argument("--tracks", type=int, default=20, help="Number of tracks to evaluate")
    parser.add_argument("--device", default="auto", help="Evaluation device: auto/cpu/cuda")
    parser.add_argument("--tta-flip", action="store_true", help="Enable TTA flipping during eval")
    parser.add_argument("--tta-aggregate", choices=["mean", "median"], default="mean")
    parser.add_argument("--transition-power", type=float, default=1.0)
    parser.add_argument("--offline", action="store_true", help="Skip model loading and reuse ground truth stems")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    if ensure_test_small_dataset is not None and args.subset == "test_small":
        ensure_test_small_dataset(dataset_root)
    subset_dir = dataset_root / args.subset
    if not subset_dir.exists():
        raise FileNotFoundError(f"Subset directory {subset_dir} does not exist")

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    tracks = sorted([p for p in subset_dir.iterdir() if p.is_dir()])
    if args.tracks:
        tracks = tracks[:args.tracks]

    if not args.offline and th is None:
        raise RuntimeError("torch is required unless --offline is provided")

    separator: Optional[Separator] = None
    if not args.offline:
        device = "cuda" if args.device == "auto" and th.cuda.is_available() else args.device
        separator = Separator(model=args.model,
                              device=device,
                              shifts=1,
                              overlap=0.25,
                              tta_flips=args.tta_flip,
                              tta_aggregate=args.tta_aggregate,
                              transition_power=args.transition_power,
                              progress=False)

    all_results: Dict[str, Dict[str, Dict[str, float]]] = {}
    for track in tracks:
        if args.offline:
            results = evaluate_offline(track, args.metrics)
        else:
            results = evaluate_track(separator, track, args.metrics)
        all_results[track.name] = results

    csv_path = out_dir / "metrics.csv"
    write_csv(csv_path, all_results)

    summary = summarise(all_results, args.metrics)
    summary_path = out_dir / "summary.md"
    write_summary(summary_path, summary, {
        "model": args.model,
        "subset": args.subset,
        "metrics": args.metrics,
        "tta_flip": args.tta_flip,
        "tta_aggregate": args.tta_aggregate,
        "transition_power": args.transition_power,
        "offline": args.offline,
    })

    print(f"Wrote metrics to {csv_path}")
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
