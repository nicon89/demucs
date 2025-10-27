# Experiments & Audit

## Repository audit

### Model variants
- **HTDemucs family** (`demucs/htdemucs.py`): hybrid time-frequency model supporting 4-source and multi-source settings via `sources` argument. Variants like `htdemucs`, `htdemucs_ft`, and optional bags are exposed through `demucs.pretrained`.
- **Hybrid Demucs (HDemucs)** (`demucs/hdemucs.py`): multi-resolution STFT front-end with optional Wiener filtering.
- **Demucs v3** (`demucs/demucs.py`): convolutional encoder/decoder with LSTM bottleneck.
- **MDX-style models** (configs under `conf/variant/`): use transformer heads for frequency modelling.

### Training pipeline
- Hydra config root at `conf/config.yaml` with dataset definitions under `conf/dset/` and variants in `conf/variant/`.
- Entry-point `demucs/train.py` builds datasets (`demucs/wav.py`) and solver (`demucs/solver.py`).
- Augmentations implemented in `demucs/augment.py` (remix, repitch, scale, channel flip).
- Loss selection controlled via `optim.loss` (`l1` or `mse`); evaluation uses `demucs/evaluate.py` with museval.

### Evaluation utilities
- `tools/test_pretrained.py` evaluates pre-trained checkpoints via Dora experiments.
- `demucs/evaluate.py` provides SDR/SIR/SAR computation (BSSEval v4) and MDX `new_sdr` metric.
- `demucs/separate.py` is the CLI front-end for inference, relying on `demucs/api.Separator`.

### Dataset support
- MusDB HQ + custom WAV directories are configured in `conf/dset/musdb44.yaml` etc.
- Metadata building handled by `demucs/wav.build_metadata` with train/valid/test splits.

### Dependencies & environment
- Minimum dependencies in `requirements_minimal.txt`: Python ≥3.9, `torch` (CPU/CUDA wheels), `torchaudio`, `museval`, `dora-search`, `tqdm`, `soundfile`.
- CUDA support via PyTorch wheels; `environment-cuda.yml` pins CUDA 11.7; CPU fallback via `environment-cpu.yml`.

### Fork divergences vs upstream
- Pre-existing configs under `conf/variant/` diverge through dataset balancing and optimizer tweaks.
- Custom CLI flags for stem selection and bag-of-model support in `demucs/separate.py`.
- Training loop extends upstream with Dora experiment tracking (`dora` integration) and optional EMA.
- Loss weighting via `weights` argument with per-source logging in `demucs/solver.py`.

## Planned improvements
- Reproducible evaluation scripts (`tools/eval_baseline.py`).
- Accuracy boosts through TTA, MR-STFT loss, augmentations, and instrument-aware weighting.
- Expanded instrument coverage (strings, keys/organ, brass) with new configs in `configs/`.
- Benchmarks + plots (`tools/benchmark.py`) and deterministic inference switches.


## Implemented improvements
- Added `tools/eval_baseline.py` with offline fallback mode for CI smoke tests and full torch/museval support when dependencies are installed.
- Introduced MR-STFT + SI-SDR hybrid loss wiring in `demucs/solver.py` with configurable augmentations (EQ tilt, pitch, stretch, noise).
- Extended CLI (`demucs.separate`) with `--tta-flip`, `--tta-aggregate`, `--transition-power`, and `--deterministic` flags.
- Registered new local model variants (`htdemucs_6s`, `htdemucs_6s_mrstft`, `htdemucs_6s_tta`, `htdemucs_6s_mrstft_tta`, `htdemucs_keys_strings_7s`, `htdemucs_8s`).
- Added benchmarking helper `tools/benchmark.py` and smoke dataset under `tests/fixtures/audio/`.
- Documented fork updates in `README.md` and generated baseline metrics at `runs/baseline_ht6s/`.

## Recommended recipes
- **Balanced 6 stems**: `python -m demucs.separate -n htdemucs_6s_mrstft_tta --tta-flip --tta-aggregate median --transition-power 1.5 my_track.wav`
- **Keys + strings expansion**: `python -m demucs.separate -n htdemucs_8s --tta-flip --tta-aggregate median track.wav`
- **Quick benchmark**: `python tools/eval_baseline.py --model htdemucs_6s --subset test_small --out runs/baseline_ht6s --offline` followed by `python tools/benchmark.py runs/baseline_ht6s --names baseline --output runs/baseline_ht6s/benchmark.md`.
