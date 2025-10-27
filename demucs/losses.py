"""Loss helpers for advanced training recipes."""

from dataclasses import dataclass
from typing import Iterable, List, Optional

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class MRSTFTConfig:
    fft_sizes: Iterable[int]
    hop: int
    win_length: Optional[int] = None
    power: float = 1.0
    weight: float = 1.0


class MultiResolutionSTFTLoss(nn.Module):
    """Approximate multi-resolution STFT loss combining spectral convergence and magnitude."""

    def __init__(self, fft_sizes: Iterable[int], hop: int, win_length: Optional[int] = None,
                 power: float = 1.0):
        super().__init__()
        self.fft_sizes: List[int] = list(fft_sizes)
        if not self.fft_sizes:
            raise ValueError("fft_sizes must contain at least one value")
        self.hop = hop
        self.win_length = win_length
        self.power = power
        self.register_buffer("_dummy", torch.zeros(1))
        self._windows = {}

    def _window(self, size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (size, device, dtype)
        if key not in self._windows:
            win_length = self.win_length or size
            window = torch.hann_window(win_length, device=device, dtype=dtype)
            self._windows[key] = window
        return self._windows[key]

    def forward(self, estimate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if estimate.dim() != target.dim():
            raise ValueError("estimate and target must have the same dimensionality")
        # collapse non-batch dimensions
        estimate = estimate.reshape(-1, estimate.shape[-1])
        target = target.reshape(-1, target.shape[-1])
        losses = []
        for fft_size in self.fft_sizes:
            hop = self.hop
            window = self._window(fft_size, estimate.device, estimate.dtype)
            est_stft = torch.stft(estimate, n_fft=fft_size, hop_length=hop,
                                  win_length=window.numel(), window=window,
                                  return_complex=True)
            tgt_stft = torch.stft(target, n_fft=fft_size, hop_length=hop,
                                  win_length=window.numel(), window=window,
                                  return_complex=True)
            if self.power != 1.0:
                est_mag = est_stft.abs().pow(self.power)
                tgt_mag = tgt_stft.abs().pow(self.power)
            else:
                est_mag = est_stft.abs()
                tgt_mag = tgt_stft.abs()
            spectral_convergence = ((tgt_mag - est_mag).norm(p=2, dim=-1) /
                                    (tgt_mag.norm(p=2, dim=-1) + 1e-8)).mean()
            magnitude = F.l1_loss(est_mag, tgt_mag)
            losses.append(spectral_convergence + magnitude)
        total = torch.stack(losses).mean()
        return total


class SISDRLoss(nn.Module):
    """Scale-invariant SDR loss (negative SI-SDR)."""

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, estimate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if estimate.dim() != target.dim():
            raise ValueError("estimate and target must have the same dimensionality")
        batch_dims = list(range(estimate.dim() - 1))
        dot = (estimate * target).sum(dim=-1, keepdim=True)
        target_energy = (target ** 2).sum(dim=-1, keepdim=True) + self.eps
        scale = dot / target_energy
        projection = scale * target
        noise = estimate - projection
        sisdr = ((projection ** 2).sum(dim=-1) + self.eps) / ((noise ** 2).sum(dim=-1) + self.eps)
        sisdr = -10 * torch.log10(sisdr + self.eps)
        return sisdr.mean(dim=tuple(d for d in range(len(batch_dims))))
