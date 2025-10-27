# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Data augmentations.
"""

import random
import torch as th
from torch import nn
from torch.nn import functional as F


class Shift(nn.Module):
    """
    Randomly shift audio in time by up to `shift` samples.
    """
    def __init__(self, shift=8192, same=False):
        super().__init__()
        self.shift = shift
        self.same = same

    def forward(self, wav):
        batch, sources, channels, time = wav.size()
        length = time - self.shift
        if self.shift > 0:
            if not self.training:
                wav = wav[..., :length]
            else:
                srcs = 1 if self.same else sources
                offsets = th.randint(self.shift, [batch, srcs, 1, 1], device=wav.device)
                offsets = offsets.expand(-1, sources, channels, -1)
                indexes = th.arange(length, device=wav.device)
                wav = wav.gather(3, indexes + offsets)
        return wav


class FlipChannels(nn.Module):
    """
    Flip left-right channels.
    """
    def forward(self, wav):
        batch, sources, channels, time = wav.size()
        if self.training and wav.size(2) == 2:
            left = th.randint(2, (batch, sources, 1, 1), device=wav.device)
            left = left.expand(-1, -1, -1, time)
            right = 1 - left
            wav = th.cat([wav.gather(2, left), wav.gather(2, right)], dim=2)
        return wav


class FlipSign(nn.Module):
    """
    Random sign flip.
    """
    def forward(self, wav):
        batch, sources, channels, time = wav.size()
        if self.training:
            signs = th.randint(2, (batch, sources, 1, 1), device=wav.device, dtype=th.float32)
            wav = wav * (2 * signs - 1)
        return wav


class Remix(nn.Module):
    """
    Shuffle sources to make new mixes.
    """
    def __init__(self, proba=1, group_size=4):
        """
        Shuffle sources within one batch.
        Each batch is divided into groups of size `group_size` and shuffling is done within
        each group separatly. This allow to keep the same probability distribution no matter
        the number of GPUs. Without this grouping, using more GPUs would lead to a higher
        probability of keeping two sources from the same track together which can impact
        performance.
        """
        super().__init__()
        self.proba = proba
        self.group_size = group_size

    def forward(self, wav):
        batch, streams, channels, time = wav.size()
        device = wav.device

        if self.training and random.random() < self.proba:
            group_size = self.group_size or batch
            if batch % group_size != 0:
                raise ValueError(f"Batch size {batch} must be divisible by group size {group_size}")
            groups = batch // group_size
            wav = wav.view(groups, group_size, streams, channels, time)
            permutations = th.argsort(th.rand(groups, group_size, streams, 1, 1, device=device),
                                      dim=1)
            wav = wav.gather(1, permutations.expand(-1, -1, -1, channels, time))
            wav = wav.view(batch, streams, channels, time)
        return wav


class Scale(nn.Module):
    def __init__(self, proba=1., min=0.25, max=1.25):
        super().__init__()
        self.proba = proba
        self.min = min
        self.max = max

    def forward(self, wav):
        batch, streams, channels, time = wav.size()
        device = wav.device
        if self.training and random.random() < self.proba:
            scales = th.empty(batch, streams, 1, 1, device=device).uniform_(self.min, self.max)
            wav *= scales
        return wav


class EQTilt(nn.Module):
    """Applies a gentle spectral tilt implemented via FFT magnitudes."""

    def __init__(self, proba=0.0, gain=3.0):
        super().__init__()
        self.proba = proba
        self.gain = gain

    def forward(self, wav):
        if not self.training or random.random() >= self.proba:
            return wav
        batch_shape = wav.shape
        flat = wav.reshape(-1, batch_shape[-1])
        spectrum = th.fft.rfft(flat, dim=-1)
        freqs = th.linspace(0, 1, spectrum.shape[-1], device=wav.device, dtype=wav.dtype)
        direction = random.choice([-1.0, 1.0])
        magnitude = random.random() * (self.gain / 6.0)
        tilt = 1.0 + direction * magnitude * (freqs - 0.5)
        tilt = tilt.clamp(min=0.25)
        spectrum = spectrum * tilt
        tilted = th.fft.irfft(spectrum, n=batch_shape[-1], dim=-1)
        return tilted.reshape(batch_shape)


class PitchShift(nn.Module):
    """Approximate pitch shift via linear resampling."""

    def __init__(self, proba=0.0, cents=20):
        super().__init__()
        self.proba = proba
        self.cents = cents

    def forward(self, wav):
        if not self.training or random.random() >= self.proba or self.cents <= 0:
            return wav
        ratio = 2 ** (random.uniform(-self.cents, self.cents) / 1200.0)
        time = wav.shape[-1]
        target = max(8, int(time / ratio))
        flat = wav.reshape(-1, 1, time)
        stretched = F.interpolate(flat, size=target, mode='linear', align_corners=False)
        restored = F.interpolate(stretched, size=time, mode='linear', align_corners=False)
        return restored.reshape_as(wav)


class TimeStretch(nn.Module):
    """Random time stretching using linear interpolation."""

    def __init__(self, proba=0.0, min=0.97, max=1.03):
        super().__init__()
        self.proba = proba
        self.min = min
        self.max = max

    def forward(self, wav):
        if not self.training or random.random() >= self.proba:
            return wav
        ratio = random.uniform(self.min, self.max)
        time = wav.shape[-1]
        target = max(8, int(time * ratio))
        flat = wav.reshape(-1, 1, time)
        stretched = F.interpolate(flat, size=target, mode='linear', align_corners=False)
        restored = F.interpolate(stretched, size=time, mode='linear', align_corners=False)
        return restored.reshape_as(wav)


class MixWithNoise(nn.Module):
    """Adds low level Gaussian noise to the mixture."""

    def __init__(self, proba=0.0, snr=30.0):
        super().__init__()
        self.proba = proba
        self.snr = snr

    def forward(self, wav):
        if not self.training or random.random() >= self.proba:
            return wav
        power = wav.pow(2).mean(dim=-1, keepdim=True)
        target_power = power / (10 ** (self.snr / 10.0))
        noise = th.randn_like(wav) * (target_power.sqrt())
        return wav + noise
