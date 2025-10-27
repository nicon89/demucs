"""Utility helpers shared across separator backends."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Mapping, Tuple

import torch as th

CANONICAL_STEMS: Tuple[str, ...] = ("vocals", "drums", "bass", "other")


@dataclass
class CanonicalSeparation:
    """Container returned by :func:`ensure_canonical_stems`."""

    mixture: th.Tensor
    stems: Dict[str, th.Tensor]


def ensure_canonical_stems(
    mixture: th.Tensor,
    stems: Mapping[str, th.Tensor],
    *,
    logger: logging.Logger,
) -> CanonicalSeparation:
    """Ensure that the separator output aligns with :data:`CANONICAL_STEMS`.

    Missing stems are filled with the residual ``other`` stem.  If the residual stem
    does not exist the mixture is used instead so that downstream tooling can still
    operate on the result.  A warning is emitted for each missing stem so that users
    understand that the backend could not faithfully reproduce the canonical layout.
    """

    resolved: Dict[str, th.Tensor] = {}
    available = dict(stems)
    for stem in CANONICAL_STEMS:
        if stem in available:
            resolved[stem] = available.pop(stem)
            continue
        replacement_stem = resolved.get("other") or available.get("other")
        if replacement_stem is None:
            replacement_stem = mixture
        logger.warning(
            "Stem '%s' is not provided by the backend. Using the 'other' stem instead.",
            stem,
        )
        resolved[stem] = replacement_stem
    return CanonicalSeparation(mixture=mixture, stems=resolved)
