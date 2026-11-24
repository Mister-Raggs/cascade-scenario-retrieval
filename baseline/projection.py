"""A feature-based residual adapter and positive-aware ranking loss (no ID parameters)."""
from __future__ import annotations
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class SharedProjection(nn.Module):
    def __init__(self, dimension, rank=128):
        super().__init__()
        self.down = nn.Linear(dimension, rank, bias=False)
        self.up = nn.Linear(rank, dimension, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, features):
        return F.normalize(features + self.up(F.gelu(self.down(features))), dim=-1)


def positive_aware_loss(scores, positive):
    """Average each positive's loss against ALL nonpositives, never against other positives.

    scores already include temperature. Rows must have at least one positive.
    With no negatives the loss is zero; no sampled false-negative positive pairs occur.
    """
    if scores.shape != positive.shape or positive.dtype != torch.bool or scores.ndim != 2:
        raise ValueError("Scores and boolean positives must share a 2D shape")
    counts = positive.sum(dim=1)
    if not torch.all(counts > 0) or not torch.isfinite(scores).all():
        raise ValueError("Each fitted query needs positive relevance and finite scores")
    has_negative = (~positive).any(dim=1)
    if not has_negative.any():
        return scores.sum() * 0
    fraction = has_negative.float().mean()
    scores, positive, counts = scores[has_negative], positive[has_negative], counts[has_negative]
    negatives = scores.masked_fill(positive, -torch.inf)
    normalizer = torch.logsumexp(negatives, dim=1, keepdim=True)
    terms = F.softplus(normalizer - scores).masked_fill(~positive, 0)
    return (terms.sum(dim=1) / counts).mean() * fraction


def heldout_partition(count, seed):
    if count < 10:
        raise ValueError("At least ten items required for fit/dev/audit partition")
    shuffled = np.random.default_rng(seed).permutation(count)
    groups = np.split(shuffled, [int(count * .8), int(count * .9)])
    return {name: np.sort(group) for name, group in zip(("fit", "dev", "audit"), groups, strict=True)}
