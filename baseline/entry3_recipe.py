"""Audit-confirmed entry-3 recipe (2026-10-21): t4 atoms late-fused with fine-tuned Qwen3.8-27B atom answers, rank-fused
at weight 0.8 with the submitted raw/head ranking.

t4 = pooled + motion2 + window linear probes (late per-block fusion), plus a temporal CNN on the motion series for ego
atoms (calibrated by a logistic fit on inner cross-fitted logits), with parser v2. Qwen3.8 probabilities are late-fused
into every atom they cover. Shared by research verification and the offline package so both use identical scoring code.
No annotations, labels or known-video features are needed at inference.
"""
from __future__ import annotations

import numpy as np

from .egomotion import clip_vector, decode_gray, grid_flow, signals
from .query_parser import Unparsed
from .query_parser_v2 import translate as translate_v2
from .query_struct import atoms, tree
from .structured_recipe import BLOCKS, MOTION, Probes, fuse

FUSION_WEIGHT = 0.8


def motion_series(path):
    """Per-frame motion signals (the research 'motion2' series) and their clip vector (the 'motion2' probe block)."""
    series = signals(grid_flow(decode_gray(path, MOTION['fps'], MOTION['width']), MOTION['rows'], MOTION['cols']))
    if not np.isfinite(series).all():
        raise ValueError('Nonfinite motion signal')
    return series.astype(np.float32), clip_vector(series, MOTION['bins'])


def query_atoms_v2(text):
    try:
        return sorted(set(atoms(tree(translate_v2(text)))))
    except Unparsed:
        return None


def late(probs):
    logit = np.mean([np.log(p / (1 - p)) for p in probs], axis=0)
    return np.clip(1 / (1 + np.exp(-logit)), 1e-4, 1 - 1e-4)


def motion_net(channels, outputs):
    """The research temporal CNN (scripts/structured_v2.py motionnet_logits): two dilated conv layers, max+mean pooling."""
    import torch
    conv = torch.nn.Sequential(torch.nn.Conv1d(channels, 32, 5, padding=2), torch.nn.ReLU(),
                               torch.nn.Conv1d(32, 32, 5, padding=4, dilation=2), torch.nn.ReLU())
    return conv, torch.nn.Linear(64, outputs)


class EgoNet:
    """Trained motion CNN plus per-atom logistic calibration of its logits (prior where an atom had too few positives)."""

    def __init__(self, data):
        import torch
        self.atoms = [str(a) for a in data['ego_atoms']]; self.length = int(data['ego_series_length'])
        self.mean, self.std = data['ego_mean'].astype(np.float32), data['ego_std'].astype(np.float32)
        self.conv, self.head = motion_net(self.mean.shape[0], len(self.atoms))
        with torch.no_grad():
            for i in (0, 2):
                self.conv[i].weight.copy_(torch.as_tensor(data[f'ego_conv{i}_weight'])); self.conv[i].bias.copy_(torch.as_tensor(data[f'ego_conv{i}_bias']))
            self.head.weight.copy_(torch.as_tensor(data['ego_head_weight'])); self.head.bias.copy_(torch.as_tensor(data['ego_head_bias']))
        self.cal = {k: data[f'ego_cal_{k}'] for k in ('mean', 'std', 'weight', 'bias', 'prior', 'is_prior')}

    def logits(self, series):
        """series: list of [T, C] arrays; clips longer than the research length are truncated to it, as in research."""
        import torch
        out = []
        with torch.no_grad():
            for s in series:
                x = torch.as_tensor((np.asarray(s, np.float32)[:self.length] - self.mean) / self.std).T[None]
                h = self.conv(x); out.append(self.head(torch.cat([h.amax(2), h.mean(2)], 1))[0].numpy())
        return np.stack(out)

    def probabilities(self, series):
        z = (self.logits(series).astype(np.float64) - self.cal['mean']) / self.cal['std']; out = {}
        for j, a in enumerate(self.atoms):
            if self.cal['is_prior'][j]:
                out[a] = np.full(len(z), float(self.cal['prior'][j]))
            else:
                out[a] = np.clip(1 / (1 + np.exp(-(z @ self.cal['weight'][j] + self.cal['bias'][j]))), 1e-4, 1 - 1e-4)
        return out


def t4_probabilities(path, features, series):
    """Atom -> [videos] probabilities: block probes for every atom, the ego CNN late-fused into the ego atoms it covers."""
    probes = Probes(path); data = np.load(path, allow_pickle=False)
    block = probes.probabilities(features); base = {a: block[:, j] for j, a in enumerate(probes.atoms)}
    ego = EgoNet(data).probabilities(series)
    return {a: (late([p, ego[a]]) if a in ego else p) for a, p in base.items()}


def with_vlm(base, vlm):
    return {a: (late([p, vlm[a]]) if a in vlm else p) for a, p in base.items()}


def structured_scores(texts, prob):
    """[queries, videos] summed atom log-probabilities over parser-v2 atoms with a probability, and the parsed mask."""
    n = len(next(iter(prob.values()))); scores = np.zeros((len(texts), n)); mask = np.zeros(len(texts), bool)
    for i, text in enumerate(texts):
        known = [a for a in (query_atoms_v2(text) or []) if a in prob]
        if known:
            scores[i] = np.sum([np.log(prob[a]) for a in known], axis=0); mask[i] = True
    return scores, mask


__all__ = ['BLOCKS', 'FUSION_WEIGHT', 'EgoNet', 'fuse', 'late', 'motion_net', 'motion_series', 'structured_scores',
           't4_probabilities', 'with_vlm']
