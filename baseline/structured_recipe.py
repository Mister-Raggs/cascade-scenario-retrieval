"""Validated structured recipe (2026-10-16): parser v1 atoms scored by late-fused linear probes on pooled,
motion2 and window features, rank-fused 50/50 with the submitted raw/head recipe.

Shared by research verification and the offline package so both use identical feature and scoring code.
No annotations, labels or known-video features are needed at inference.
"""
from __future__ import annotations

import numpy as np

from .egomotion import decode_gray, grid_flow, signals, clip_vector
from .native_ranking import blended_scores
from .query_parser import translate, Unparsed
from .query_struct import tree, atoms
from .windows import probe, window_bounds, window_times, decode_at

MOTION = {'fps': 8.0, 'width': 256, 'rows': 4, 'cols': 6, 'bins': 10}
WINDOWS = {'count': 5, 'frames': 4}
BLOCKS = ('pooled', 'motion2', 'windows')
FUSION_WEIGHT = 0.5


def motion_features(path):
    series = signals(grid_flow(decode_gray(path, MOTION['fps'], MOTION['width']), MOTION['rows'], MOTION['cols']))
    if not np.isfinite(series).all():
        raise ValueError('Nonfinite motion signal')
    return clip_vector(series, MOTION['bins'])


def window_items(path):
    """Source-timed frames for each window, in the NativeEncoder decode format (stamps, frames, duration, decoder)."""
    info = probe(path); items = []
    for w, (start, end) in enumerate(window_bounds(info['duration'], WINDOWS['count'])):
        frames, stamps, _ = decode_at(path, window_times(start, end, WINDOWS['frames']), info)
        items.append((stamps, frames, info['duration'], {'window': w, 'window_bounds': [start, end]}))
    return items


def window_feature(vectors):
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.shape[0] != WINDOWS['count']:
        raise ValueError('Expected one vector per window')
    return np.concatenate([vectors.mean(0), vectors.max(0)])


class Probes:
    """Per-block standardization plus logistic weights per atom; late fusion averages block log-odds."""

    def __init__(self, path):
        data = np.load(path, allow_pickle=False)
        self.atoms = [str(a) for a in data['atoms']]; self.index = {a: i for i, a in enumerate(self.atoms)}
        self.blocks = {b: (data[f'{b}_mean'], data[f'{b}_std'], data[f'{b}_weight'], data[f'{b}_bias']) for b in BLOCKS}
        self.prior = data['prior']; self.is_prior = data['is_prior'].astype(bool)

    def probabilities(self, features):
        """features: block -> [videos, dims]; returns [videos, atoms] probabilities."""
        logits = []
        for b in BLOCKS:
            mean, std, weight, bias = self.blocks[b]
            z = (np.asarray(features[b], dtype=np.float64) - mean) / std
            p = np.clip(1 / (1 + np.exp(-(z @ weight + bias))), 1e-4, 1 - 1e-4)
            p[:, self.is_prior] = np.clip(self.prior[self.is_prior], 1e-3, 1 - 1e-3)
            logits.append(np.log(p / (1 - p)))
        return np.clip(1 / (1 + np.exp(-np.mean(logits, axis=0))), 1e-4, 1 - 1e-4)


def query_atoms(text):
    try:
        return sorted(set(atoms(tree(translate(text)))))
    except Unparsed:
        return None


def structured_scores(texts, probabilities, probes):
    """[queries, videos] summed atom log-probabilities and a mask of queries with at least one known atom."""
    scores = np.zeros((len(texts), probabilities.shape[0])); mask = np.zeros(len(texts), bool)
    logp = np.log(probabilities)
    for i, text in enumerate(texts):
        known = [probes.index[a] for a in (query_atoms(text) or []) if a in probes.index]
        if known:
            scores[i] = logp[:, known].sum(1); mask[i] = True
    return scores, mask


def fuse(raw, submitted, structured, mask, weight=FUSION_WEIGHT):
    """Rank-fuse structured scores with the submitted recipe; queries without parsed atoms keep the submitted ranking."""
    fused = blended_scores(np.where(mask[:, None], structured, raw), submitted, weight)
    return np.where(mask[:, None], fused, submitted)
