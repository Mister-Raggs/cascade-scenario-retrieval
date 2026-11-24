"""Entry-4 linear learned scorer (audit-confirmed 2026-10-22: audit RP 0.2286 versus 0.2021 for the entry-3 recipe).

21 query-clip features, then one linear layer (scripts/learned_scorer.py, arm "linear"); the score is the final ranking.
Per atom slot (the query's parsed atoms, sorted, at most two): the combined (t4 + Qwen3.8 late fusion), t4 and Qwen3.8
log-odds divided by 5, a Qwen-available flag and the family one-hot; then the within-query z-scores of the raw and head
similarities and of the summed atom log-probabilities (0 for unparsed queries), the parsed flag and the atom count / 2.
With verifier answers (the bundle, audit-confirmed 2026-10-22: audit RP 0.2480), two more features follow: the
whole-query verifier log-odds z-scored within each query's top-30 candidates (others at the candidates' minimum minus 1)
and an in-candidates flag (23 features). Entry 5 adds, before the verifier features, each query's summed property-atom
log-probability (z within the query, 0 without property atoms) and a has-property flag (25 features), the order of
scripts/learned_scorer.py pair_features. NumPy only, shared by research parity checks and the offline package.
"""
from __future__ import annotations

import numpy as np

from .entry3_recipe import query_atoms_v2

FAMILIES = ('agent', 'light', 'obj', 'ego')
DIMENSION = 2 * (4 + len(FAMILIES)) + 5
VERIFIER_DIMENSION = DIMENSION + 2
PROPERTY_DIMENSION = VERIFIER_DIMENSION + 2


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def sigmoid(z):
    return np.clip(1 / (1 + np.exp(-z)), 1e-4, 1 - 1e-4)


def zrow(x):
    return (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-6)


def property_score(texts, props, prop_parse):
    """Summed log-probability of each query's property atoms over clips, and the has-property mask; props maps a property
    atom to [clips] probabilities and prop_parse maps a query text to its property atoms."""
    n_v = len(next(iter(props.values()))) if props else 0
    score = np.zeros((len(texts), n_v)); mask = np.zeros(len(texts), bool)
    for i, text in enumerate(texts):
        known = [a for a in prop_parse(text) if a in props]
        if known:
            score[i] = np.sum([np.log(np.clip(props[a], 1e-4, 1 - 1e-4)) for a in known], 0); mask[i] = True
    return score, mask


def features(texts, t4, vlm, raw, sub, parse=query_atoms_v2, verify=None, props=None, prop_parse=None):
    """[queries, clips, 21, 23 or 25]; t4 and vlm map atom -> [clips] probabilities (t4 includes the ego CNN late fusion);
    verify is [queries, clips] verifier log-odds, NaN outside each query's candidates; props maps a property atom to
    [clips] probabilities (entry 5)."""
    t4_logit = {a: logit(p) for a, p in t4.items()}; vlm_logit = {a: logit(p) for a, p in vlm.items()}
    comb = {a: (logit(sigmoid((t + vlm_logit[a]) / 2)) if a in vlm_logit else logit(sigmoid(t))) for a, t in t4_logit.items()}
    n_q, n_v = raw.shape; slot = np.zeros((n_q, n_v, 2, 4 + len(FAMILIES)), np.float32)
    summed = np.zeros((n_q, n_v)); parsed = np.zeros(n_q, bool); count = np.zeros(n_q, np.float32)
    for i, text in enumerate(texts):
        known = sorted(a for a in (parse(text) or []) if a in comb)
        if known:
            summed[i] = np.sum([np.log(sigmoid(comb[a])) for a in known], axis=0); parsed[i] = True
        known = known[:2]; count[i] = len(known)
        for k, a in enumerate(known):
            slot[i, :, k, 0] = comb[a]; slot[i, :, k, 1] = t4_logit[a]
            if a in vlm_logit:
                slot[i, :, k, 2] = vlm_logit[a]; slot[i, :, k, 3] = 1
            slot[i, :, k, 4 + FAMILIES.index(a.split('|')[0])] = 1
    per_q = np.stack([zrow(raw), zrow(sub), np.where(parsed[:, None], zrow(np.where(parsed[:, None], summed, 0)), 0),
                      np.broadcast_to(parsed[:, None], raw.shape).astype(np.float32), np.broadcast_to(count[:, None], raw.shape) / 2], -1)
    parts = [slot.reshape(n_q, n_v, -1) / 5, per_q]
    if props is not None:
        ps, pm = property_score(texts, props, prop_parse)
        parts.append(np.stack([np.where(pm[:, None], zrow(np.where(pm[:, None], ps, 0)), 0),
                               np.broadcast_to(pm[:, None], raw.shape).astype(np.float32)], -1))
    if verify is not None:
        cand = np.isfinite(verify); masked = np.where(cand, verify, np.nan)
        z = (verify - np.nanmean(masked, 1, keepdims=True)) / (np.nanstd(masked, 1, keepdims=True) + 1e-6)
        lowest = np.nanmin(np.where(cand, z, np.nan), 1, keepdims=True) - 1
        # A query left without candidates by the time guard gets neutral verifier features (never in research data).
        parts.append(np.nan_to_num(np.stack([np.where(cand, z, lowest), cand.astype(np.float32)], -1), nan=0.0))
    return np.concatenate(parts, -1).astype(np.float32), parsed


class LinearScorer:
    def __init__(self, path):
        data = np.load(path, allow_pickle=False)
        self.weight, self.bias = data['weight'].astype(np.float32), float(data['bias'])
        if self.weight.shape not in ((DIMENSION,), (VERIFIER_DIMENSION,), (PROPERTY_DIMENSION,)):
            raise ValueError('Linear scorer weights have the wrong dimension')

    def scores(self, x):
        return x @ self.weight + self.bias
