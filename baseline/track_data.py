"""Model-free data preparation shared by the fine-tuned reranker (Track 2) and VLM atom predictor (Track 3).

`truth[atom]` is a boolean array aligned with `vids`; `fit` holds indices of fitting videos. All sampling uses
fitting videos only, so held-fold videos never appear in training pairs or questions.
"""
from __future__ import annotations

import numpy as np

from .structured_recipe import query_atoms


def near_miss_negatives(atom_list, truth, fit, exclude, rng, count):
    """Fitting-video indices satisfying some but not all of the atoms (hard compositional negatives)."""
    atom_list = [a for a in atom_list if a in truth]
    if not atom_list or count <= 0:
        return []
    have = np.stack([truth[a][fit] for a in atom_list], 1).sum(1)
    pool = [int(fit[i]) for i in np.flatnonzero((have > 0) & (have < len(atom_list))) if int(fit[i]) not in exclude]
    return [int(x) for x in rng.choice(pool, min(count, len(pool)), replace=False)] if pool else []


def reranker_pairs(queries, qrels, vids, fit, truth, rng, negatives=4, positives=2):
    """[(text, video_index, label)]: positives among fitting videos, half near-miss and half random negatives."""
    index = {v: i for i, v in enumerate(vids)}; fit_set = {int(i) for i in fit}; pairs = []
    for qid, text in queries:
        pos = sorted({index[v] for v in qrels[qid] if v in index} & fit_set)
        if not pos:
            continue
        excluded = {index[v] for v in qrels[qid] if v in index}
        for i in rng.choice(pos, min(positives, len(pos)), replace=False):
            pairs.append((text, int(i), 1))
        hard = near_miss_negatives(query_atoms(text) or [], truth, fit, excluded, rng, negatives // 2)
        pool = [int(i) for i in fit if int(i) not in excluded and int(i) not in hard]
        rest = [int(i) for i in rng.choice(pool, negatives - len(hard), replace=False)]
        pairs += [(text, i, 0) for i in hard + rest]
    return pairs


def atom_question_batches(fit, truth, atoms, rng, questions=8):
    """One example per fitting video: (video_index, [(atom, label)]) balanced between true and false atoms."""
    batches = []
    for i in fit:
        yes = [a for a in atoms if truth[a][i]]; no = [a for a in atoms if not truth[a][i]]
        k = min(questions // 2, len(yes))
        chosen = [(a, 1) for a in rng.choice(yes, k, replace=False)] if k else []
        chosen += [(a, 0) for a in rng.choice(no, min(questions - k, len(no)), replace=False)]
        batches.append((int(i), [(str(a), int(l)) for a, l in chosen]))
    return batches
