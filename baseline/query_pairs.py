"""Training pairs for a query-level verifier (model-free, unit-tested).

For each fitting clip: positive queries (the clip is in the query's relevant set), near-miss negatives (every parsed atom
of the query is present in the clip, yet the clip is not relevant: relation, order or agent binding is wrong) and random
negatives. Only fitting queries and fitting clips are used.
"""
from __future__ import annotations

import numpy as np

VERIFY = 'Does this video match the following description? "{}"'


def verifier_question(text):
    return VERIFY.format(text)


def clip_query_pools(fit_query_ids, qrels, parsed, truth, vids, fit):
    """{clip index: (positives, near_misses)} over fitting clips; `truth[atom]` is a bool array aligned with `vids`."""
    fit_set = set(int(i) for i in fit); index = {v: i for i, v in enumerate(vids)}
    pos = {i: [] for i in fit_set}; near = {i: [] for i in fit_set}
    for q in fit_query_ids:
        relevant = {index[v] for v in qrels[q]['relevant_video_ids'] if v in index} & fit_set
        for i in relevant:
            pos[i].append(q)
        atoms = parsed.get(q)
        if atoms and all(a in truth for a in atoms):
            present = np.logical_and.reduce([truth[a] for a in atoms])
            for i in np.flatnonzero(present):
                if int(i) in fit_set and int(i) not in relevant:
                    near[int(i)].append(q)
    return {i: (pos[i], near[i]) for i in fit_set}


def sample_queries(pools, i, all_queries, relevant_of, rng, count):
    """(query id, label) list: up to half positives, then up to a quarter near misses, then random negatives."""
    positives, near = pools[i]
    def take(items, k):
        k = min(k, len(items))
        return [items[j] for j in rng.choice(len(items), k, replace=False)] if k else []
    chosen = [(q, 1) for q in take(positives, count // 2)]
    chosen += [(q, 0) for q in take(near, max(1, count // 4) if count > 1 else 0) if len(chosen) < count]
    tries = 0
    while len(chosen) < count and tries < 50 * count:
        q = all_queries[int(rng.integers(len(all_queries)))]; tries += 1
        if i not in relevant_of[q] and q not in {c for c, _ in chosen}:
            chosen.append((q, 0))
    return chosen[:count]
