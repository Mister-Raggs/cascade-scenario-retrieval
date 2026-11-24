"""Atom vocabulary from parsed training queries and clip/window atom truth from annotation events."""
from __future__ import annotations

import numpy as np

from .query_parser import translate, Unparsed
from .query_struct import tree, atoms, atom_present


def query_atoms(texts):
    vocabulary = set()
    for text in texts:
        try:
            vocabulary.update(atoms(tree(translate(text))))
        except Unparsed:
            continue
    return sorted(vocabulary)


def clip_atom_matrix(events, video_ids, vocabulary):
    labels = {v: {e[0] for e in events[v]['events']} for v in video_ids}
    return np.asarray([[atom_present(a, labels[v]) for a in vocabulary] for v in video_ids], dtype=bool)


def window_atom_matrix(events, video_ids, bounds, vocabulary):
    """[videos, windows, atoms]: an atom is true in a window if its supporting events overlap that window."""
    out = []
    for v in video_ids:
        rows = []
        for start, end in bounds[v]:
            active = {label for label, s, e in events[v]['events'] if s < end and e > start}
            rows.append([atom_present(a, active) for a in vocabulary])
        out.append(rows)
    return np.asarray(out, dtype=bool)
