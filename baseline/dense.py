"""Deterministic full-corpus vector ranking and reciprocal-rank fusion."""
from __future__ import annotations

import numpy as np


def normalize(vectors):
    vectors = np.asarray(vectors, dtype=np.float32)
    if not np.isfinite(vectors).all():
        raise ValueError("Nonfinite embedding")
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("Zero embedding")
    return vectors / norms


def rank_scores(video_ids, scores):
    if len(set(video_ids)) != len(video_ids) or len(scores) != len(video_ids):
        raise ValueError("Ranking requires unique IDs matching the score vector")
    if not np.isfinite(scores).all():
        raise ValueError("Nonfinite similarity")
    # Stable sorting breaks ties by sorted video ID, not input order.
    ids = np.asarray(video_ids)
    return ids[np.lexsort((ids, -np.asarray(scores)))].tolist()


def fuse(video_ids, dense, lexical, weight, rank_constant=60):
    if not 0 <= weight <= 1 or rank_constant <= 0:
        raise ValueError("Invalid fusion settings")
    scores = dict.fromkeys(video_ids, 0.0)
    for ranking, coefficient in ((dense, weight), (lexical, 1 - weight)):
        if len(set(ranking)) != len(ranking) or set(ranking) - set(video_ids):
            raise ValueError("Invalid ranking for fusion")
        for position, video_id in enumerate(ranking, 1):
            scores[video_id] += coefficient / (rank_constant + position)
    return sorted(scores, key=lambda video_id: (-scores[video_id], video_id))
