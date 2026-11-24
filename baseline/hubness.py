"""Test-time score normalization across a query bank (hubness correction); no labels or training."""
from __future__ import annotations

import numpy as np


def _softmax_over_queries(scores, temperature):
    logits = scores / temperature
    logits = logits - logits.max(0, keepdims=True)
    weights = np.exp(logits)
    return weights / weights.sum(0, keepdims=True)


def dual_softmax(scores, temperature):
    # Each video's similarity is reweighted by how strongly that video prefers this query over the others.
    return scores * _softmax_over_queries(scores, temperature)


def inverted_softmax(scores, temperature):
    logits = scores / temperature
    top = logits.max(0, keepdims=True)
    return logits - (top + np.log(np.exp(logits - top).sum(0, keepdims=True)))


def csls(scores, k):
    # The query-side CSLS term is constant within a query's ranking, so only the video hub term matters.
    k = min(k, scores.shape[0])
    hub = -np.sort(-scores, axis=0)[:k].mean(0, keepdims=True)
    return 2 * scores - hub


def normalize(scores, method):
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 2 or not np.isfinite(scores).all():
        raise ValueError('Invalid score matrix')
    kind = method['kind']
    if kind == 'none':
        result = scores
    elif kind == 'dual-softmax':
        result = dual_softmax(scores, method['temperature'])
    elif kind == 'inverted-softmax':
        result = inverted_softmax(scores, method['temperature'])
    elif kind == 'csls':
        result = csls(scores, method['k'])
    else:
        raise ValueError('Unknown normalization: ' + kind)
    if not np.isfinite(result).all():
        raise ValueError('Normalization produced nonfinite scores')
    return result.astype(np.float32)


def ranking_metrics(scores, query_ids, video_ids, qrels):
    """R-Precision/MAP identical to baseline.post_submission.evaluate, given precomputed scores."""
    order = np.argsort(np.asarray(video_ids), kind='stable'); ids = [video_ids[i] for i in order]
    ranking = np.argsort(-np.asarray(scores)[:, order], axis=1, kind='stable')
    relevant = [set(qrels[qid]['relevant_video_ids']) for qid in query_ids]
    positive = np.asarray([[vid in r for vid in ids] for r in relevant], dtype=bool)
    hits = np.take_along_axis(positive, ranking, axis=1)
    counts = positive.sum(1); positions = np.arange(1, len(ids) + 1)
    rp = np.divide((hits & (positions <= counts[:, None])).sum(1), counts,
                   out=np.full(len(counts), np.nan), where=counts > 0)
    ap = np.divide((hits * hits.cumsum(1) / positions).sum(1), counts,
                   out=np.full(len(counts), np.nan), where=counts > 0)
    return {'r_precision': float(np.nanmean(rp)), 'map': float(np.nanmean(ap)),
            'scored_queries': int(np.isfinite(rp).sum()),
            'per_query': [{'query_id': qid, 'r_precision': None if not np.isfinite(r) else float(r)}
                          for qid, r in zip(query_ids, rp, strict=True)]}
