"""Training-only pilot sampling with whole-video and relevance-group exclusions."""
from __future__ import annotations

import re
import numpy as np

FOCUS = re.compile(r'\b(pedestrian|person|people|adult|bus|before|after|while|then|yield|turn|stop)\b', re.I)


def check_partitions(parent, qrels):
    query_sets = {name: set(ids) for name, ids in parent['query_groups'].items()}
    signatures = {}
    for name, ids in query_sets.items():
        if len(ids) != len(parent['query_groups'][name]) or not ids <= set(qrels):
            raise ValueError('Duplicate or unknown partition query')
        signatures[name] = {tuple(sorted(set(qrels[qid]['relevant_video_ids']))) for qid in ids}
    names = list(query_sets)
    for i, name in enumerate(names):
        for other in names[i + 1:]:
            if query_sets[name] & query_sets[other] or signatures[name] & signatures[other]:
                raise ValueError('Query or complete-positive-set leakage')
    all_groups = parent['development_video_groups'] + parent.get('audit_video_groups', [])
    seen = set()
    for group in all_groups:
        if len(group) != len(set(group)) or seen & set(group):
            raise ValueError('Video groups overlap')
        seen.update(group)


def select_queries(queries, qrels, fitting_query_ids, fitting_video_ids, count=256, seed=43):
    videos = set(fitting_video_ids)
    groups = {}
    for qid in sorted(fitting_query_ids):
        relevant = set(qrels[qid]['relevant_video_ids'])
        if relevant & videos and videos - relevant:
            groups.setdefault(tuple(sorted(relevant)), []).append(qid)
    rng = np.random.default_rng(seed)
    keys = sorted(groups)
    if not keys:
        raise ValueError('No fitting groups with both positives and negatives')
    picked = []
    for index in rng.permutation(len(keys)):
        choices = groups[keys[index]]
        focused = [qid for qid in choices if FOCUS.search(queries[qid]['text'])]
        picked.append(str(rng.choice(focused or choices)))
        if len(picked) == count:
            break
    return picked


def sample_batch(query_ids, fitting_video_ids, qrels, *, query_batch=2, excluded=None, rng=None):
    rng = np.random.default_rng(43) if rng is None else rng
    excluded = {} if excluded is None else excluded
    videos = sorted(fitting_video_ids)
    if len(query_ids) < query_batch or len(videos) < query_batch * 2:
        raise ValueError('Insufficient fitting data for batch')
    for _ in range(200):
        chosen = rng.choice(query_ids, query_batch, replace=False).tolist()
        signatures = [tuple(sorted(set(qrels[qid]['relevant_video_ids']))) for qid in chosen]
        if len(set(signatures)) != len(signatures):
            continue
        anchors = []
        for qid in chosen:
            candidates = sorted((set(qrels[qid]['relevant_video_ids']) & set(videos)) - set(anchors))
            if not candidates:
                break
            anchors.append(str(rng.choice(candidates)))
        if len(anchors) != query_batch:
            continue
        forbidden = set(anchors)
        for qid in chosen:
            forbidden.update(qrels[qid]['relevant_video_ids'])
            forbidden.update(excluded.get(qid, []))
        negatives = sorted(set(videos) - forbidden)
        if len(negatives) < query_batch:
            continue
        chosen_videos = anchors + rng.choice(negatives, query_batch, replace=False).tolist()
        positive, negative = [], []
        for qid in chosen:
            known = set(qrels[qid]['relevant_video_ids'])
            ambiguous = set(excluded.get(qid, []))
            positive.append([vid in known for vid in chosen_videos])
            negative.append([vid not in known and vid not in ambiguous for vid in chosen_videos])
        return {'query_ids': chosen, 'video_ids': chosen_videos,
                'positive_mask': positive, 'negative_mask': negative}
    raise ValueError('Unable to sample a valid multi-positive fitting-only batch')
