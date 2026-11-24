"""Matched, fitting-video-only head experiments and paired development gates."""
from __future__ import annotations

import numpy as np
import torch
from .lora_data import check_partitions
from .native_ranking import blended_scores
from .projection import SharedProjection, positive_aware_loss


def fold_indices(parent, qrels, query_ids, video_ids, fold):
    check_partitions(parent, qrels)
    if len(query_ids) != len(set(query_ids)) or len(video_ids) != len(set(video_ids)):
        raise ValueError('Duplicate feature IDs')
    qp = {qid: i for i, qid in enumerate(query_ids)}
    vp = {vid: i for i, vid in enumerate(video_ids)}
    audit = {v for group in parent.get('audit_video_groups', []) for v in group}
    if fold == 'all':
        # Final package after the audit confirmations (2026-10-22): fit on every annotated video; nothing is held out.
        held = []
        fitting = [v for key in ('development_video_groups', 'audit_video_groups') for group in parent[key] for v in group]
    elif fold == 'audit':
        # Final confirmation: fit on every development video, judge the reserved audit queries on the reserved audit videos.
        held = [v for group in parent['audit_video_groups'] for v in group]
        fitting = [v for group in parent['development_video_groups'] for v in group]
        if set(fitting) & set(held):
            raise ValueError('Development/audit video leakage')
    else:
        held = parent['development_video_groups'][fold]
        fitting = [v for i, group in enumerate(parent['development_video_groups']) if i != fold for v in group]
        if set(fitting) & (set(held) | audit) or set(held) & audit:
            raise ValueError('Held/audit video leakage')
    if not set(fitting + held) <= set(vp):
        raise ValueError('Feature videos missing from partition')
    fitting_set = set(fitting)
    fit_queries = [qid for qid in parent['query_groups']['fit']
                   if set(qrels[qid]['relevant_video_ids']) & fitting_set]
    dev_queries = [] if fold == 'all' else list(parent['query_groups']['audit' if fold == 'audit' else 'dev'])
    if not fit_queries or (not dev_queries and fold != 'all') or not set(fit_queries + dev_queries) <= set(qp):
        raise ValueError('Missing fitting/development features')
    return {'fit_query_ids': fit_queries, 'dev_query_ids': dev_queries,
            'fit_video_ids': fitting, 'held_video_ids': held,
            'fit_q': [qp[qid] for qid in fit_queries], 'dev_q': [qp[qid] for qid in dev_queries],
            'fit_v': [vp[vid] for vid in fitting], 'held_v': [vp[vid] for vid in held]}


def positives(query_ids, video_ids, qrels):
    relevant_sets = [set(qrels[qid]['relevant_video_ids']) for qid in query_ids]
    return np.asarray([[vid in relevant for vid in video_ids] for relevant in relevant_sets], dtype=bool)


def fit_heads(q, v, positive, settings, seed, epochs, device='cpu', progress=None):
    """Save matched checkpoints along a single deterministic optimization path."""
    if not epochs or any(e < 1 or isinstance(e, bool) for e in epochs):
        raise ValueError('Positive epoch checkpoints required')
    q = np.asarray(q, dtype=np.float32); v = np.asarray(v, dtype=np.float32)
    positive = np.asarray(positive, dtype=bool)
    if q.ndim != 2 or v.ndim != 2 or q.shape[1] != v.shape[1] or not len(q) or not len(v):
        raise ValueError('Invalid fitting matrices')
    if positive.shape != (len(q), len(v)) or not positive.any(1).all():
        raise ValueError('Every fitting query requires a fitting-video positive')
    if not np.isfinite(q).all() or not np.isfinite(v).all():
        raise ValueError('Nonfinite fitting features')
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    model = SharedProjection(q.shape[1], settings['rank']).to(device)
    qt = torch.as_tensor(q, device=device); vt = torch.as_tensor(v, device=device)
    pt = torch.as_tensor(positive, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'], weight_decay=settings['weight_decay'])
    states = {}
    for epoch in range(1, max(epochs) + 1):
        order = rng.permutation(len(q))
        for start in range(0, len(order), settings['batch_size']):
            ix = order[start:start + settings['batch_size']]
            raw = qt[ix]; aq, av = model(raw), model(vt)
            loss = positive_aware_loss(aq @ av.T / settings['temperature'], pt[ix])
            anchor = ((aq - raw).square().sum(1).mean() + (av - vt).square().sum(1).mean()) / 2
            loss = loss + settings['anchor_weight'] * anchor
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite head loss')
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1); optimizer.step()
        if epoch in epochs:
            states[epoch] = {k: x.detach().cpu().clone() for k, x in model.state_dict().items()}
        if progress:
            progress(epoch)
    return states


def project(features, state, device='cpu', batch_size=256):
    model = SharedProjection(features.shape[1], state['down.weight'].shape[0]).to(device)
    model.load_state_dict(state); model.eval()
    with torch.no_grad():
        return np.concatenate([model(torch.as_tensor(features[i:i + batch_size], device=device)).cpu().numpy()
                               for i in range(0, len(features), batch_size)])


def evaluate(q, v, query_ids, video_ids, qrels, state, weight, device='cpu'):
    # A deterministic ID order makes similarity and RRF ties comparable across arms.
    order = np.argsort(np.asarray(video_ids), kind='stable'); ids = [video_ids[i] for i in order]
    v = np.asarray(v)[order]
    raw = q @ v.T; head = project(q, state, device) @ project(v, state, device).T
    ranking = np.argsort(-blended_scores(raw, head, weight), axis=1, kind='stable')
    positive = positives(query_ids, ids, qrels)
    hits = np.take_along_axis(positive, ranking, axis=1)
    counts = positive.sum(1); positions = np.arange(1, len(ids) + 1)
    rp = np.divide((hits & (positions <= counts[:, None])).sum(1), counts,
                   out=np.full(len(counts), np.nan), where=counts > 0)
    ap = np.divide((hits * hits.cumsum(1) / positions).sum(1), counts,
                   out=np.full(len(counts), np.nan), where=counts > 0)
    if not np.isfinite(rp).any():
        raise ValueError('No scored development queries in candidate pool')
    signatures = [tuple(sorted(set(qrels[qid]['relevant_video_ids']))) for qid in query_ids]
    group_means = [float(np.nanmean(rp[[i for i, s in enumerate(signatures) if s == sig]]))
                   for sig in sorted(set(signatures)) if any(np.isfinite(rp[i]) for i, s in enumerate(signatures) if s == sig)]
    return {'r_precision': float(np.nanmean(rp)), 'map': float(np.nanmean(ap)),
            'group_macro_r_precision': float(np.mean(group_means)),
            'scored_queries': int(np.isfinite(rp).sum()),
            'per_query': [{'query_id': qid, 'r_precision': None if not np.isfinite(r) else float(r),
                           'average_precision': None if not np.isfinite(a) else float(a)}
                          for qid, r, a in zip(query_ids, rp, ap, strict=True)]}


def select_head_recipe(cells, folds, seeds, epochs, weights, minimum_gain=.001):
    expected = {(fold, seed) for fold in folds for seed in seeds}
    recipes = []
    for epoch in epochs:
        for weight in weights:
            rows = [c for c in cells if c['epoch'] == epoch and c['raw_weight'] == weight]
            if len(rows) != len(expected) or {(c['fold'], c['seed']) for c in rows} != expected:
                raise ValueError('Incomplete/duplicate development cells')
            baseline = [next(c for c in cells if c['fold'] == row['fold'] and c['seed'] == row['seed']
                             and c['epoch'] == 4 and c['raw_weight'] == .25) for row in rows]
            delta = np.asarray([r['r_precision'] - b['r_precision'] for r, b in zip(rows, baseline, strict=True)])
            map_delta = np.asarray([r['map'] - b['map'] for r, b in zip(rows, baseline, strict=True)])
            fold_means = [float(np.mean([d for d, r in zip(delta, rows, strict=True) if r['fold'] == fold])) for fold in folds]
            gate = (float(delta.mean()) > minimum_gain and (delta > 0).sum() >= np.ceil(len(rows) * 2 / 3)
                    and min(fold_means) >= 0 and float(map_delta.mean()) >= 0)
            recipes.append({'epoch': epoch, 'raw_weight': weight,
                            'r_precision': float(np.mean([r['r_precision'] for r in rows])),
                            'map': float(np.mean([r['map'] for r in rows])),
                            'mean_rp_delta': float(delta.mean()), 'mean_map_delta': float(map_delta.mean()),
                            'positive_cells': int((delta > 0).sum()), 'fold_mean_rp_deltas': fold_means,
                            'gate_passed': bool(gate)})
    eligible = [r for r in recipes if r['gate_passed']]
    selected = max(eligible, key=lambda r: (r['r_precision'], r['map'], -r['epoch'], -r['raw_weight'])) if eligible else None
    return recipes, selected


def paired_gain(adapted, frozen, minimum_gain=.001):
    ap = {x['query_id']: x['r_precision'] for x in adapted['per_query']}
    fp = {x['query_id']: x['r_precision'] for x in frozen['per_query']}
    if set(ap) != set(fp) or any((ap[qid] is None) != (fp[qid] is None) for qid in ap):
        raise ValueError('Paired query coverage differs')
    delta = adapted['r_precision'] - frozen['r_precision']
    map_delta = adapted['map'] - frozen['map']
    return {'r_precision_delta': delta, 'map_delta': map_delta,
            'improved_queries': sum(ap[qid] > fp[qid] for qid in ap if ap[qid] is not None),
            'regressed_queries': sum(ap[qid] < fp[qid] for qid in ap if ap[qid] is not None),
            'gate_passed': delta > minimum_gain and map_delta >= 0}
