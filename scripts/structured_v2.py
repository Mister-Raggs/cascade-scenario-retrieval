"""Directions 1-7 on development folds: every arm judged against the validated combo recipe in one run.

Arms (all fused 50/50 with the submitted recipe; unparsed queries keep the submitted ranking):
  combo      pooled + motion2 + windows, linear probes, late per-block fusion, parser v1 (the validated recipe)
  temporal   combo, plus window-level atom probabilities for while/before queries (direction 1)
  expert     per-atom block set chosen by inner-fold AUC on fitting videos (direction 2)
  learned    per-entity exponents on atom log-probabilities chosen on fitting queries (direction 3)
  mlp        multi-label MLP probes per block (direction 4)
  parser2    combo with parser v2 (direction 5)
  motionnet  combo + cross-fitted temporal CNN on motion series for ego atoms (direction 7)
  attrft     combo + per-fold vision-LoRA window features (direction 6; separate run)
Jobs: one per fold ('fold-0'..'fold-2') then 'merge'.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from baseline.annotation_events import load_events
from baseline.egomotion import clip_vector
from baseline.hubness import ranking_metrics
from baseline.native_ranking import blended_scores
from baseline.post_submission import fold_indices, positives, fit_heads, project, paired_gain
from baseline.probe import auc, fit_logistic, standardize
from baseline.query_parser import translate as translate_v1, Unparsed
from baseline.query_parser_v2 import translate as translate_v2
from baseline.query_struct import tree, atoms, atom_present
from scripts.feature_retrieval import atomic_json, load_vectors
from scripts.post_submission import stage_directory, verified_result
from scripts.query_side import comparison_gate, finish
from scripts.structured_eval import provider
from scripts import lora_pilot

COMBO = ('pooled', 'motion2', 'windows')
EXPERT_SETS = (COMBO, COMBO + ('vqa',), COMBO + ('detector',), COMBO + ('vqa', 'detector'))
ENTITIES = ('ego', 'agent', 'obj', 'light')


# ----- probes -------------------------------------------------------------------------------------------
def logistic_prob(x, fit, held, y):
    y_fit = y[fit]
    if y_fit.sum() < 3 or y_fit.sum() > len(y_fit) - 3:
        return np.full(len(held), float(np.clip(y_fit.mean(), 1e-3, 1 - 1e-3)))
    z = standardize(x, fit); w, b = fit_logistic(z[fit], y_fit)
    return np.clip(1 / (1 + np.exp(-(z[held] @ w + b))), 1e-4, 1 - 1e-4)


def mlp_probs(x, fit, held, labels, seed=0, epochs=60, hidden=256):
    """One multi-label MLP per block; returns held probabilities [held, atoms]."""
    import torch
    torch.manual_seed(seed)
    z = standardize(x, fit); xt = torch.as_tensor(z, dtype=torch.float32); y = torch.as_tensor(labels, dtype=torch.float32)
    prior = y[fit].mean(0).clamp(1e-3, 1 - 1e-3); pos_weight = ((1 - prior) / prior).clamp(max=20)
    model = torch.nn.Sequential(torch.nn.Linear(x.shape[1], hidden), torch.nn.ReLU(), torch.nn.Dropout(.2), torch.nn.Linear(hidden, labels.shape[1]))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    fit_t = torch.as_tensor(fit); rng = np.random.default_rng(seed)
    for _ in range(epochs):
        model.train()
        for chunk in np.array_split(rng.permutation(len(fit)), max(1, len(fit) // 64)):
            ix = fit_t[chunk]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model(xt[ix]), y[ix], pos_weight=pos_weight)
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        return np.clip(torch.sigmoid(model(xt[torch.as_tensor(held)])).numpy(), 1e-4, 1 - 1e-4)


def motionnet_logits(series, fit, held, labels, seed=0, epochs=40):
    """Small temporal CNN on [videos, T, C] motion series; returns held logits [held, atoms]."""
    import torch
    torch.manual_seed(seed)
    x = torch.as_tensor(series, dtype=torch.float32)
    mean, std = x[fit].mean((0, 1), keepdim=True), x[fit].std((0, 1), keepdim=True) + 1e-6
    x = ((x - mean) / std).transpose(1, 2)
    y = torch.as_tensor(labels, dtype=torch.float32); prior = y[fit].mean(0).clamp(1e-3, 1 - 1e-3)
    pos_weight = ((1 - prior) / prior).clamp(max=20)
    conv = torch.nn.Sequential(torch.nn.Conv1d(x.shape[1], 32, 5, padding=2), torch.nn.ReLU(),
                               torch.nn.Conv1d(32, 32, 5, padding=4, dilation=2), torch.nn.ReLU())
    head = torch.nn.Linear(64, labels.shape[1])
    params = list(conv.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-2)
    forward = lambda b: head(torch.cat([conv(b).amax(2), conv(b).mean(2)], 1))
    rng = np.random.default_rng(seed); fit_t = torch.as_tensor(fit)
    for _ in range(epochs):
        for chunk in np.array_split(rng.permutation(len(fit)), max(1, len(fit) // 64)):
            ix = fit_t[chunk]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(forward(x[ix]), y[ix], pos_weight=pos_weight)
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return forward(x[torch.as_tensor(held)]).numpy()


def inner_folds(fit, k, seed):
    order = np.random.default_rng(seed).permutation(len(fit))
    return [np.sort(fit[part]) for part in np.array_split(order, k)]


def oof_prob(x, fit, inner, y):
    """Inner cross-fitted probabilities aligned with the positions of `fit`."""
    out = np.zeros(len(fit)); pos = {v: i for i, v in enumerate(fit)}
    for part in inner:
        out[[pos[v] for v in part]] = logistic_prob(x, np.setdiff1d(fit, part), part, y)
    return out


def late(probs):
    logit = np.mean([np.log(p / (1 - p)) for p in probs], axis=0)
    return np.clip(1 / (1 + np.exp(-logit)), 1e-4, 1 - 1e-4)


# ----- temporal scoring ---------------------------------------------------------------------------------
def tree_windows(node, wp):
    """Window-resolved score: atoms/and -> vector over windows; while/before -> scalar broadcast."""
    if node[0] == 'atom':
        return wp[node[1]]
    if node[0] == 'and':
        return np.prod([tree_windows(c, wp) for c in node[1]], axis=0)
    a, b = tree_windows(node[1], wp), tree_windows(node[2], wp)
    a = a if np.ndim(a) else np.full(len(b) if np.ndim(b) else 1, a)
    b = b if np.ndim(b) else np.full(len(a), b)
    if node[0] == 'while':
        return np.full_like(a, (a * b).max())
    return np.full_like(a, max(a[i] * b[j] for i in range(len(a)) for j in range(i, len(b))))


def temporal_scalar(node, wp):
    value = tree_windows(node, wp)
    return float(np.max(value))


# ----- fold job -----------------------------------------------------------------------------------------
def load_common(plan):
    parent, queries, qrels = lora_pilot.inputs(plan)
    fm = json.loads((ROOT / plan['feature_directory'] / 'feature_manifest.json').read_text())
    vids = fm['video_ids']
    blocks = {name: provider(name, spec, vids) for name, spec in plan['blocks'].items()}
    events = load_events(ROOT / plan['annotation_directory'])
    texts = {qid: r['text'] for qid, r in queries.items()}
    parsed = {}
    for version, fn in (('v1', translate_v1), ('v2', translate_v2)):
        for qid in parent['query_groups']['dev']:
            try:
                t = tree(fn(texts[qid])); parsed.setdefault(version, {})[qid] = (t, sorted(set(atoms(t))))
            except Unparsed:
                parsed.setdefault(version, {})[qid] = None
    return parent, queries, qrels, fm, vids, blocks, events, parsed


def window_data(plan, vids):
    """Per-window features (window vector, motion-segment stats) and window bounds for development videos."""
    spec = plan['blocks']['windows']; wm = json.loads((ROOT / spec['manifest']).read_text()); k = wm['windows']
    flat = load_vectors(ROOT / spec['vectors']); wv = flat.reshape(-1, k, flat.shape[1])
    widx = {key.split('@')[0]: i for i, key in enumerate(wm['keys'][::k])}
    bounds = json.loads((ROOT / plan['window_bounds']).read_text())
    series = np.load(ROOT / plan['blocks']['motion2']['path']); fps = plan['motion_fps']
    feats_v, feats_m, have = [], [], []
    for v in vids:
        if v not in widx:
            have.append(False); feats_v.append(np.zeros((k, flat.shape[1]), np.float32)); feats_m.append(np.zeros((k, 6 * 9), np.float32)); continue
        have.append(True); feats_v.append(wv[widx[v]]); s = series[v]; rows = []
        for b in bounds[v]:
            lo, hi = int(b['bounds'][0] * fps), max(int(b['bounds'][1] * fps), int(b['bounds'][0] * fps) + 2)
            rows.append(clip_vector(s[lo:hi], 2))
        feats_m.append(np.asarray(rows, np.float32))
    return np.asarray(feats_v), np.asarray(feats_m), np.asarray(have), bounds


def fold_job(out, plan, fold):
    parent, queries, qrels, fm, vids, blocks, events, parsed = load_common(plan)
    qvec = load_vectors(ROOT / plan['feature_directory'] / 'queries-train.npz')
    vvec = load_vectors(ROOT / plan['feature_directory'] / 'aggregate.npz')
    labels_of = {v: {e[0] for e in events[v]['events']} for v in vids}
    ix = fold_indices(parent, qrels, fm['query_ids'], vids, fold)
    fit, held = np.asarray(ix['fit_v']), np.asarray(ix['held_v'])
    for name, (_, mask) in blocks.items():
        if not mask[fit].all() or not mask[held].all():
            raise ValueError('Fold uses videos without features: ' + name)
    needed = sorted({a for version in parsed.values() for p in version.values() if p for a in p[1]})
    if plan.get('atom_limit'):  # smoke runs only
        needed = needed[::max(1, len(needed) // plan['atom_limit'])][:plan['atom_limit']]
        for version in parsed.values():
            for q, p in version.items():
                if p and not set(p[1]) <= set(needed):
                    version[q] = None
    truth = {a: np.asarray([atom_present(a, labels_of[v]) for v in vids]) for a in needed}
    started = time.monotonic(); seed = plan['seed']
    # Block probabilities on held videos and inner out-of-fold probabilities on fitting videos.
    held_p, oof_p = {}, {}
    inner = inner_folds(fit, plan['inner_folds'], seed + fold)
    for name, (x, _) in blocks.items():
        held_p[name] = {a: logistic_prob(x, fit, held, truth[a]) for a in needed}
        oof_p[name] = {a: oof_prob(x, fit, inner, truth[a]) for a in needed}
        atomic_json(out / 'progress.json', {'fold': fold, 'block': name, 'elapsed_seconds': time.monotonic() - started})
    combo = {a: late([held_p[b][a] for b in COMBO]) for a in needed}
    # Direction 2: per-atom expert selection on fitting-video out-of-fold AUC (ties keep the combo set).
    choice = {}
    for a in needed:
        scores = [auc(late([oof_p[b][a] for b in s]), truth[a][fit]) for s in EXPERT_SETS]
        scores = [(-1 if np.isnan(s) else s) for s in scores]
        choice[a] = int(np.argmax([s + (1e-9 if i == 0 else 0) for i, s in enumerate(scores)]))
    expert = {a: late([held_p[b][a] for b in EXPERT_SETS[choice[a]]]) for a in needed}
    # Direction 4: multi-label MLP per combo block.
    matrix = np.stack([truth[a] for a in needed], 1)
    mlp_block = [mlp_probs(blocks[b][0], fit, held, matrix, seed, plan['mlp_epochs']) for b in COMBO]
    mlp = {a: late([m[:, j] for m in mlp_block]) for j, a in enumerate(needed)}
    # Direction 7: cross-fitted temporal CNN on motion series for ego atoms.
    store = np.load(ROOT / plan['blocks']['motion2']['path']); length = min(store[v].shape[0] for v in store.files)
    series = np.zeros((len(vids), length, 6), np.float32)
    for i, v in enumerate(vids):
        if v in store.files:
            series[i] = store[v][:length]
    ego = [a for a in needed if a.startswith('ego')]
    net_p = {}
    if ego:
        ego_y = np.stack([truth[a] for a in ego], 1)
        net_feat = np.zeros((len(vids), len(ego)), np.float32)
        net_feat[held] = motionnet_logits(series, fit, held, ego_y, seed, plan['motionnet_epochs'])
        for part in inner:
            net_feat[part] = motionnet_logits(series, np.setdiff1d(fit, part), part, ego_y, seed, plan['motionnet_epochs'])
        net_p = {a: logistic_prob(net_feat, fit, held, truth[a]) for a in ego}
    motionnet = {a: (late([combo[a], net_p[a]]) if a in net_p else combo[a]) for a in needed}
    # Direction 1: window-level probes for atoms used in relational queries.
    wv, wm, have, bounds = window_data(plan, vids); k = wv.shape[1]
    rel_atoms = sorted({a for p in parsed['v1'].values() if p and p[0][0] in ('while', 'before') for a in p[1]})
    from baseline.atom_labels import window_atom_matrix
    dev = np.flatnonzero(have)
    wtruth = window_atom_matrix(events, [vids[i] for i in dev], {vids[i]: [tuple(b['bounds']) for b in bounds[vids[i]]] for i in dev}, rel_atoms)
    full = np.zeros((len(vids), k, len(rel_atoms)), bool); full[dev] = wtruth
    rows_fit = (fit[:, None] * k + np.arange(k)).reshape(-1); rows_held = (held[:, None] * k + np.arange(k)).reshape(-1)
    fv, fmot = wv.reshape(len(vids) * k, -1), wm.reshape(len(vids) * k, -1)
    window_p = {}
    for j, a in enumerate(rel_atoms):
        y = full[:, :, j].reshape(-1)
        window_p[a] = late([logistic_prob(fv, rows_fit, rows_held, y), logistic_prob(fmot, rows_fit, rows_held, y)]).reshape(len(held), k)
    # Direction 3: per-entity exponents chosen on fitting queries with out-of-fold fitting probabilities.
    oof_combo = {a: late([oof_p[b][a] for b in COMBO]) for a in needed}
    texts = {qid: r['text'] for qid, r in queries.items()}
    sample = list(np.random.default_rng(seed + 100 + fold).permutation(ix['fit_query_ids'])[:plan['learned_query_sample']])
    fit_parsed = {}
    for q in sample:
        try:
            fit_parsed[q] = sorted(set(atoms(tree(translate_v1(texts[q])))))
        except Unparsed:
            continue
    for a in sorted({a for ks in fit_parsed.values() for a in ks} - set(oof_combo)):
        truth[a] = np.asarray([atom_present(a, labels_of[v]) for v in vids])
        oof_combo[a] = late([oof_prob(blocks[b][0], fit, inner, truth[a]) for b in COMBO])
    fit_ids = list(fit_parsed)
    best = (None, -1.0)
    for alpha in itertools.product(plan['learned_grid'], repeat=len(ENTITIES)):
        weights = dict(zip(ENTITIES, alpha))
        scores = np.stack([np.sum([weights[a.split('|')[0]] * np.log(oof_combo[a]) for a in fit_parsed[q]], axis=0) for q in fit_ids])
        rp = ranking_metrics(scores, fit_ids, list(np.asarray(vids)[fit]), qrels)['r_precision']
        if rp > best[1] + 1e-12:
            best = (weights, rp)
    learned_weights = best[0]
    # Scores for every arm, then fusion with the submitted recipe per head seed.
    dev_ids = ix['dev_query_ids']; dq = qvec[ix['dev_q']]; vh = vvec[held]; raw = dq @ vh.T
    def clip_scores(prob, version='v1', weights=None):
        s = np.zeros_like(raw); m = np.zeros(len(dev_ids), bool)
        for i, q in enumerate(dev_ids):
            p = parsed[version].get(q)
            if p:
                s[i] = np.sum([(weights[a.split('|')[0]] if weights else 1.0) * np.log(prob[a]) for a in p[1]], axis=0); m[i] = True
        return s, m
    structured = {name: clip_scores(prob) for name, prob in [('combo', combo), ('expert', expert), ('mlp', mlp), ('motionnet', motionnet)]}
    structured['learned'] = clip_scores(combo, weights=learned_weights)
    if plan.get('attrft_by_fold'):
        # Direction 6: this fold's vision-LoRA window features (adapter trained without this fold's videos).
        x, present = provider('attrft', plan['attrft_by_fold'][str(fold)], vids)
        if not present[fit].all() or not present[held].all():
            raise ValueError('Fold uses videos without attrft features')
        structured['attrft'] = clip_scores({a: late([combo[a], logistic_prob(x, fit, held, truth[a])]) for a in needed})
    structured['parser2'] = clip_scores(combo, version='v2')
    temporal, mask = structured['combo'][0].copy(), structured['combo'][1]
    for i, q in enumerate(dev_ids):
        p = parsed['v1'].get(q)
        if p and p[0][0] in ('while', 'before'):
            for j in range(len(held)):
                wp = {a: window_p[a][j] if a in window_p else np.full(k, combo[a][j]) for a in p[1]}
                temporal[i, j] = .5 * temporal[i, j] + .5 * np.log(temporal_scalar(p[0], wp) + 1e-6)
    structured['temporal'] = (temporal, mask)
    cells = []
    labels = positives(ix['fit_query_ids'], ix['fit_video_ids'], qrels)
    for head_seed in plan['head_seeds']:
        state = fit_heads(qvec[ix['fit_q']], vvec[fit], labels, plan['head_training'], head_seed, [plan['head_epochs']], 'cpu')[plan['head_epochs']]
        submitted = blended_scores(raw, project(dq, state) @ project(vh, state).T, plan['submitted_raw_weight'])
        arms = {'submitted': submitted}
        for name, (s, m) in structured.items():
            fused = blended_scores(np.where(m[:, None], s, raw), submitted, plan['fusion_weight'])
            arms[name] = np.where(m[:, None], fused, submitted)
        for name, scores in arms.items():
            cells.append({'fold': fold, 'seed': head_seed, 'arm': name, **ranking_metrics(scores, dev_ids, ix['held_video_ids'], qrels)})
    atomic_json(out / 'results/cells.json', cells)
    finish(out, plan, {'kind': f'directions 1-7 development fold {fold}', 'gate_passed': True, 'accuracy_score': True, 'fold': fold,
           'expert_choice_counts': {str(i): sum(c == i for c in choice.values()) for i in range(len(EXPERT_SETS))},
           'learned_weights': learned_weights, 'parse_coverage': {v: float(np.mean([bool(parsed[v][q]) for q in dev_ids])) for v in parsed},
           'relational_atoms': len(rel_atoms), 'seconds': time.monotonic() - started})


def merge(out, plan):
    cells = []
    for fold in plan['folds']:
        cells += json.loads(verified_result(out.parent, f'{plan["job_prefix"]}fold-{fold}', 'cells.json').read_text())
    base = {(c['fold'], c['seed']): c for c in cells if c['arm'] == plan['reference_arm']}
    sub = {(c['fold'], c['seed']): c for c in cells if c['arm'] == 'submitted'}
    summary = {}
    for arm in sorted({c['arm'] for c in cells}):
        rows = [{'fold': c['fold'], 'seed': c['seed'], 'r_precision': c['r_precision'], 'map': c['map'],
                 **paired_gain(c, base[(c['fold'], c['seed'])], plan['minimum_rp_gain'])} for c in cells if c['arm'] == arm]
        vs_sub = [paired_gain(c, sub[(c['fold'], c['seed'])], plan['minimum_rp_gain'])['r_precision_delta'] for c in cells if c['arm'] == arm]
        summary[arm] = {'mean_rp': float(np.mean([r['r_precision'] for r in rows])), 'mean_map': float(np.mean([r['map'] for r in rows])),
                        'mean_rp_delta_vs_submitted': float(np.mean(vs_sub)), **comparison_gate(rows, plan['minimum_rp_gain']), 'comparisons': rows}
    passing = [a for a, s in summary.items() if a not in (plan['reference_arm'], 'submitted') and s['gate_passed']]
    atomic_json(out / 'results/cells.json', [{k: v for k, v in c.items() if k != 'per_query'} for c in cells])
    finish(out, plan, {'kind': 'directions 1-7 versus the validated combo on development folds', 'accuracy_score': True,
           'reference_arm': plan['reference_arm'], 'gate_passed': bool(passing), 'passing_arms': passing, 'summary': summary,
           'limits': 'Reused development folds; every passing arm still needs held-out confirmation (reserved audit videos).'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', required=True); parser.add_argument('--plan', type=Path, required=True); parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); os.chdir(ROOT)
    import torch
    plan = json.loads(args.plan.read_text()); plan['plan_path'] = str(args.plan)
    if not plan['development_only'] or plan['official_submission_allowed']:
        raise ValueError('Only private local experiments are supported')
    torch.set_num_threads(plan['cpu_threads']); stage_directory(args.out)
    JOBS[args.job](args.out, plan)


JOBS = {'fold-0': lambda o, p: fold_job(o, p, 0), 'fold-1': lambda o, p: fold_job(o, p, 1),
        'fold-2': lambda o, p: fold_job(o, p, 2), 'merge': merge}

if __name__ == '__main__':
    main()
