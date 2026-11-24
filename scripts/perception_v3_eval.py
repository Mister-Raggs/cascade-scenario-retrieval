"""Bigger-VLM and dense-flow experiments judged against the confirmed combo and t4 (2026-10-19; development folds only).

Jobs:
  flow  CPU, folds 0-2: ego atoms from the motion2 CNN (t4), RAFT flow CNNs (full grid or affine-only) and their late
        fusion; parser v2 as in t4. Pre-registered gate versus t4: mean RP gain > 0.001, at least 4/6 fold-seed cells
        improve, no fold regresses on average, mean MAP change >= 0.
  vlm   CPU, fold 0: adds fine-tuned VLM probabilities for agent/light/object atoms (8B from tracks-v1, 30B-A3B from
        this batch). Expansion to folds 1-2 needs 30B query-weighted AUC >= 0.88 on those families and RP gain
        >= +0.01 over combo for vlm30-late at fusion weight 0.5. Weight 0.8 arms are exploratory.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from baseline.hubness import ranking_metrics
from baseline.probe import auc
from scripts.feature_retrieval import atomic_json, load_vectors
from scripts.post_submission import stage_directory, verified_result
from scripts.query_side import finish
from scripts.structured_v2 import inner_folds, late, logistic_prob, motionnet_logits
from scripts.tracks_eval import combo_probs, common, fused, parse, structured, submitted_scores, translate_v1, translate_v2

AFFINE = slice(96, 110)  # flow channels after the 6x8 grid: two affine fits plus magnitude statistics


def fold_state(plan, fold, flow_store):
    p = {**plan, 'held_fold': fold}
    parent, queries, qrels, fm, vids, ix, texts, sets = common(p)
    p1, p2 = parse(texts, translate_v1), parse(texts, translate_v2)
    needed = sorted({a for d in (p1, p2) for v in d.values() if v for a in v})
    prob, truth = combo_probs(p, vids, ix, sets, needed)
    fit, held = np.asarray(ix['fit_v']), np.asarray(ix['held_v'])
    ego = [a for a in needed if a.startswith('ego')]; y = np.stack([truth[a] for a in ego], 1)
    inner = inner_folds(fit, plan['inner_folds'], plan['seed'] + (3 if fold == 'audit' else fold))

    def net(series):
        logits = np.zeros((len(vids), len(ego)), np.float32)
        logits[held] = motionnet_logits(series, fit, held, y, plan['seed'], plan['motionnet_epochs'])
        for part in inner:
            logits[part] = motionnet_logits(series, np.setdiff1d(fit, part), part, y, plan['seed'], plan['motionnet_epochs'])
        return {a: logistic_prob(logits, fit, held, truth[a]) for a in ego}

    def series(store, channels=slice(None)):
        length = min(store[v].shape[0] for v in store.files); width = store[store.files[0]][:, channels].shape[1]
        out = np.zeros((len(vids), length, width), np.float32); present = set(store.files)
        if any(vids[i] not in present for i in np.concatenate([fit, held])):
            raise ValueError('Motion series missing for a fold video')
        for i, v in enumerate(vids):
            if v in present:  # videos outside the development folds are never used
                out[i] = store[v][:length, channels]
        return out
    m2 = np.load(ROOT / plan['blocks']['motion2']['path'])
    nets = {'m2': net(series(m2))}
    if flow_store is not None:  # flow is optional (the audit recipe does not use it)
        nets.update(flow=net(series(flow_store)), flowaff=net(series(flow_store, AFFINE)))
    qvec = load_vectors(ROOT / plan['feature_directory'] / 'queries-train.npz'); vvec = load_vectors(ROOT / plan['feature_directory'] / 'aggregate.npz')
    subs = {seed: submitted_scores(p, qvec, vvec, ix, qrels, seed) for seed in plan['head_seeds']}
    weights = {a: sum(1 for v in p2.values() if v and a in v) for a in needed}
    return dict(plan=p, qrels=qrels, vids=vids, ix=ix, p1=p1, p2=p2, prob=prob, truth=truth, ego=ego, nets=nets, subs=subs, weights=weights)


def with_ego(st, *sources):
    return {a: (late([st['prob'][a]] + [st['nets'][s][a] for s in sources]) if a in st['ego'] else st['prob'][a]) for a in st['prob']}


def with_vlm(base, vlm, replace=False):
    return {a: ((vlm[a] if replace else late([base[a], vlm[a]])) if a in vlm else base[a]) for a in base}


def evaluate(st, arms):
    ix = st['ix']; rows = []
    for seed, (raw, sub) in st['subs'].items():
        for name, (parsed, prob, w) in arms.items():
            m = ranking_metrics(fused(raw, sub, *structured(raw, ix['dev_query_ids'], parsed, prob), w), ix['dev_query_ids'], ix['held_video_ids'], st['qrels'])
            rows.append({'fold': st['plan']['held_fold'], 'seed': seed, 'arm': name, 'r_precision': m['r_precision'], 'map': m['map']})
    return rows


def family_auc(st, prob, atoms):
    held = np.asarray(st['ix']['held_v']); out = {}
    for fam in sorted({a.split('|')[0] for a in atoms}):
        vals = [(auc(prob[a], st['truth'][a][held]), st['weights'][a]) for a in atoms if a.split('|')[0] == fam and a in prob and st['weights'].get(a)]
        vals = [(v, w) for v, w in vals if np.isfinite(v)]
        if vals:
            out[fam] = {'query_weighted_auc': float(np.average([v for v, _ in vals], weights=[w for _, w in vals])), 'atoms': len(vals)}
    return out


def compare(rows, arm, base):
    pairs = [(r, next(b for b in rows if b['arm'] == base and b['fold'] == r['fold'] and b['seed'] == r['seed'])) for r in rows if r['arm'] == arm]
    d = [r['r_precision'] - b['r_precision'] for r, b in pairs]; dm = [r['map'] - b['map'] for r, b in pairs]
    folds = sorted({r['fold'] for r, _ in pairs})
    per_fold = {f: float(np.mean([r['r_precision'] - b['r_precision'] for r, b in pairs if r['fold'] == f])) for f in folds}
    return {'mean_rp_delta': float(np.mean(d)), 'mean_map_delta': float(np.mean(dm)), 'cells_improved': int(sum(x > 0 for x in d)),
            'cells': len(d), 'per_fold_rp_delta': per_fold}


def summarise(rows):
    out = {}
    for arm in sorted({r['arm'] for r in rows}):
        x = [r for r in rows if r['arm'] == arm]
        out[arm] = {'mean_rp': float(np.mean([r['r_precision'] for r in x])), 'mean_map': float(np.mean([r['map'] for r in x]))}
    return out


def flow_job(out, plan):
    started = time.monotonic(); store = np.load(verified_result(out.parent, plan['job_names']['flow'], 'motion-flow.npz'))
    rows, aucs = [], {}
    for fold in (0, 1, 2):
        st = fold_state(plan, fold, store); p1, p2, w = st['p1'], st['p2'], plan['fusion_weight']
        arms = {'combo': (p1, st['prob'], w), 't4': (p2, with_ego(st, 'm2'), w), 'flow': (p2, with_ego(st, 'flow'), w),
                'flowaff': (p2, with_ego(st, 'flowaff'), w), 't4flow': (p2, with_ego(st, 'm2', 'flow'), w),
                't4flowaff': (p2, with_ego(st, 'm2', 'flowaff'), w)}
        rows += evaluate(st, arms)
        aucs[fold] = {name: family_auc(st, arms[name][1], st['ego']).get('ego') for name in ('combo', 't4', 'flow', 'flowaff', 't4flow', 't4flowaff')}
        print('fold', fold, 'done', flush=True)
    gates = {}
    for arm in ('flow', 'flowaff', 't4flow', 't4flowaff'):
        c = compare(rows, arm, 't4')
        c['passes_vs_t4'] = bool(c['mean_rp_delta'] > .001 and c['cells_improved'] >= 4 and min(c['per_fold_rp_delta'].values()) >= 0 and c['mean_map_delta'] >= 0)
        c['vs_combo'] = compare(rows, arm, 'combo'); gates[arm] = c
    gates['t4'] = {'vs_combo': compare(rows, 't4', 'combo')}
    atomic_json(out / 'results/cells.json', rows)
    finish(out, plan, {'kind': 'dense-flow ego atoms versus t4 on development folds 0-2', 'accuracy_score': True,
           'gate_passed': any(g.get('passes_vs_t4') for g in gates.values()), 'gates': gates, 'summary': summarise(rows),
           'ego_query_weighted_auc': aucs, 'seconds': time.monotonic() - started})


def vlm_job(out, plan):
    started = time.monotonic(); store = np.load(verified_result(out.parent, plan['job_names']['flow'], 'motion-flow.npz'))
    st = fold_state(plan, 0, store); p1, p2, w = st['p1'], st['p2'], plan['fusion_weight']; held_ids = st['ix']['held_video_ids']

    def load(path):
        v = np.load(path)
        if list(v['video_ids']) != list(held_ids):
            raise ValueError('VLM probabilities are not aligned with held videos: ' + str(path))
        return {str(a): np.clip(v['probs'][:, j], 1e-4, 1 - 1e-4) for j, a in enumerate(v['atoms']) if a.split('|')[0] in plan['vlm_families']}
    v8 = load(verified_result(ROOT / plan['vlm8b_run'], plan['vlm8b_job'], 'vlm-atom-probs.npz'))
    v30 = load(verified_result(out.parent, plan['job_names']['vlm_score'], 'vlm-atom-probs.npz'))
    t4, t4f = with_ego(st, 'm2'), with_ego(st, 'm2', plan['flow_arm_source'])
    arms = {'combo': (p1, st['prob'], w), 't4': (p2, t4, w), 't4flow': (p2, t4f, w),
            'vlm8b-late': (p1, with_vlm(st['prob'], v8), w), 'vlm30-late': (p1, with_vlm(st['prob'], v30), w),
            'vlm30-replace': (p1, with_vlm(st['prob'], v30, True), w),
            'vlm30-t4': (p2, with_vlm(t4, v30), w), 'vlm30-t4flow': (p2, with_vlm(t4f, v30), w),
            'vlm30-t4flow@w0.8': (p2, with_vlm(t4f, v30), .8), 'combo@w0.8': (p1, st['prob'], .8)}
    rows = evaluate(st, arms); atoms = sorted(v30)
    fams = {'probes': family_auc(st, st['prob'], atoms), 'vlm8b': family_auc(st, v8, atoms), 'vlm30': family_auc(st, v30, atoms)}
    w30 = [(fams['vlm30'][f]['query_weighted_auc'], sum(st['weights'].get(a, 0) for a in atoms if a.split('|')[0] == f)) for f in fams['vlm30']]
    pooled = float(np.average([v for v, _ in w30], weights=[x for _, x in w30]))
    c = compare(rows, 'vlm30-late', 'combo')
    expand = bool(pooled >= plan['vlm_expand_auc'] and c['mean_rp_delta'] >= plan['vlm_expand_rp'])
    deltas = {arm: compare(rows, arm, 'combo') for arm in arms if arm != 'combo'}
    atomic_json(out / 'results/cells.json', rows)
    finish(out, plan, {'kind': 'bigger-VLM atoms on development fold 0', 'accuracy_score': True, 'gate_passed': expand,
           'expand_vlm30_to_folds_1_2': expand, 'vlm30_pooled_query_weighted_auc': pooled, 'family_auc': fams,
           'deltas_vs_combo': deltas, 'summary': summarise(rows), 'seconds': time.monotonic() - started})


def vlm3_job(out, plan):
    """30B expansion over folds 0-2 (pre-registered 2026-10-20, after the fold-0 pilot passed its gate):
    primary  vlm30-t4 versus t4: standing gate (mean RP gain > 0.001, >= 4/6 cells, no fold regresses on average,
             MAP change >= 0); vlm30-t4 versus combo >= +0.02 marks it as worth an entry after audit confirmation.
    weight   fusion weight 0.8 was suggested by fold 0, so it is judged on folds 1-2 only: adopted if vlm30-t4@w0.8 beats
             vlm30-t4 on both fold means.
    flow     kept only if vlm30-t4flow passes the standing gate versus vlm30-t4 (RAFT adds a licence review).
    """
    started = time.monotonic(); store = np.load(verified_result(ROOT / plan['flow_run'][0], plan['flow_run'][1], 'motion-flow.npz'))
    rows = []
    for fold in (0, 1, 2):
        st = fold_state(plan, fold, store); p1, p2 = st['p1'], st['p2']; held_ids = st['ix']['held_video_ids']
        root, job = plan['vlm30_jobs'][str(fold)]; v = np.load(verified_result(ROOT / root, job, 'vlm-atom-probs.npz'))
        if list(v['video_ids']) != list(held_ids):
            raise ValueError('30B probabilities are not aligned with held videos of fold %d' % fold)
        v30 = {str(a): np.clip(v['probs'][:, j], 1e-4, 1 - 1e-4) for j, a in enumerate(v['atoms']) if a.split('|')[0] in plan['vlm_families']}
        t4, t4f = with_ego(st, 'm2'), with_ego(st, 'm2', plan['flow_arm_source'])
        arms = {'combo': (p1, st['prob'], .5), 'combo@w0.8': (p1, st['prob'], .8), 't4': (p2, t4, .5), 't4@w0.8': (p2, t4, .8),
                'vlm30-late': (p1, with_vlm(st['prob'], v30), .5), 'vlm30-t4': (p2, with_vlm(t4, v30), .5),
                'vlm30-t4@w0.8': (p2, with_vlm(t4, v30), .8), 'vlm30-t4flow': (p2, with_vlm(t4f, v30), .5),
                'vlm30-t4flow@w0.8': (p2, with_vlm(t4f, v30), .8)}
        rows += evaluate(st, arms); print('fold', fold, 'done', flush=True)
    def gate(c):
        return bool(c['mean_rp_delta'] > .001 and c['cells_improved'] >= 4 and min(c['per_fold_rp_delta'].values()) >= 0 and c['mean_map_delta'] >= 0)
    primary = compare(rows, 'vlm30-t4', 't4'); primary['passes'] = gate(primary)
    vs_combo = compare(rows, 'vlm30-t4', 'combo')
    w = compare([r for r in rows if r['fold'] in (1, 2)], 'vlm30-t4@w0.8', 'vlm30-t4')
    flow = compare(rows, 'vlm30-t4flow', 'vlm30-t4'); flow['passes'] = gate(flow)
    atomic_json(out / 'results/cells.json', rows)
    finish(out, plan, {'kind': '30B VLM + t4 on development folds 0-2', 'accuracy_score': True, 'gate_passed': primary['passes'],
           'primary_vlm30_t4_vs_t4': primary, 'vlm30_t4_vs_combo': vs_combo, 'worth_entry_signal': bool(primary['passes'] and vs_combo['mean_rp_delta'] >= .02),
           'weight_0_8_folds_1_2': w, 'adopt_weight_0_8': bool(min(w['per_fold_rp_delta'].values()) > 0),
           'flow_vs_no_flow': flow, 'deltas_vs_combo': {a: compare(rows, a, 'combo') for a in sorted({r['arm'] for r in rows}) if a != 'combo'},
           'summary': summarise(rows), 'seconds': time.monotonic() - started})


def vlmx_job(out, plan):
    """Scaled-up 30B recipe versus the first 30B recipe on fold 0 (pre-registered 2026-10-20 before training):
    the scaled recipe (all atoms including ego, 48 questions per clip, about two epochs, LoRA rank 32) replaces the first if
    vlmx-t4 beats vlm30-t4 by >= +0.005 RP with both head seeds improving; folds 1-2 then confirm. Ego AUC is reported
    for the scaled 30B against t4 (motion CNN fused with probes)."""
    started = time.monotonic(); store = np.load(verified_result(ROOT / plan['flow_run'][0], plan['flow_run'][1], 'motion-flow.npz'))
    st = fold_state(plan, 0, store); p1, p2 = st['p1'], st['p2']; held_ids = st['ix']['held_video_ids']

    def load(root, job, families):
        v = np.load(verified_result(ROOT / root, job, 'vlm-atom-probs.npz'))
        if list(v['video_ids']) != list(held_ids):
            raise ValueError('VLM probabilities are not aligned with held videos: ' + job)
        return {str(a): np.clip(v['probs'][:, j], 1e-4, 1 - 1e-4) for j, a in enumerate(v['atoms']) if a.split('|')[0] in families}
    v1 = load(*plan['vlm30_first'], tuple(plan.get('first_families', ('agent', 'light', 'obj'))))
    vx = load(*plan['vlm30_scaled'], ('agent', 'light', 'obj', 'ego'))
    vx_nonego = {a: p for a, p in vx.items() if not a.startswith('ego')}
    t4 = with_ego(st, 'm2')
    arms = {'combo': (p1, st['prob'], .5), 't4': (p2, t4, .5), 'vlm30-t4': (p2, with_vlm(t4, v1), .5),
            'vlmx-t4': (p2, with_vlm(t4, vx), .5), 'vlmx-t4-nonego': (p2, with_vlm(t4, vx_nonego), .5),
            'vlmx-t4@w0.8': (p2, with_vlm(t4, vx), .8), 'vlm30-t4@w0.8': (p2, with_vlm(t4, v1), .8)}
    rows = evaluate(st, arms); atoms = sorted(vx)
    fams = {'probes': family_auc(st, st['prob'], atoms), 't4': family_auc(st, t4, atoms), 'vlm30-first': family_auc(st, v1, atoms),
            'vlm30-scaled': family_auc(st, vx, atoms), 'scaled+t4 (ego late fusion)': family_auc(st, with_vlm(t4, vx), atoms)}
    w = plan.get('primary_weight', 0.5)  # 0.8 from 2026-10-21 (best weight for both models on folds 1-2)
    c = compare(rows, 'vlmx-t4' if w == 0.5 else f'vlmx-t4@w{w}', 'vlm30-t4' if w == 0.5 else f'vlm30-t4@w{w}')
    replace = bool(c['mean_rp_delta'] >= plan['replace_gain'] and c['cells_improved'] == c['cells'])
    atomic_json(out / 'results/cells.json', rows)
    finish(out, plan, {'kind': plan['kind'], 'baseline': plan['vlm30_first'], 'candidate': plan['vlm30_scaled'], 'accuracy_score': True, 'gate_passed': replace,
           'scaled_replaces_first': replace, 'vlmx_t4_vs_vlm30_t4': c, 'family_auc': fams,
           'deltas_vs_combo': {a: compare(rows, a, 'combo') for a in arms if a != 'combo'}, 'summary': summarise(rows),
           'seconds': time.monotonic() - started})


def audit_job(out, plan):
    """Single audit confirmation (rules fixed 2026-10-20, before any audit scoring): every component is fitted on all 1,549
    development clips and judged on the 773 reserved audit queries over the 517 reserved audit clips.
    Arms: entry2 (the submitted structured recipe: combo at weight 0.5), t4, and the candidate (30B atoms late-fused into t4
    at the plan's fusion weight; the 30B was fine-tuned on all development clips). The candidate is confirmed for an entry if
    its mean RP gain over entry2 is >= +0.01 with both head seeds improving and mean MAP change >= 0.
    Optional reference (2026-10-21): a previously audited candidate's audit probabilities, fused the same way; the candidate
    replaces it if its mean RP gain over the reference is > +0.001 with both head seeds improving and MAP change >= 0."""
    dry = plan.get('dry_run_fold')  # development fold used to rehearse this exact code path without touching audit labels
    started = time.monotonic(); st = fold_state(plan, 'audit' if dry is None else dry, None)
    p1, p2 = st['p1'], st['p2']; held_ids = st['ix']['held_video_ids']
    root, job = plan['vlm_audit']; v = np.load(verified_result(ROOT / root, job, 'vlm-atom-probs.npz'))
    if list(v['video_ids']) != list(held_ids):
        raise ValueError('30B probabilities are not aligned with the audit clips')
    v30 = {str(a): np.clip(v['probs'][:, j], 1e-4, 1 - 1e-4) for j, a in enumerate(v['atoms']) if a.split('|')[0] in plan['vlm_families']}
    t4 = with_ego(st, 'm2'); w = plan['candidate_weight']
    arms = {'entry2': (p1, st['prob'], .5), 't4': (p2, t4, .5), 'candidate': (p2, with_vlm(t4, v30), w)}
    if w != .5:
        arms['candidate@w0.5'] = (p2, with_vlm(t4, v30), .5)  # reported, not gating
    extra = {}
    if plan.get('reference_audit'):
        rv = np.load(verified_result(ROOT / plan['reference_audit'][0], plan['reference_audit'][1], 'vlm-atom-probs.npz'))
        if list(rv['video_ids']) != list(held_ids):
            raise ValueError('reference probabilities are not aligned with the audit clips')
        ref = {str(a): np.clip(rv['probs'][:, j], 1e-4, 1 - 1e-4) for j, a in enumerate(rv['atoms']) if a.split('|')[0] in plan['vlm_families']}
        arms['reference'] = (p2, with_vlm(t4, ref), w)
    rows = evaluate(st, arms); atoms = sorted(v30)
    c = compare(rows, 'candidate', 'entry2')
    confirmed = bool(c['mean_rp_delta'] >= plan['confirm_gain'] and c['cells_improved'] == c['cells'] and c['mean_map_delta'] >= 0)
    if 'reference' in arms:
        r = compare(rows, 'candidate', 'reference')
        extra = {'reference': plan['reference_audit'], 'candidate_vs_reference': r, 'reference_vs_entry2': compare(rows, 'reference', 'entry2'),
                 'candidate_replaces_reference': bool(confirmed and r['mean_rp_delta'] > .001 and r['cells_improved'] == r['cells'] and r['mean_map_delta'] >= 0)}
    atomic_json(out / 'results/cells.json', rows)
    finish(out, plan, {'kind': 'single audit-video confirmation' if dry is None else f'audit code rehearsal on development fold {dry}',
           'accuracy_score': True, 'audit_confirmation': dry is None, 'audit_used': dry is None,
           'gate_passed': confirmed, 'candidate_confirmed': confirmed, 'candidate_vs_entry2': c,
           'candidate_vs_t4': compare(rows, 'candidate', 't4'), 't4_vs_entry2': compare(rows, 't4', 'entry2'),
           'family_auc': {'probes': family_auc(st, st['prob'], atoms), 'vlm30': family_auc(st, v30, atoms)},
           'summary': summarise(rows), 'recipe': plan['recipe'], 'audit_queries': len(st['ix']['dev_query_ids']),
           'audit_clips': len(held_ids), 'seconds': time.monotonic() - started, **extra})


def vlm7_job(out, plan):
    """Folds 0-2 confirmation of the improved 30B recipe (rules fixed 2026-10-20 before folds 1-2 were trained):
    recipe   cand-t4 versus vlm30-t4 (first recipe): standing gate over 6 fold-seed cells (mean RP gain > 0.001, >= 4/6
             improve, no fold regresses on average, MAP change >= 0).
    weight   fusion weight 0.8 adopted for the audit if cand-t4@w0.8 beats cand-t4 on both fold 1 and fold 2 means
             (second look at folds 1-2 after the first recipe's test; noted as such)."""
    started = time.monotonic(); rows = []
    for fold in (0, 1, 2):
        st = fold_state(plan, fold, None); p2 = st['p2']; held_ids = st['ix']['held_video_ids']

        def load(spec, families):
            v = np.load(verified_result(ROOT / spec[0], spec[1], 'vlm-atom-probs.npz'))
            if list(v['video_ids']) != list(held_ids):
                raise ValueError('VLM probabilities misaligned on fold %d: %s' % (fold, spec[1]))
            return {str(a): np.clip(v['probs'][:, j], 1e-4, 1 - 1e-4) for j, a in enumerate(v['atoms']) if a.split('|')[0] in families}
        first = load(plan['first_jobs'][str(fold)], tuple(plan.get('first_families', ('agent', 'light', 'obj'))))
        cand = load(plan['candidate_jobs'][str(fold)], tuple(plan['candidate_families']))
        t4 = with_ego(st, 'm2')
        arms = {'combo': (st['p1'], st['prob'], .5), 't4': (p2, t4, .5), 'vlm30-t4': (p2, with_vlm(t4, first), .5),
                'vlm30-t4@w0.8': (p2, with_vlm(t4, first), .8), 'cand-t4': (p2, with_vlm(t4, cand), .5),
                'cand-t4@w0.8': (p2, with_vlm(t4, cand), .8)}
        rows += evaluate(st, arms); print('fold', fold, 'done', flush=True)
    def gate(c):
        return bool(c['mean_rp_delta'] > .001 and c['cells_improved'] >= 4 and min(c['per_fold_rp_delta'].values()) >= 0 and c['mean_map_delta'] >= 0)
    w = plan.get('primary_weight', 0.5)  # 0.8 from 2026-10-21
    recipe = compare(rows, 'cand-t4' if w == 0.5 else f'cand-t4@w{w}', 'vlm30-t4' if w == 0.5 else f'vlm30-t4@w{w}'); recipe['passes'] = gate(recipe)
    w = compare([r for r in rows if r['fold'] in (1, 2)], 'cand-t4@w0.8', 'cand-t4')
    atomic_json(out / 'results/cells.json', rows)
    finish(out, plan, {'kind': f"{plan['candidate_name']} 30B recipe over folds 0-2", 'accuracy_score': True, 'gate_passed': recipe['passes'],
           'candidate_name': plan['candidate_name'], 'candidate_vs_first': recipe, 'weight_0_8_folds_1_2': w,
           'adopt_weight_0_8': bool(min(w['per_fold_rp_delta'].values()) > 0),
           'deltas_vs_combo': {a: compare(rows, a, 'combo') for a in sorted({r['arm'] for r in rows}) if a != 'combo'},
           'summary': summarise(rows), 'seconds': time.monotonic() - started})


def vlmw_job(out, plan):
    """Time-localized 30B versus the scaled 30B on fold 0 (rules fixed 2026-10-20 before training): the candidate (clip answer
    late-fused with the maximum window answer for action atoms, clip answer otherwise, fused into t4 at weight 0.5) replaces
    the scaled recipe if it beats vlmx-t4 by >= +0.005 RP with both head seeds improving; folds 1-2 would then confirm.
    Agent-action and ego-action query-weighted AUC are reported for both."""
    started = time.monotonic(); st = fold_state(plan, 0, None); p2 = st['p2']; held_ids = st['ix']['held_video_ids']
    def npz(spec):
        v = np.load(verified_result(ROOT / spec[0], spec[1], 'vlm-atom-probs.npz'))
        if list(v['video_ids']) != list(held_ids):
            raise ValueError('VLM probabilities are not aligned with held videos: ' + spec[1])
        return v
    vx = npz(plan['scaled']); scaled = {str(a): np.clip(vx['probs'][:, j], 1e-4, 1 - 1e-4) for j, a in enumerate(vx['atoms'])}
    w = npz(plan['windowed']); clip = {str(a): np.clip(w['probs'][:, j], 1e-4, 1 - 1e-4) for j, a in enumerate(w['atoms'])}
    wmax = {str(a): np.clip(w['window_probs'][:, :, k].max(1), 1e-4, 1 - 1e-4) for k, a in enumerate(w['window_atoms'])}
    comb = {a: (late([clip[a], wmax[a]]) if a in wmax else clip[a]) for a in clip}
    t4 = with_ego(st, 'm2')
    arms = {'combo': (st['p1'], st['prob'], .5), 't4': (p2, t4, .5), 'scaled-t4': (p2, with_vlm(t4, scaled), .5),
            'windowed-clip-t4': (p2, with_vlm(t4, clip), .5), 'windowed-max-t4': (p2, with_vlm(t4, {**clip, **wmax}), .5),
            'candidate': (p2, with_vlm(t4, comb), .5), 'scaled-t4@w0.8': (p2, with_vlm(t4, scaled), .8), 'candidate@w0.8': (p2, with_vlm(t4, comb), .8)}
    rows = evaluate(st, arms); held = np.asarray(st['ix']['held_v'])
    def qw(p, pick):
        vals = [(auc(p[a], st['truth'][a][held]), st['weights'].get(a, 0)) for a in p if pick(a) and a in st['truth']]
        vals = [(v, x) for v, x in vals if np.isfinite(v) and x]
        return float(np.average([v for v, _ in vals], weights=[x for _, x in vals])) if vals else None
    groups = {'agent actions': lambda a: a.startswith('agent') and 'action=' in a, 'ego actions': lambda a: a.startswith('ego'),
              'all atoms': lambda a: True}
    sources = {'scaled': scaled, 'windowed clip': clip, 'windowed combined': comb,
               'scaled+t4': with_vlm(t4, scaled), 'candidate (combined+t4)': with_vlm(t4, comb)}
    aucs = {g: {name: qw(src, f) for name, src in sources.items()} for g, f in groups.items()}
    c = compare(rows, 'candidate', 'scaled-t4'); replace = bool(c['mean_rp_delta'] >= plan['replace_gain'] and c['cells_improved'] == c['cells'])
    atomic_json(out / 'results/cells.json', rows)
    finish(out, plan, {'kind': 'time-localized 30B versus scaled 30B on fold 0', 'accuracy_score': True, 'gate_passed': replace,
           'windowed_replaces_scaled': replace, 'candidate_vs_scaled': c, 'query_weighted_auc': aucs,
           'deltas_vs_combo': {a: compare(rows, a, 'combo') for a in arms if a != 'combo'}, 'summary': summarise(rows),
           'seconds': time.monotonic() - started})


def rerank_matrix(scores, verifier, top_k, weight):
    """Reorder each row's top-K by rank fusion of the base order (`weight`) and the verifier (1 - weight); others unchanged."""
    from baseline.native_ranking import blended_scores
    out = scores.copy()
    for i in range(len(scores)):
        order = np.argsort(-scores[i], kind='stable')[:top_k]
        blended = blended_scores(scores[i, order][None], verifier[i, order][None].astype(np.float32), weight)[0]
        out[i, order] = scores[i].max() + 1 + blended
    return out


def vlmq_job(out, plan):
    """Multi-task answerer (atoms + query verifier + mirror) versus the previous Qwen3.8 pilot on fold 0 (rules fixed
    2026-10-21 before the run; base weight changed from 0.95 to 0.8 before the run, after folds 1-2 showed weight 0.95
    losing to 0.8 by 0.013 RP, 0/4 cells): candidate = mirror-averaged atoms fused into t4 at weight 0.8, then the top 50 of
    each query re-ranked by the verifier (rank fusion, weight 0.5 on the base order). It replaces the previous Qwen3.8 at
    weight 0.8 if it gains >= +0.005 RP with both head seeds improving. Contributions (mirror averaging, verifier alone,
    re-ranking depth and weight, whole-list fusion, weight 0.95 variants) are reported, not gating."""
    from baseline.hubness import ranking_metrics
    from scripts.tracks_eval import fused, structured
    started = time.monotonic(); st = fold_state(plan, 0, None); ix = st['ix']; dev, held_ids = ix['dev_query_ids'], ix['held_video_ids']
    def npz(spec):
        v = np.load(verified_result(ROOT / spec[0], spec[1], 'vlm-atom-probs.npz'))
        if list(v['video_ids']) != list(held_ids):
            raise ValueError('Probabilities are not aligned with held videos: ' + spec[1])
        return v
    prev = npz(plan['previous']); new = npz(plan['multitask'])
    if list(new['query_ids']) != list(dev):
        raise ValueError('Verifier rows are not aligned with the development queries')
    as_dict = lambda v, key: {str(a): np.clip(v[key][:, j], 1e-4, 1 - 1e-4) for j, a in enumerate(v['atoms'])}
    t4 = with_ego(st, 'm2'); verifier = new['verifier'].astype(np.float32).T  # [queries, held clips]
    sources = {'previous': as_dict(prev, 'probs'), 'mt-original': as_dict(new, 'probs_original'), 'mt-mirror-avg': as_dict(new, 'probs')}
    rows = []
    for seed, (raw, sub) in st['subs'].items():
        arms = {}
        for name, prob in sources.items():
            s, m = structured(raw, dev, st['p2'], with_vlm(t4, prob))
            for w in (.8, .95):
                arms[f'{name}@{w}'] = fused(raw, sub, s, m, w)
        base = arms['mt-mirror-avg@0.8']
        arms['verifier-only'] = verifier
        for k in (30, 50, 100):
            for lam in (.3, .5, .7):
                arms[f'candidate-top{k}-w{lam}'] = rerank_matrix(base, verifier, k, lam)
        arms['candidate-top50-w0.5@base0.95'] = rerank_matrix(arms['mt-mirror-avg@0.95'], verifier, 50, .5)
        from baseline.native_ranking import blended_scores
        arms['whole-list-fusion-w0.5'] = blended_scores(base, verifier, .5)
        for name, sc in arms.items():
            r = ranking_metrics(sc, dev, held_ids, st['qrels'])
            rows.append({'fold': 0, 'seed': seed, 'arm': name, 'r_precision': r['r_precision'], 'map': r['map']})
    primary = 'candidate-top50-w0.5'
    c = compare(rows, primary, 'previous@0.8'); replace = bool(c['mean_rp_delta'] >= plan['replace_gain'] and c['cells_improved'] == c['cells'])
    contributions = {'mirror averaging (atoms only)': compare(rows, 'mt-mirror-avg@0.8', 'mt-original@0.8'),
                     'multi-task atoms versus previous (no verifier)': compare(rows, 'mt-mirror-avg@0.8', 'previous@0.8'),
                     'verifier re-rank on top of new atoms': compare(rows, primary, 'mt-mirror-avg@0.8'),
                     'weight 0.95 variant versus previous at 0.95': compare(rows, 'candidate-top50-w0.5@base0.95', 'previous@0.95')}
    atomic_json(out / 'results/cells.json', rows)
    finish(out, plan, {'kind': 'multi-task answerer versus previous Qwen3.8 on fold 0', 'accuracy_score': True, 'gate_passed': replace,
           'candidate_replaces_previous': replace, 'candidate_vs_previous': c, 'contributions': contributions,
           'summary': summarise(rows), 'seconds': time.monotonic() - started})


def vlmq3_job(out, plan):
    """Multi-task answerer over folds 0-2 (rules fixed 2026-10-21 before folds 1-2 were trained): candidate = mirror-averaged
    atoms fused into t4 at weight 0.8, top 50 of each query re-ranked by the verifier (weight 0.5 on the base order), versus
    Qwen3.8 + t4 at weight 0.8 on the same fold; standing gate (mean RP gain > 0.001, >= 4/6 cells, no fold regressing on
    average, MAP change >= 0). Component comparisons are reported."""
    from baseline.hubness import ranking_metrics
    from scripts.tracks_eval import fused, structured
    started = time.monotonic(); rows = []
    for fold in (0, 1, 2):
        st = fold_state(plan, fold, None); ix = st['ix']; dev, held_ids = ix['dev_query_ids'], ix['held_video_ids']
        def npz(spec):
            v = np.load(verified_result(ROOT / spec[0], spec[1], 'vlm-atom-probs.npz'))
            if list(v['video_ids']) != list(held_ids):
                raise ValueError('Probabilities are not aligned with held videos: ' + spec[1])
            return v
        prev = npz(plan['previous_jobs'][str(fold)]); new = npz(plan['multitask_jobs'][str(fold)])
        if list(new['query_ids']) != list(dev):
            raise ValueError('Verifier rows are not aligned with the development queries')
        as_dict = lambda v, key: {str(a): np.clip(v[key][:, j], 1e-4, 1 - 1e-4) for j, a in enumerate(v['atoms'])}
        t4 = with_ego(st, 'm2'); verifier = new['verifier'].astype(np.float32).T
        for seed, (raw, sub) in st['subs'].items():
            arms = {}
            for name, prob in (('previous', as_dict(prev, 'probs')), ('mt-mirror-avg', as_dict(new, 'probs'))):
                s, m = structured(raw, dev, st['p2'], with_vlm(t4, prob)); arms[f'{name}@0.8'] = fused(raw, sub, s, m, .8)
            arms['candidate'] = rerank_matrix(arms['mt-mirror-avg@0.8'], verifier, 50, .5)
            for name, sc in arms.items():
                r = ranking_metrics(sc, dev, held_ids, st['qrels'])
                rows.append({'fold': fold, 'seed': seed, 'arm': name, 'r_precision': r['r_precision'], 'map': r['map']})
        print('fold', fold, 'done', flush=True)
    def gate(c):
        return bool(c['mean_rp_delta'] > .001 and c['cells_improved'] >= 4 and min(c['per_fold_rp_delta'].values()) >= 0 and c['mean_map_delta'] >= 0)
    c = compare(rows, 'candidate', 'previous@0.8'); c['passes'] = gate(c)
    atomic_json(out / 'results/cells.json', rows)
    finish(out, plan, {'kind': 'multi-task answerer versus Qwen3.8 over folds 0-2', 'accuracy_score': True, 'gate_passed': c['passes'],
           'candidate_vs_previous': c, 'atoms_only_vs_previous': compare(rows, 'mt-mirror-avg@0.8', 'previous@0.8'),
           'verifier_rerank_vs_atoms_only': compare(rows, 'candidate', 'mt-mirror-avg@0.8'), 'summary': summarise(rows),
           'seconds': time.monotonic() - started})


JOBS = {'flow': flow_job, 'vlm': vlm_job, 'vlm3': vlm3_job, 'vlmx': vlmx_job, 'audit': audit_job, 'vlm7': vlm7_job, 'vlmw': vlmw_job, 'vlmq': vlmq_job, 'vlmq3': vlmq3_job}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', choices=JOBS, required=True)
    parser.add_argument('--plan', type=Path, required=True); parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); os.chdir(ROOT)
    import torch
    plan = json.loads(args.plan.read_text()); plan['plan_path'] = str(args.plan)
    if not plan['development_only'] or plan['official_submission_allowed']:
        raise ValueError('Only private local experiments are supported')
    torch.set_num_threads(plan['cpu_threads']); stage_directory(args.out)
    JOBS[args.job](args.out, plan)


if __name__ == '__main__':
    main()
