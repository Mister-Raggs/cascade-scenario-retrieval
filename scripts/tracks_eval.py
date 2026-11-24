"""Tracks 1-4 judged against the confirmed combo recipe on one development fold (pre-registered pilot gate).

Jobs (fold f from plan['held_fold']):
  encode-synthetic  GPU: 8B query vectors for the synthetic training queries (Track 1)
  stage1            CPU: combo probabilities and top-K candidates (union over head seeds) for the reranker
  stage2            CPU: arms combo, t1-synthetic-head, t2-rerank, t3-vlmft, t4-motionnet-parser2, t14-synthetic-motionnet-parser2,
                    t134-all; gates (t2/t3/t134 only when the plan names the reranker/VLM jobs)
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
from baseline.annotation_events import load_events
from baseline.hubness import ranking_metrics
from baseline.io import sha256_file, unique_rows
from baseline.native_ranking import blended_scores
from baseline.post_submission import fold_indices, positives, fit_heads, project, paired_gain
from baseline.query_parser import translate as translate_v1, Unparsed
from baseline.query_parser_v2 import translate as translate_v2
from baseline.query_struct import tree, atoms, atom_present
from scripts.feature_retrieval import atomic_json, load_vectors, save_vectors
from scripts.post_submission import stage_directory, verified_result
from scripts.query_side import finish
from scripts.structured_eval import provider
from scripts.structured_v2 import COMBO, inner_folds, late, logistic_prob, motionnet_logits
from scripts import lora_pilot


def parse(texts, fn):
    out = {}
    for q, t in texts.items():
        try:
            out[q] = sorted(set(atoms(tree(fn(t)))))
        except Unparsed:
            out[q] = None
    return out


def common(plan):
    parent, queries, qrels = lora_pilot.inputs(plan)
    fm = json.loads((ROOT / plan['feature_directory'] / 'feature_manifest.json').read_text()); vids = fm['video_ids']
    ix = fold_indices(parent, qrels, fm['query_ids'], vids, plan['held_fold'])
    texts = {q: queries[q]['text'] for q in ix['dev_query_ids']}
    events = load_events(ROOT / plan['annotation_directory']); sets = {v: {e[0] for e in events[v]['events']} for v in vids}
    return parent, queries, qrels, fm, vids, ix, texts, sets


def combo_probs(plan, vids, ix, sets, needed):
    blocks = {b: provider(b, plan['blocks'][b], vids) for b in COMBO}
    fit, held = np.asarray(ix['fit_v']), np.asarray(ix['held_v'])
    for b, (_, mask) in blocks.items():
        if not mask[fit].all() or not mask[held].all():
            raise ValueError('Fold uses videos without features: ' + b)
    truth = {a: np.asarray([atom_present(a, sets[v]) for v in vids]) for a in needed}
    return {a: late([logistic_prob(blocks[b][0], fit, held, truth[a]) for b in COMBO]) for a in needed}, truth


def structured(raw, dev_ids, parsed, prob):
    s = np.zeros_like(raw); m = np.zeros(len(dev_ids), bool)
    for i, q in enumerate(dev_ids):
        known = [a for a in (parsed.get(q) or []) if a in prob]
        if known:
            s[i] = np.sum([np.log(prob[a]) for a in known], axis=0); m[i] = True
    return s, m


def fused(raw, submitted, s, m, weight):
    return np.where(m[:, None], blended_scores(np.where(m[:, None], s, raw), submitted, weight), submitted)


def submitted_scores(plan, qvec, vvec, ix, qrels, seed, extra=None):
    fq, fv = qvec[ix['fit_q']], vvec[ix['fit_v']]
    labels = positives(ix['fit_query_ids'], ix['fit_video_ids'], qrels)
    if extra is not None:
        xq, xlabels = extra; keep = xlabels.any(1)
        fq = np.concatenate([fq, xq[keep]]); labels = np.concatenate([labels, xlabels[keep]])
    state = fit_heads(fq, fv, labels, plan['head_training'], seed, [plan['head_epochs']], 'cpu')[plan['head_epochs']]
    dq, vh = qvec[ix['dev_q']], vvec[ix['held_v']]; raw = dq @ vh.T
    return raw, blended_scores(raw, project(dq, state) @ project(vh, state).T, plan['submitted_raw_weight'])


def encode_synthetic(out, plan):
    import torch
    from scripts.query_side import query_inputs
    rows = unique_rows(ROOT / plan['synthetic']['queries'], 'query_id'); ids = list(rows)
    if sha256_file(ROOT / plan['synthetic']['queries']) != plan['synthetic']['queries_sha256']:
        raise ValueError('Synthetic queries changed')
    encoder = lora_pilot.TrainingEncoder(plan); encoder.model.eval(); parts = []
    with torch.no_grad():
        for i in range(0, len(ids), 16):
            parts.append(encoder.encode(query_inputs(encoder, [rows[q]['text'] for q in ids[i:i + 16]], plan['query_instruction'])).cpu().numpy())
            if i % 1600 == 0:
                atomic_json(out / 'progress.json', {'completed': i, 'total': len(ids)})
    save_vectors(out / 'results/synthetic-queries.npz', np.concatenate(parts)); atomic_json(out / 'results/synthetic-ids.json', ids)
    finish(out, plan, {'kind': 'synthetic query vectors (8B, canonical instruction)', 'gate_passed': True, 'accuracy_score': False, 'queries': len(ids)})


def stage1(out, plan):
    parent, queries, qrels, fm, vids, ix, texts, sets = common(plan)
    qvec = load_vectors(ROOT / plan['feature_directory'] / 'queries-train.npz'); vvec = load_vectors(ROOT / plan['feature_directory'] / 'aggregate.npz')
    parsed = parse(texts, translate_v1); needed = sorted({a for v in parsed.values() if v for a in v})
    prob, _ = combo_probs(plan, vids, ix, sets, needed); dev_ids = ix['dev_query_ids']; held_ids = ix['held_video_ids']
    candidates = {q: set() for q in dev_ids}
    for seed in plan['head_seeds']:
        raw, sub = submitted_scores(plan, qvec, vvec, ix, qrels, seed)
        scores = fused(raw, sub, *structured(raw, dev_ids, parsed, prob), plan['fusion_weight'])
        for i, q in enumerate(dev_ids):
            candidates[q].update(held_ids[j] for j in np.argsort(-scores[i], kind='stable')[:plan['top_k']])
    atomic_json(out / 'results/candidates.json', {q: {'text': texts[q], 'videos': sorted(v)} for q, v in candidates.items()})
    np.savez_compressed(out / 'results/combo-probs.npz', atoms=np.asarray(needed), probs=np.stack([prob[a] for a in needed], 1),
                        held=np.asarray(held_ids))
    finish(out, plan, {'kind': 'combo probabilities and reranker candidates', 'gate_passed': True, 'accuracy_score': False,
           'candidate_pairs': sum(len(v) for v in candidates.values()), 'fold': plan['held_fold']})


def rerank(scores, dev_ids, held_ids, table, top_k, weight):
    """Reorder each query's top-K by rank fusion: `weight` on the existing order, 1 - weight on reranker scores."""
    out = scores.copy()
    for i, q in enumerate(dev_ids):
        order = np.argsort(-scores[i], kind='stable')[:top_k]
        if q not in table or any(held_ids[j] not in table[q] for j in order):
            continue
        base = scores[i, order]; rr = np.asarray([table[q][held_ids[j]] for j in order])
        blended = blended_scores(base[None], rr[None], weight)[0]
        # Keep the block above every non-candidate while reordering inside it.
        out[i, order] = scores[i].max() + 1 + blended
    return out


def stage2(out, plan):
    names = plan['job_names']; started = time.monotonic()
    parent, queries, qrels, fm, vids, ix, texts, sets = common(plan)
    qvec = load_vectors(ROOT / plan['feature_directory'] / 'queries-train.npz'); vvec = load_vectors(ROOT / plan['feature_directory'] / 'aggregate.npz')
    s1 = np.load(verified_result(out.parent, names['stage1'], 'combo-probs.npz'))
    prob = {str(a): s1['probs'][:, j] for j, a in enumerate(s1['atoms'])}
    dev_ids, held_ids = ix['dev_query_ids'], ix['held_video_ids']; fit, held = np.asarray(ix['fit_v']), np.asarray(ix['held_v'])
    p1, p2 = parse(texts, translate_v1), parse(texts, translate_v2)
    extra_atoms = sorted({a for v in p2.values() if v for a in v} - set(prob))
    if extra_atoms:
        more, _ = combo_probs(plan, vids, ix, sets, extra_atoms); prob.update(more)
    # Track 4: cross-fitted motion CNN for ego atoms (as in the directions evaluator).
    truth = {a: np.asarray([atom_present(a, sets[v]) for v in vids]) for a in prob}
    store = np.load(ROOT / plan['blocks']['motion2']['path']); length = min(store[v].shape[0] for v in store.files)
    series = np.zeros((len(vids), length, 6), np.float32)
    for i, v in enumerate(vids):
        if v in store.files:
            series[i] = store[v][:length]
    ego = [a for a in prob if a.startswith('ego')]; ego_y = np.stack([truth[a] for a in ego], 1)
    inner = inner_folds(fit, plan['inner_folds'], plan['seed'] + plan['held_fold'])
    net = np.zeros((len(vids), len(ego)), np.float32)
    net[held] = motionnet_logits(series, fit, held, ego_y, plan['seed'], plan['motionnet_epochs'])
    for part in inner:
        net[part] = motionnet_logits(series, np.setdiff1d(fit, part), part, ego_y, plan['seed'], plan['motionnet_epochs'])
    t4prob = {a: (late([prob[a], logistic_prob(net, fit, held, truth[a])]) if a in ego else prob[a]) for a in prob}
    # Track 3: fine-tuned VLM atom probabilities for the held videos.
    # GPU arms are optional so CPU-only folds (no VLM or reranker job) can still judge tracks 1 and 4.
    vlm = {}
    if names.get('vlm_score'):
        v = np.load(verified_result(out.parent, names['vlm_score'], 'vlm-atom-probs.npz'))
        if list(v['video_ids']) != list(held_ids):
            raise ValueError('VLM probabilities are not aligned with held videos')
        vlm = {str(a): np.clip(v['probs'][:, j], 1e-4, 1 - 1e-4) for j, a in enumerate(v['atoms'])}
    t3prob = {a: (late([prob[a], vlm[a]]) if a in vlm else prob[a]) for a in prob}
    t34prob = {a: (late([t4prob[a], vlm[a]]) if a in vlm else t4prob[a]) for a in prob}
    # Track 1: synthetic queries added to head training (labels restricted to fitting videos by fit_heads inputs).
    sids = json.loads(verified_result(out.parent, names['encode_synthetic'], 'synthetic-ids.json').read_text())
    svec = load_vectors(verified_result(out.parent, names['encode_synthetic'], 'synthetic-queries.npz'))
    srels = unique_rows(ROOT / plan['synthetic']['qrels'], 'query_id')
    slabels = positives(sids, ix['fit_video_ids'], srels)
    table = json.loads(verified_result(out.parent, names['rerank_score'], 'reranker-scores.json').read_text()) if names.get('rerank_score') else None
    w = plan['fusion_weight']; cells = []
    for seed in plan['head_seeds']:
        raw, sub = submitted_scores(plan, qvec, vvec, ix, qrels, seed)
        _, sub_syn = submitted_scores(plan, qvec, vvec, ix, qrels, seed, (svec, slabels))
        combo = fused(raw, sub, *structured(raw, dev_ids, p1, prob), w)
        arms = {'submitted': sub, 'combo': combo,
                't1-synthetic-head': fused(raw, sub_syn, *structured(raw, dev_ids, p1, prob), w),
                't4-motionnet-parser2': fused(raw, sub, *structured(raw, dev_ids, p2, t4prob), w),
                't14-synthetic-motionnet-parser2': fused(raw, sub_syn, *structured(raw, dev_ids, p2, t4prob), w)}
        if table is not None:
            arms['t2-rerank'] = rerank(combo, dev_ids, held_ids, table, plan['top_k'], plan['rerank_weight'])
        if vlm:
            arms['t3-vlmft'] = fused(raw, sub, *structured(raw, dev_ids, p1, t3prob), w)
            arms['t134-all'] = fused(raw, sub_syn, *structured(raw, dev_ids, p2, t34prob), w)
            arms['t34-vlmft-motionnet-parser2'] = fused(raw, sub, *structured(raw, dev_ids, p2, t34prob), w)  # added after the pilot: t134 without t1
        for name, scores in arms.items():
            cells.append({'seed': seed, 'arm': name, **ranking_metrics(scores, dev_ids, held_ids, qrels)})
    base = {c['seed']: c for c in cells if c['arm'] == 'combo'}; summary = {}
    for arm in sorted({c['arm'] for c in cells}):
        rows = [{'seed': c['seed'], 'r_precision': c['r_precision'], 'map': c['map'], **paired_gain(c, base[c['seed']], plan['minimum_rp_gain'])}
                for c in cells if c['arm'] == arm]
        d = [r['r_precision_delta'] for r in rows]; dm = [r['map_delta'] for r in rows]
        summary[arm] = {'mean_rp': float(np.mean([r['r_precision'] for r in rows])), 'mean_map': float(np.mean([r['map'] for r in rows])),
                        'mean_rp_delta': float(np.mean(d)), 'mean_map_delta': float(np.mean(dm)),
                        'expand': bool(np.mean(d) >= plan['pilot_gain'] and min(d) > 0 and np.mean(dm) >= 0), 'comparisons': rows}
    atomic_json(out / 'results/cells.json', [{k: v for k, v in c.items() if k != 'per_query'} for c in cells])
    report = {'kind': f'tracks 1-4 pilot on development fold {plan["held_fold"]}', 'accuracy_score': True,
              'gate_passed': any(s['expand'] for a, s in summary.items() if a not in ('combo', 'submitted')), 'summary': summary,
              'reranker_projection': (json.loads(verified_result(out.parent, names['rerank_score'], 'report.json').read_text()).get('projected_test_hours_h100')
                                      if table is not None else None),
              'seconds': time.monotonic() - started,
              'limits': 'One reused development fold; expansion needs folds 1-2, then a single audit-video confirmation.'}
    for arm, s in summary.items():
        report[f'expand_{arm}'] = s['expand']
    finish(out, plan, report)


JOBS = {'encode-synthetic': encode_synthetic, 'stage1': stage1, 'stage2': stage2}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', choices=JOBS, required=True)
    parser.add_argument('--plan', type=Path, required=True); parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); os.chdir(ROOT)
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    import torch
    plan = json.loads(args.plan.read_text()); plan['plan_path'] = str(args.plan)
    if not plan['development_only'] or plan['official_submission_allowed']:
        raise ValueError('Only private local experiments are supported')
    torch.set_num_threads(plan['cpu_threads']); stage_directory(args.out)
    JOBS[args.job](args.out, plan)


if __name__ == '__main__':
    main()
