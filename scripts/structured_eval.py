"""Structured retrieval on held-out videos: parsed queries scored by atom probes, fused with the submitted recipe.

Feature providers are pluggable so every perception experiment is judged by the same retrieval harness.
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
from baseline.native_ranking import blended_scores
from baseline.post_submission import fold_indices, positives, fit_heads, project, paired_gain
from baseline.probe import fit_logistic, standardize
from baseline.query_parser import translate, Unparsed
from baseline.query_struct import tree, atoms, atom_present
from scripts.feature_retrieval import atomic_json, load_vectors
from scripts.post_submission import stage_directory
from scripts.query_side import comparison_gate
from scripts import lora_pilot


def _aligned(rows, ids, video_ids):
    """Place per-video rows at video_ids positions; videos without features get zero rows and a False mask."""
    index = {v: i for i, v in enumerate(ids)}
    rows = np.asarray(rows, dtype=np.float32)
    out = np.zeros((len(video_ids), rows.shape[1]), np.float32); mask = np.zeros(len(video_ids), bool)
    for j, v in enumerate(video_ids):
        if v in index:
            out[j] = rows[index[v]]; mask[j] = True
    return out, mask


def provider(name, spec, video_ids):
    """Return ([videos, dims] features aligned to video_ids, mask of videos that actually have features)."""
    if spec['kind'] == 'pooled-embedding':
        fm = json.loads((ROOT / spec['manifest']).read_text())
        return _aligned(load_vectors(ROOT / spec['vectors']), fm['video_ids'], video_ids)
    if spec['kind'] == 'motion-series':
        from baseline.egomotion import clip_vector
        store = np.load(ROOT / spec['path']); ids = list(store.files)
        return _aligned([clip_vector(store[v], spec.get('bins', 5)) for v in ids], ids, video_ids)
    if spec['kind'] == 'window-vectors':
        fm = json.loads((ROOT / spec['manifest']).read_text()); k = fm['windows']
        flat = load_vectors(ROOT / spec['vectors']); w = flat.reshape(-1, k, flat.shape[1])
        ids = [key.split('@')[0] for key in fm['keys'][::k]]
        return _aligned(np.concatenate([w.mean(1), w.max(1)], 1), ids, video_ids)
    if spec['kind'] == 'clip-matrix':
        data = np.load(ROOT / spec['path'])
        return _aligned(data['features'], list(data['video_ids']), video_ids)
    raise ValueError('Unknown feature provider: ' + name)


def feature_set(providers, video_ids, late=False):
    """Concatenated features (default) or, with late=True, a list of per-provider blocks for late probe fusion."""
    parts = [provider(n, s, video_ids) for n, s in providers.items()]
    mask = np.logical_and.reduce([k for _, k in parts])
    return ([m for m, _ in parts] if late else np.concatenate([m for m, _ in parts], 1)), mask


def parse_queries(query_ids, texts):
    parsed = {}
    for qid in query_ids:
        try:
            parsed[qid] = sorted(set(atoms(tree(translate(texts[qid])))))
        except Unparsed:
            parsed[qid] = None
    return parsed


def atom_probabilities(features, fit_rows, held_rows, labels):
    x = standardize(features, fit_rows); y = labels[fit_rows]
    if y.sum() < 3 or y.sum() > len(y) - 3:
        return np.full(len(held_rows), float(np.clip(y.mean(), 1e-3, 1 - 1e-3)))
    w, b = fit_logistic(x[fit_rows], y)
    return np.clip(1 / (1 + np.exp(-(x[held_rows] @ w + b))), 1e-4, 1 - 1e-4)


def late_probabilities(blocks, fit_rows, held_rows, labels):
    """Separate probe per feature block; average their log-odds (no fitted weights)."""
    logits = [np.log(p / (1 - p)) for p in (atom_probabilities(b, fit_rows, held_rows, labels) for b in blocks)]
    return np.clip(1 / (1 + np.exp(-np.mean(logits, axis=0))), 1e-4, 1 - 1e-4)


def run(out, plan):
    parent, queries, qrels = lora_pilot.inputs(plan)
    texts = {qid: row['text'] for qid, row in queries.items()}
    fm = json.loads((ROOT / plan['feature_directory'] / 'feature_manifest.json').read_text())
    qvec = load_vectors(ROOT / plan['feature_directory'] / 'queries-train.npz')
    vvec = load_vectors(ROOT / plan['feature_directory'] / 'aggregate.npz')
    vids = fm['video_ids']; vindex = {v: i for i, v in enumerate(vids)}
    late = plan.get('probe_fusion') == 'late'
    feature_sets = {'': feature_set(plan['providers'], vids, late)}
    if plan.get('baseline_providers'):
        feature_sets['baseline-'] = feature_set(plan['baseline_providers'], vids)
    events = load_events(ROOT / plan['annotation_directory'])
    labels_of = {v: {e[0] for e in events[v]['events']} for v in vids}
    if plan.get('evaluation_queries') == 'val':
        # Held-video validation: probes and heads still train on training data only; validation labels score the result.
        from baseline.io import sha256_file, unique_rows
        for name, digest in plan['validation_input_hashes'].items():
            if sha256_file(ROOT / name) != digest:
                raise ValueError('Frozen validation input changed: ' + name)
        val_rows = unique_rows(ROOT / 'data/retrieval/cascade-v1.0/val/queries.jsonl', 'query_id')
        eval_ids = list(val_rows); eval_texts = {q: r['text'] for q, r in val_rows.items()}
        eval_vec = load_vectors(ROOT / plan['validation_features'])
        eval_qrels = unique_rows(ROOT / 'data/retrieval/cascade-v1.0/val/qrels.jsonl', 'query_id')
        if eval_vec.shape != (len(eval_ids), qvec.shape[1]):
            raise ValueError('Validation query feature alignment changed')
    else:
        eval_ids = eval_texts = eval_vec = None; eval_qrels = qrels
    parsed = parse_queries(eval_ids or parent['query_groups']['dev'], eval_texts or texts)
    cells = []; started = time.monotonic(); coverage = None
    for fold in plan['folds']:
        ix = fold_indices(parent, qrels, fm['query_ids'], vids, fold)
        held, fit_v = ix['held_v'], ix['fit_v']
        needed = sorted({a for ks in parsed.values() if ks for a in ks})
        truth = {a: np.asarray([atom_present(a, labels_of[v]) for v in vids]) for a in needed}
        dev_ids, dq = (eval_ids, eval_vec) if eval_ids else (ix['dev_query_ids'], qvec[ix['dev_q']])
        vh = vvec[held]
        raw = dq @ vh.T
        parsed_mask = np.asarray([bool(parsed[qid]) for qid in dev_ids])
        structured_by_set = {}
        for prefix, (features, present) in feature_sets.items():
            if not present[np.asarray(fit_v)].all() or not present[np.asarray(held)].all():
                raise ValueError('Fold uses videos without features: ' + (prefix or 'primary'))
            score = late_probabilities if isinstance(features, list) else atom_probabilities
            prob = {a: score(features, np.asarray(fit_v), np.asarray(held), truth[a]) for a in needed}
            structured = np.zeros_like(raw)
            for i, qid in enumerate(dev_ids):
                if parsed[qid]:
                    structured[i] = np.sum([np.log(prob[a]) for a in parsed[qid]], axis=0)
            structured_by_set[prefix] = structured
        coverage = float(parsed_mask.mean())
        labels = positives(ix['fit_query_ids'], ix['fit_video_ids'], qrels)
        for seed in plan['head_seeds']:
            state = fit_heads(qvec[ix['fit_q']], vvec[fit_v], labels, plan['head_training'], seed, [plan['head_epochs']], 'cpu')[plan['head_epochs']]
            head = project(dq, state) @ project(vh, state).T
            submitted = blended_scores(raw, head, plan['submitted_raw_weight'])
            arms = {'frozen-raw': raw, 'submitted': submitted}
            for prefix, structured in structured_by_set.items():
                arms[prefix + 'structured'] = np.where(parsed_mask[:, None], structured, submitted)
                for w in plan['fusion_weights']:
                    fused = blended_scores(np.where(parsed_mask[:, None], structured, raw), submitted, w)
                    arms[f'{prefix}fusion-{w}'] = np.where(parsed_mask[:, None], fused, submitted)
            for name, scores in arms.items():
                metrics = ranking_metrics(scores, dev_ids, ix['held_video_ids'], eval_qrels)
                cells.append({'fold': fold, 'seed': seed, 'arm': name, **metrics,
                              'parsed_only_rp': float(np.nanmean([r['r_precision'] for r, m in zip(metrics['per_query'], parsed_mask)
                                                                  if m and r['r_precision'] is not None]))})
        atomic_json(out / 'progress.json', {'fold': fold, 'elapsed_seconds': time.monotonic() - started})
    reference = plan.get('reference_arm', 'submitted')
    base = {(c['fold'], c['seed']): c for c in cells if c['arm'] == reference}
    summary = {}
    for arm in sorted({c['arm'] for c in cells}):
        rows = []
        for c in (c for c in cells if c['arm'] == arm):
            rows.append({'fold': c['fold'], 'seed': c['seed'], 'r_precision': c['r_precision'], 'map': c['map'],
                         'parsed_only_rp': c['parsed_only_rp'], **paired_gain(c, base[(c['fold'], c['seed'])], plan['minimum_rp_gain'])})
        summary[arm] = {'mean_rp': float(np.mean([r['r_precision'] for r in rows])), 'mean_map': float(np.mean([r['map'] for r in rows])),
                        'mean_parsed_only_rp': float(np.mean([r['parsed_only_rp'] for r in rows])),
                        **comparison_gate(rows, plan['minimum_rp_gain']), 'comparisons': rows}
    atomic_json(out / 'results/cells.json', [{k: v for k, v in c.items() if k != 'per_query'} for c in cells])
    report = {'kind': 'structured retrieval versus submitted recipe on held-out videos', 'providers': list(plan['providers']),
              'primary_arm': plan['primary_arm'], 'reference_arm': reference, 'gate_passed': summary[plan['primary_arm']]['gate_passed'],
              'parse_coverage_dev': coverage, 'summary': summary, 'seconds': time.monotonic() - started,
              'official_evaluation': False, 'validation_used': False,
              'limits': 'Reused training-side development folds; atom probes trained on fitting-video annotations only.'}
    if plan.get('backup_prefix'):
        # Queue runs need the standard report, verified Spaces backup and status receipt.
        from scripts.query_side import finish
        finish(out, plan, report)
    else:
        atomic_json(out / 'results/report.json', report)
    print(json.dumps({k: (v if k != 'summary' else {a: {m: round(s[m], 4) for m in ('mean_rp', 'mean_map', 'mean_parsed_only_rp', 'mean_rp_delta')}
                                                    | {'gate': s['gate_passed'], 'positive_cells': s['positive_cells']}
                                                    for a, s in v.items()}) for k, v in report.items()}, indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True); parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); os.chdir(ROOT)
    import torch
    plan = json.loads(args.plan.read_text()); plan['plan_path'] = str(args.plan)
    if not plan['development_only'] or plan['official_submission_allowed']:
        raise ValueError('Only private local experiments are supported')
    torch.set_num_threads(plan['cpu_threads'])
    stage_directory(args.out)
    run(args.out, plan)


if __name__ == '__main__':
    main()
