"""Fit the packaged entry-3 t4 components on all 1,549 development clips and prove parity with the audited research path.

Components (development labels only; audit clips excluded from fitting, as in the audit run):
  probes  pooled + motion2 + window linear probes for every parser v1/v2 atom of the released train and validation query
          texts (rare atoms use the prior), in the entry-2 probe format
  ego     the motion-series CNN on the 13 ego atoms of the audited run, trained exactly as research (seed 0, 40 epochs),
          plus per-atom logistic calibration on inner cross-fitted logits (3 inner folds, seed 3)
Parity (computed on the audit clips' research features, with the audit run's own Qwen3.8 probabilities; no metric is
computed): probe and t4 probabilities versus scripts/perception_v3_eval.py fold_state('audit'), and the final weight-0.8
rankings of all 773 audit queries for both head seeds.
Usage: fit_entry3.py --out DIR
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from baseline.annotation_events import load_events
from baseline.dense import rank_scores
from baseline.entry3_recipe import FUSION_WEIGHT, fuse, motion_net, structured_scores, t4_probabilities, with_vlm
from baseline.io import sha256_file, unique_rows
from baseline.probe import fit_logistic, standardize
from baseline.query_struct import atom_present
from baseline.structured_recipe import BLOCKS
from scripts.feature_retrieval import atomic_json
from scripts.perception_v3_eval import fold_state, with_ego, with_vlm as research_with_vlm
from scripts.post_submission import verified_result
from scripts.structured_eval import provider
from scripts.structured_v2 import inner_folds, motionnet_logits
from scripts.tracks_eval import fused, parse, structured, translate_v1, translate_v2

PLAN = ROOT / 'configs/audit-q38/evaluator.json'
VLM = ['runs/audit-v2', 'audit-vlm-score']


def train_motionnet(series, fit, labels, seed=0, epochs=40):
    """scripts/structured_v2.py motionnet_logits, returning the trained modules and normalization instead of held logits."""
    import torch
    torch.manual_seed(seed)
    x = torch.as_tensor(series, dtype=torch.float32)
    mean, std = x[fit].mean((0, 1), keepdim=True), x[fit].std((0, 1), keepdim=True) + 1e-6
    x = ((x - mean) / std).transpose(1, 2)
    y = torch.as_tensor(labels, dtype=torch.float32); prior = y[fit].mean(0).clamp(1e-3, 1 - 1e-3)
    pos_weight = ((1 - prior) / prior).clamp(max=20)
    conv, head = motion_net(x.shape[1], labels.shape[1])
    params = list(conv.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-2)
    forward = lambda b: head(torch.cat([conv(b).amax(2), conv(b).mean(2)], 1))
    rng = np.random.default_rng(seed); fit_t = torch.as_tensor(fit)
    for _ in range(epochs):
        for chunk in np.array_split(rng.permutation(len(fit)), max(1, len(fit) // 64)):
            ix = fit_t[chunk]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(forward(x[ix]), y[ix], pos_weight=pos_weight)
            opt.zero_grad(); loss.backward(); opt.step()
    return conv, head, mean.reshape(-1).numpy(), std.reshape(-1).numpy(), lambda rows: forward(x[torch.as_tensor(rows)]).detach().numpy()


def series_matrix(store, vids, rows):
    """scripts/perception_v3_eval.py fold_state series(): every clip truncated to the shortest stored series."""
    length = min(store[v].shape[0] for v in store.files); width = store[store.files[0]].shape[1]
    out = np.zeros((len(vids), length, width), np.float32); present = set(store.files)
    if any(vids[i] not in present for i in rows):
        raise ValueError('Motion series missing for a fold video')
    for i, v in enumerate(vids):
        if v in present:
            out[i] = store[v][:length]
    return out, length


def main():
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--all-clips', action='store_true', help='entry 4: fit on all 2,066 annotated clips (development + audit); no parity reference remains')
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=False); started = time.monotonic()
    plan = json.loads(PLAN.read_text())
    st = fold_state(plan, 'audit', None)  # research reference, exactly as the audit evaluator built it
    vids, ix = st['vids'], st['ix']; fit, held = np.asarray(ix['fit_v']), np.asarray(ix['held_v'])
    parent = json.loads((ROOT / plan['partition_parent']).read_text())
    if sorted(vids[i] for i in fit) != sorted(v for g in parent['development_video_groups'] for v in g):
        raise ValueError('Fitting clips are not exactly the development clips')
    if set(vids[i] for i in fit) & set(v for g in parent['audit_video_groups'] for v in g):
        raise ValueError('Audit clips in the fitting set')
    if a.all_clips:  # same code path as the audited components, fitted on every annotated clip
        fit = np.sort(np.concatenate([fit, held])); held = np.asarray([], dtype=fit.dtype)

    # Probes for every parser v1/v2 atom of the released query texts (texts only; no qrels).
    texts = {q: r['text'] for s in ('train', 'val') for q, r in unique_rows(ROOT / f'data/retrieval/cascade-v1.0/{s}/queries.jsonl', 'query_id').items()}
    vocabulary = sorted({x for p in (parse(texts, translate_v1), parse(texts, translate_v2)) for v in p.values() if v for x in v})
    events = load_events(ROOT / plan['annotation_directory']); sets = {v: {e[0] for e in events[v]['events']} for v in vids}
    truth = np.asarray([[atom_present(x, sets[vids[i]]) for x in vocabulary] for i in fit], dtype=bool)
    positives = truth.sum(0); is_prior = (positives < 3) | (positives > len(fit) - 3)
    out = {'atoms': np.asarray(vocabulary), 'prior': np.clip(truth.mean(0), 1e-3, 1 - 1e-3), 'is_prior': is_prior}
    blocks = {b: provider(b, plan['blocks'][b], vids)[0] for b in BLOCKS}
    for b in BLOCKS:
        x = blocks[b]; z = standardize(x, fit)
        weight = np.zeros((x.shape[1], len(vocabulary)), np.float32); bias = np.zeros(len(vocabulary), np.float32)
        for j in range(len(vocabulary)):
            if not is_prior[j]:
                w, c = fit_logistic(z[fit], truth[:, j]); weight[:, j] = w; bias[j] = c
        out.update({f'{b}_mean': x[fit].mean(0), f'{b}_std': x[fit].std(0) + 1e-6, f'{b}_weight': weight, f'{b}_bias': bias})
        print(f'{b}: {x.shape[1]} dims, {int((~is_prior).sum())} fitted atoms', flush=True)

    # Ego CNN on the audited run's ego atoms, then calibration on inner cross-fitted logits.
    ego = st['ego']; y = np.stack([st['truth'][e] for e in ego], 1)
    m2 = np.load(ROOT / plan['blocks']['motion2']['path']); series, length = series_matrix(m2, vids, np.concatenate([fit, held]))
    conv, head, mean, std, predict = train_motionnet(series, fit, y, plan['seed'], plan['motionnet_epochs'])
    logits = np.zeros((len(vids), len(ego)), np.float32); cnn_exact = None
    if not a.all_clips:
        reference = motionnet_logits(series, fit, held, y, plan['seed'], plan['motionnet_epochs'])
        held_logits = predict(held); cnn_exact = bool(np.array_equal(held_logits, reference)); logits[held] = held_logits
    for part in inner_folds(fit, plan['inner_folds'], plan['seed'] + 3):
        logits[part] = motionnet_logits(series, np.setdiff1d(fit, part), part, y, plan['seed'], plan['motionnet_epochs'])
    cal_mean, cal_std = logits[fit].mean(0), logits[fit].std(0) + 1e-6; z = (logits - cal_mean) / cal_std
    cal = {'weight': np.zeros((len(ego), len(ego))), 'bias': np.zeros(len(ego)), 'prior': np.zeros(len(ego)), 'is_prior': np.zeros(len(ego), bool)}
    for j, e in enumerate(ego):
        yf = st['truth'][e][fit]
        if yf.sum() < 3 or yf.sum() > len(yf) - 3:
            cal['is_prior'][j] = True; cal['prior'][j] = float(np.clip(yf.mean(), 1e-3, 1 - 1e-3))
        else:
            w, c = fit_logistic(z[fit], yf); cal['weight'][j] = w; cal['bias'][j] = c
    out.update({'ego_atoms': np.asarray(ego), 'ego_series_length': np.asarray(length), 'ego_mean': mean, 'ego_std': std,
                'ego_conv0_weight': conv[0].weight.detach().numpy(), 'ego_conv0_bias': conv[0].bias.detach().numpy(),
                'ego_conv2_weight': conv[2].weight.detach().numpy(), 'ego_conv2_bias': conv[2].bias.detach().numpy(),
                'ego_head_weight': head.weight.detach().numpy(), 'ego_head_bias': head.bias.detach().numpy(),
                'ego_cal_mean': cal_mean, 'ego_cal_std': cal_std, **{f'ego_cal_{k}': v for k, v in cal.items()}})
    path = a.out / 'entry3_atoms.npz'; np.savez_compressed(path, **out)

    if a.all_clips:
        manifest = {'kind': 'packaged entry-4 t4 components (all annotated clips)', 'atoms_sha256': sha256_file(path), 'atoms': len(vocabulary),
                    'prior_atoms': int(is_prior.sum()), 'ego_atoms': ego, 'ego_series_length': length, 'training_videos': int(len(fit)),
                    'audit_videos_used_for_fitting': len([v for g in parent['audit_video_groups'] for v in g]),
                    'plan': str(PLAN.relative_to(ROOT)), 'plan_sha256': sha256_file(PLAN),
                    'parity': {'passed': True, 'note': ('no held-out reference remains; same code path as the audited components, whose '
                                                        'parity passed (runs/entry3/components)')},
                    'annotations_shipped': False, 'seconds': time.monotonic() - started}
        atomic_json(a.out / 'entry3_manifest.json', manifest); print(json.dumps({k: manifest[k] for k in ('atoms', 'training_videos', 'audit_videos_used_for_fitting')}), flush=True)
        return

    # Parity on the audit clips' research features (probabilities and rankings only).
    feats = {b: blocks[b][held] for b in BLOCKS}; pkg = t4_probabilities(path, feats, [series[i] for i in held])
    research_t4 = with_ego(st, 'm2'); shared = sorted(set(research_t4) & set(pkg))
    missing = sorted(set(research_t4) - set(pkg))
    diff = {k: float(max(np.abs(pkg[x] - research_t4[x]).max() for x in shared if (x.startswith('ego') == (k == 'ego')))) for k in ('ego', 'other')}
    v = np.load(verified_result(ROOT / VLM[0], VLM[1], 'vlm-atom-probs.npz'))
    if list(v['video_ids']) != list(ix['held_video_ids']):
        raise ValueError('Audit Qwen3.8 probabilities are not aligned with the audit clips')
    vlm = {str(x): np.clip(v['probs'][:, j], 1e-4, 1 - 1e-4) for j, x in enumerate(v['atoms'])}
    dev_ids = ix['dev_query_ids']; query_texts = [texts[q] for q in dev_ids]; ranking = {}
    for seed, (raw, sub) in st['subs'].items():
        want = fused(raw, sub, *structured(raw, dev_ids, st['p2'], research_with_vlm(research_t4, vlm)), FUSION_WEIGHT)
        got = fuse(raw, sub, *structured_scores(query_texts, with_vlm(pkg, vlm)), FUSION_WEIGHT)
        names = list(ix['held_video_ids'])
        exact = sum(rank_scores(names, g)[:10] == rank_scores(names, w)[:10] for g, w in zip(got, want))
        ranking[str(seed)] = {'top10_exact': int(exact), 'queries': len(dev_ids), 'max_abs_score_diff': float(np.abs(got - want).max())}
    passed = cnn_exact and not missing and max(diff.values()) < 1e-4 and all(r['top10_exact'] >= r['queries'] - 2 for r in ranking.values())
    manifest = {'kind': 'packaged entry-3 t4 components', 'atoms_sha256': sha256_file(path), 'atoms': len(vocabulary),
                'prior_atoms': int(is_prior.sum()), 'ego_atoms': ego, 'ego_series_length': length, 'training_videos': int(len(fit)),
                'audit_videos_used_for_fitting': 0, 'plan': str(PLAN.relative_to(ROOT)), 'plan_sha256': sha256_file(PLAN),
                'parity': {'cnn_held_logits_bit_exact': cnn_exact, 'research_atoms_missing_from_package': missing,
                           'max_abs_probability_diff': diff, 'final_rankings': ranking, 'passed': passed},
                'annotations_shipped': False, 'seconds': time.monotonic() - started}
    atomic_json(a.out / 'entry3_manifest.json', manifest); print(json.dumps(manifest['parity'], indent=1), flush=True)
    if not passed:
        raise SystemExit('Entry-3 component parity failed')


if __name__ == '__main__':
    main()
