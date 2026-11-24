"""Whole-query verifier variants for entry 5 (2026-10-22; development folds only, GPU; screening, no gate, no audit).

The entry-5 headroom diagnostics (ENTRY5_HEADROOM_2026-10-22.md) put the verifier's within-candidate AUC at 0.69 and
measured how RP grows with it. This screens verifier variants on each query's top-30 candidates (the same
runs/learned-scorer/verify-candidates.json as the entry-4 verifier) by that AUC: mean over dev queries with both relevant
and irrelevant candidates. Promising variants then run on all folds and go through the learned scorer.

  score  --variant V --fold F   answers every candidate pair; writes runs/verifier-variants/V-fF/verify.npz + report.json
  train  --fold F               hard-negative verifier fine-tune: continues the fold's Qwen3.8 atom adapter on whole-query
                                verification only. Per step one fitting clip and 16 fitting queries: up to 4 relevant,
                                up to 8 hard negatives (the clip is in the query's top 40 fitting clips by raw 8B
                                similarity but not relevant), up to 2 atom near misses, the rest random; BCE on yes/no.
Variants are defined in VARIANTS (adapter, question template, frames and resolution).
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
OUT = ROOT / 'runs/verifier-variants'
ADAPTERS = {0: 'runs/tracks-v8/q38-train-f0', 1: 'runs/tracks-v9/q38-train-f1', 2: 'runs/tracks-v9/q38-train-f2', 'audit': 'runs/audit-v2/audit-vlm-train'}
T0 = 'Does this video match the following description? "{}"'
T1 = ('Watch the whole video carefully. Does everything in the following description happen in this video, with the same '
      'road users, the same actions and the same order of events? "{}"')
T2 = 'Is the following statement true for this video? "{}"'
F32 = {'frames': 32, 'frame_directory': 'runs/verifier-variants/frames32', 'question_batch': 32}
F64 = {'frames': 64, 'frame_directory': 'runs/verifier-variants/frames64', 'question_batch': 16, 'max_length': 8192}
F64LO = {**F64, 'height': 256, 'width': 448, 'question_batch': 32, 'max_length': 4096}
HIRES = {'frame_directory': 'runs/verifier-variants/frames16-w800', 'frame_width': 800, 'height': 448, 'width': 800, 'question_batch': 48}
VARIANTS = {
    'base': {'adapter': 'fold', 'template': T0},
    'noadapter': {'adapter': None, 'template': T0},
    'strict': {'adapter': 'fold', 'template': T1},
    'statement': {'adapter': 'fold', 'template': T2},
    'frames32': {'adapter': 'fold', 'template': T0, 'plan': F32},
    'hires': {'adapter': 'fold', 'template': T0, 'plan': HIRES},
    'noadapter-strict': {'adapter': None, 'template': T1},
    'hardneg': {'adapter': 'hardneg', 'template': T0},
    'hardneg-frames32': {'adapter': 'hardneg', 'template': T0, 'plan': F32},
    'frames64': {'adapter': 'fold', 'template': T0, 'plan': F64},
    'frames64-lowres': {'adapter': 'fold', 'template': T0, 'plan': F64LO},
    'val-base': {'adapter': 'fold', 'template': T0, 'candidates': 'verify-candidates-val.json'},
    'val-frames32': {'adapter': 'fold', 'template': T0, 'plan': F32, 'candidates': 'verify-candidates-val.json'},
    'val-ft': {'adapter': 'mtv', 'template': T0, 'candidates': 'verify-candidates-val.json'},
    'c32': {'adapter': 'fold', 'template': T0, 'candidates': 'verify-candidates-32.json'},
    'val-c32': {'adapter': 'fold', 'template': T0, 'candidates': 'verify-candidates-val-32.json'},
    'audit-c32': {'adapter': 'fold', 'template': T0, 'candidates': 'verify-candidates-audit-32.json'},
    'ft-c50': {'adapter': 'mtv', 'template': T0, 'candidates': 'verify-candidates-32-50.json'},
    'val-ft-c50': {'adapter': 'mtv', 'template': T0, 'candidates': 'verify-candidates-val-32-50.json'},
    'audit-ft-c50': {'adapter': 'mtv', 'template': T0, 'candidates': 'verify-candidates-audit-32-50.json'},
    'zs-c50': {'adapter': 'fold', 'template': T0, 'candidates': 'verify-candidates-32-50.json'},
    'val-zs-c50': {'adapter': 'fold', 'template': T0, 'candidates': 'verify-candidates-val-32-50.json'},
    'audit-zs-c50': {'adapter': 'fold', 'template': T0, 'candidates': 'verify-candidates-audit-32-50.json'},
    'ft2-c50': {'adapter': 'mtv2', 'template': T0, 'candidates': 'verify-candidates-32-50.json'},
    'val-ft2-c50': {'adapter': 'mtv2', 'template': T0, 'candidates': 'verify-candidates-val-32-50.json'},
    'audit-ft2-c50': {'adapter': 'mtv2', 'template': T0, 'candidates': 'verify-candidates-audit-32-50.json'},
    'val-top50': {'adapter': 'fold', 'template': T0, 'candidates': 'verify-candidates-val50.json'},
    'val-mirror': {'adapter': 'fold', 'template': T0, 'candidates': 'verify-candidates-val.json', 'mirror': True},
    'frames32-top50': {'adapter': 'fold', 'template': T0, 'plan': F32, 'candidates': 'verify50-candidates.json'},
}
FRAMES = {'f16': {}, 'f32': {k: v for k, v in F32.items() if k != 'question_batch'}, 'f64': {k: v for k, v in F64.items() if k != 'question_batch'}}


def auc(score, label):
    label = np.asarray(label, bool); n1, n0 = label.sum(), (~label).sum()
    if not n1 or not n0:
        return None
    ranks = np.empty(len(score)); ranks[np.argsort(score, kind='stable')] = np.arange(1, len(score) + 1)
    return float((ranks[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def plan_for(fold, overrides=None):
    return {**json.loads((ROOT / 'configs/v9/q38.json').read_text()), 'held_fold': fold, **(overrides or {})}


def fold_adapter(fold):
    from scripts.post_submission import verified_result
    parent, job = ADAPTERS[fold].rsplit('/', 1)
    return verified_result(ROOT / parent, job, 'vlm-adapter.safetensors')


def candidate_auc(fold, qs, vs, logits):
    """Mean within-candidate AUC over the fold's dev queries, from the qrels."""
    from scripts.reranker_finetune import fold_data
    _, _, qrels, _, _ = fold_data(plan_for(fold))
    if any(str(q).startswith('cascade-v1.0_val') for q in qs[:1]):
        from baseline.io import unique_rows
        qrels = unique_rows(ROOT / 'data/retrieval/cascade-v1.0/val/qrels.jsonl', 'query_id')
    per = {}
    for q, v, z in zip(qs, vs, logits):
        per.setdefault(q, []).append((z, v in set(qrels[q]['relevant_video_ids'])))
    vals = [auc([z for z, _ in x], [r for _, r in x]) for x in per.values()]
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)), len(vals)


def score(variant, fold, limit=0):
    import torch
    from scripts.reranker_finetune import fold_data
    from scripts.vlm_multitask import MultiTaskAnswerer
    spec = VARIANTS[variant]; plan = plan_for(fold, spec.get('plan'))
    _, _, _, vids, ix = fold_data(plan); cand = json.loads((ROOT / 'runs/learned-scorer' / spec.get('candidates', 'verify-candidates.json')).read_text())
    per = cand['folds'][str(fold)]; texts = cand['texts']; clips = sorted(per)
    if not set(clips) <= {vids[i] for i in ix['held_v']}:
        raise ValueError('Verifier candidates outside the held clips')
    clips = clips[:limit] if limit else clips
    adapter = {'fold': lambda: fold_adapter(fold), None: lambda: None,
               'hardneg': lambda: OUT / f'hardneg-train-f{fold}/vlm-adapter.safetensors',
               'mtv2': lambda: ROOT / ('runs/entry5/mtv2-audit-train' if fold == 'audit' else f'runs/entry5/mtv2-f{fold}-train') / 'results/vlm-adapter.safetensors',
               'mtv': lambda: (ROOT / 'runs/entry5/mtv-audit-train/results/vlm-adapter.safetensors' if fold == 'audit'
                               else ROOT / f'runs/gpu-2026-10-22/mtv-train-f{fold}/results/vlm-adapter.safetensors')}[spec['adapter']]()
    model = MultiTaskAnswerer(plan, adapter); model.mode(False)
    out = OUT / f'{variant}-f{fold}'; out.mkdir(parents=True, exist_ok=True); started = time.monotonic()
    qs, vs, logits = [], [], []
    with torch.inference_mode():
        for n, v in enumerate(clips):
            if spec.get('mirror'):  # left-right mirrored frames, asked the left/right-swapped query (same content)
                from baseline.mirror import mirror_text
                z = model.answer(v, [spec['template'].format(mirror_text(texts[q])) for q in per[v]], True).float().cpu().numpy()
            else:
                z = model.answer(v, [spec['template'].format(texts[q]) for q in per[v]]).float().cpu().numpy()
            qs += per[v]; vs += [v] * len(per[v]); logits += z.tolist()
            if n % 32 == 0:
                (out / 'progress.json').write_text(json.dumps({'clips': n + 1, 'total': len(clips), 'seconds': time.monotonic() - started}))
    seconds = (time.monotonic() - started) / len(clips)
    name = 'verify.npz' if not limit else 'smoke.npz'
    np.savez_compressed(out / name, query_ids=np.asarray(qs), video_ids=np.asarray(vs), logits=np.asarray(logits, np.float32))
    mean_auc, queries = candidate_auc(fold, qs, vs, logits)
    report = {'variant': variant, 'fold': fold, 'spec': {k: v for k, v in spec.items()}, 'clips': len(clips), 'pairs': len(qs),
              'seconds_per_clip': seconds, 'peak_reserved_gib': torch.cuda.max_memory_reserved() / 2 ** 30,
              'mean_within_candidate_auc': mean_auc, 'auc_queries': queries, 'smoke': bool(limit),
              'note': 'User asked (2026-10-22) to test whole-query verifier variants on the H200 for entry 5. Development folds only.'}
    (out / ('report.json' if not limit else 'smoke.json')).write_text(json.dumps(report, indent=2)); print(json.dumps(report), flush=True)


def atoms(frames, fold):
    """The fold's 16-frame atom adapter answers every held clip's atom questions on a longer frame sampling (no retraining)."""
    import torch
    from baseline.probe import auc as probe_auc
    from scripts.post_submission import verified_result
    from scripts.reranker_finetune import fold_data
    from scripts.vlm_atom_finetune import labels
    from scripts.vlm_atom_v2 import answerer_for, plan_atoms
    plan = plan_for(fold, {**FRAMES[frames], 'question_batch': 96 if frames == 'f16' else 48})
    _, _, _, vids, ix = fold_data(plan); vocab, truth = labels(plan, vids); atom_list = plan_atoms(plan, vocab)
    parent, job = {0: ('runs/tracks-v8', 'q38-score-f0'), 1: ('runs/tracks-v9', 'q38-score-f1'), 2: ('runs/tracks-v9', 'q38-score-f2'),
                   'audit': ('runs/audit-v2', 'audit-vlm-score')}[fold]
    research = np.load(verified_result(ROOT / parent, job, 'vlm-atom-probs.npz')); held = list(ix['held_v'])
    if [str(a) for a in research['atoms']] != atom_list or list(research['video_ids']) != [vids[i] for i in held]:
        raise ValueError('Atom scoring is not aligned with the research answers')
    model = answerer_for(plan, fold_adapter(fold)); model.mode(False)
    out = OUT / f'atoms-{frames}-f{fold}'; out.mkdir(parents=True, exist_ok=True); started = time.monotonic()
    probs = np.zeros((len(held), len(atom_list)), np.float32)
    with torch.inference_mode():
        for n, i in enumerate(held):
            probs[n] = 1 / (1 + np.exp(-model.logits(vids[i], atom_list).float().cpu().numpy()))
            if n % 32 == 0:
                (out / 'progress.json').write_text(json.dumps({'clips': n + 1, 'total': len(held), 'seconds': time.monotonic() - started}))
    np.savez_compressed(out / 'vlm-atom-probs.npz', video_ids=np.asarray([vids[i] for i in held]), atoms=np.asarray(atom_list), probs=probs)
    fam = {}
    for j, a in enumerate(atom_list):
        y = truth[a][held]
        if 5 <= y.sum() <= len(y) - 5:
            fam.setdefault(a.split('|')[0] + ('-action' if 'action=' in a else ''), []).append(
                (probe_auc(probs[:, j], y), probe_auc(research['probs'][:, j], y)))
    report = {'frames': frames, 'fold': fold, 'clips': len(held), 'seconds_per_clip': (time.monotonic() - started) / len(held),
              'peak_reserved_gib': torch.cuda.max_memory_reserved() / 2 ** 30,
              'mean_auc_by_family': {k: {'new': float(np.mean([x for x, _ in v])), 'research_16f': float(np.mean([y for _, y in v])), 'atoms': len(v)}
                                     for k, v in fam.items()}}
    (out / 'report.json').write_text(json.dumps(report, indent=2)); print(json.dumps(report), flush=True)


def props(fold):
    """Zero-shot Qwen3.8 (the fold's atom adapter) answers the property questions on 32 frames for every held clip."""
    import torch
    from baseline.property_atoms import question as prop_question
    from scripts.reranker_finetune import fold_data
    from scripts.vlm_multitask import MultiTaskAnswerer
    plan = plan_for(fold, {**FRAMES['f32'], 'question_batch': 48}); _, _, _, vids, ix = fold_data(plan)
    held = [vids[i] for i in ix['held_v']]; atoms = json.loads((ROOT / 'runs/learned-scorer/props-all-atoms.json').read_text())
    model = MultiTaskAnswerer(plan, fold_adapter(fold)); model.mode(False)
    out = OUT / f'props32-f{fold}'; out.mkdir(parents=True, exist_ok=True); started = time.monotonic()
    probs = np.zeros((len(held), len(atoms)), np.float32)
    with torch.inference_mode():
        for n, v in enumerate(held):
            probs[n] = 1 / (1 + np.exp(-model.answer(v, [prop_question(a) for a in atoms]).float().cpu().numpy()))
            if n % 32 == 0:
                (out / 'progress.json').write_text(json.dumps({'clips': n + 1, 'total': len(held), 'seconds': time.monotonic() - started}))
    np.savez_compressed(out / 'props.npz', video_ids=np.asarray(held), atoms=np.asarray(atoms), probs=probs)
    print(json.dumps({'fold': fold, 'clips': len(held), 'atoms': len(atoms), 'seconds_per_clip': (time.monotonic() - started) / len(held)}), flush=True)


def hard_pools(plan, vids, ix, qrels, depth=40):
    """{fitting clip index: queries for which it is a hard negative} by raw 8B similarity among fitting clips."""
    from scripts.feature_retrieval import load_vectors
    qvec = load_vectors(ROOT / plan['feature_directory'] / 'queries-train.npz'); vvec = load_vectors(ROOT / plan['feature_directory'] / 'aggregate.npz')
    fit_q, fit_v = list(ix['fit_query_ids']), np.asarray(ix['fit_v']); raw = qvec[ix['fit_q']] @ vvec[fit_v].T
    pools = {int(i): [] for i in fit_v}
    for r, q in enumerate(fit_q):
        relevant = set(qrels[q]['relevant_video_ids'])
        for j in np.argsort(-raw[r])[:depth]:
            i = int(fit_v[j])
            if vids[i] not in relevant:
                pools[i].append(q)
    return pools


def train(fold, steps):
    import torch
    from safetensors.torch import save_file
    from baseline.lora import adapter_state
    from baseline.query_pairs import clip_query_pools
    from scripts.vlm_multitask import MultiTaskAnswerer, setup
    plan = plan_for(fold); d = setup(plan); vids, ix, qrels = d['vids'], d['ix'], d['qrels']
    fit = np.asarray(ix['fit_v']); rng = np.random.default_rng(0)
    near = clip_query_pools(d['fit_q'], qrels, d['parsed'], d['qtruth'], vids, fit)
    hard = hard_pools(plan, vids, ix, qrels)
    index = {v: i for i, v in enumerate(vids)}
    relevant_of = {q: {index[v] for v in qrels[q]['relevant_video_ids'] if v in index} for q in d['fit_q']}
    model = MultiTaskAnswerer(plan, fold_adapter(fold)); params = [p for p in model.model.parameters() if p.requires_grad]
    initial = adapter_state(model.model); model.mode(True)
    opt = torch.optim.AdamW(params, lr=plan['learning_rate'], weight_decay=.01)
    out = OUT / f'hardneg-train-f{fold}'; out.mkdir(parents=True, exist_ok=True)
    losses, started = [], time.monotonic()

    def take(items, k, exclude):
        items = [q for q in items if q not in exclude]; k = min(k, len(items))
        return [items[j] for j in rng.choice(len(items), k, replace=False)] if k else []
    order = rng.permutation(fit)
    for step in range(steps):
        i = int(order[step % len(order)])
        positives, misses = near[i]
        chosen = [(q, 1) for q in take(positives, 4, set())]
        used = {q for q, _ in chosen}
        chosen += [(q, 0) for q in take(hard[i], 8, used)]; used = {q for q, _ in chosen}
        chosen += [(q, 0) for q in take(misses, 2, used)]; used = {q for q, _ in chosen}
        while len(chosen) < 16:
            q = d['fit_q'][int(rng.integers(len(d['fit_q'])))]
            if i not in relevant_of[q] and q not in used:
                chosen.append((q, 0)); used.add(q)
        y = torch.tensor([float(l) for _, l in chosen], device='cuda')
        z = model.answer(vids[i], [T0.format(d['qtext'][q]) for q, _ in chosen])
        loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite loss')
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1); opt.step()
        losses.append(float(loss.detach()))
        if (step + 1) % 25 == 0:
            (out / 'progress.json').write_text(json.dumps({'optimizer_steps': step + 1, 'total': steps, 'recent_loss': float(np.mean(losses[-25:])),
                                                            'elapsed_seconds': time.monotonic() - started}))
    final = adapter_state(model.model)
    if not any(not torch.equal(v, initial[n]) for n, v in final.items()):
        raise ValueError('Adapter did not change')
    save_file(final, str(out / 'vlm-adapter.safetensors'))
    (out / 'report.json').write_text(json.dumps({'fold': fold, 'steps': steps, 'losses': losses, 'seconds': time.monotonic() - started,
        'hard_pool_median': float(np.median([len(v) for v in hard.values()])), 'held_videos_in_training': 0,
        'peak_reserved_gib': torch.cuda.max_memory_reserved() / 2 ** 30}, indent=1))
    print('trained', fold, steps, round(time.monotonic() - started), 's', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('job', choices=['score', 'train', 'atoms', 'props']); ap.add_argument('--frames', choices=['f16', 'f32', 'f64'], default='f32')
    ap.add_argument('--variant', choices=VARIANTS); ap.add_argument('--fold', type=lambda v: v if v == 'audit' else int(v), choices=[0, 1, 2, 'audit'], required=True)
    ap.add_argument('--limit', type=int, default=0); ap.add_argument('--steps', type=int, default=2500)
    a = ap.parse_args(); os.chdir(ROOT); os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    import torch
    torch.set_num_threads(8)
    if a.job == 'score':
        score(a.variant, a.fold, a.limit)
    elif a.job == 'atoms':
        atoms(a.frames, a.fold)
    elif a.job == 'props':
        props(a.fold)
    else:
        train(a.fold, a.steps)


if __name__ == '__main__':
    main()
