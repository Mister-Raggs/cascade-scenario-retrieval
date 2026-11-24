"""Query-side-only LoRA and query-instruction variants scored against frozen video vectors."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from baseline.io import sha256_file
from baseline.post_submission import fold_indices, positives, fit_heads, evaluate, paired_gain
from scripts.feature_retrieval import atomic_json, load_vectors, save_vectors
from scripts.post_submission import stage_directory, verified_result
from scripts import lora_pilot


def finish(out, plan, report):
    report.update(official_evaluation=False, audit_used=bool(report.get('audit_used', False)), validation_used=bool(report.get('validation_used', False)),
                  code_sha256=sha256_file(Path(__file__)), plan_sha256=sha256_file(Path(plan['plan_path'])))
    atomic_json(out / 'results/report.json', report)
    subprocess.run([sys.executable, 'scripts/spaces.py', 'upload', '--source', str(out / 'results'),
                    '--prefix', plan['backup_prefix'] + '/' + out.name + '/results',
                    '--receipt', str(out / 'receipt.json')], check=True)
    atomic_json(out / 'status.json', {'state': 'complete', 'backup_verified': True,
                'gate_passed': report['gate_passed'], 'report_sha256': sha256_file(out / 'results/report.json')})
    print(json.dumps({k: v for k, v in report.items() if k not in ('losses', 'cells')}), flush=True)


def frozen(plan):
    root = Path(plan['source_run'])
    fm = json.loads(verified_result(root, 'frozen-features', 'feature-manifest.json').read_text())
    q = load_vectors(verified_result(root, 'frozen-features', 'queries.npz'))
    v = load_vectors(verified_result(root, 'frozen-features', 'videos.npz'))
    manifest = json.loads(verified_result(root, 'prepare-lora', 'input-manifest.json').read_text())
    if q.shape[0] != len(fm['query_ids']) or v.shape[0] != len(fm['video_ids']):
        raise ValueError('Frozen feature alignment changed')
    return fm, q, v, manifest['ambiguous_negative_exclusions']


def training_masks(fit_query_ids, fit_video_ids, qrels, excluded):
    """Known positives versus allowed negatives; ambiguous reviewed pairs are neither."""
    pos = positives(fit_query_ids, fit_video_ids, qrels)
    vp = {vid: i for i, vid in enumerate(fit_video_ids)}
    ambiguous = np.zeros_like(pos)
    for i, qid in enumerate(fit_query_ids):
        for vid in excluded.get(qid, []):
            if vid in vp:
                ambiguous[i, vp[vid]] = True
    neg = ~pos & ~ambiguous
    return pos, neg, pos.any(1) & neg.any(1)


def comparison_gate(rows, minimum_gain):
    """Mean gain, at least two thirds of cells improving, no regressing fold and nonnegative MAP."""
    delta = np.asarray([r['r_precision_delta'] for r in rows])
    folds = sorted({r['fold'] for r in rows})
    fold_means = [float(delta[[r['fold'] == f for r in rows]].mean()) for f in folds]
    passed = (float(delta.mean()) > minimum_gain and int((delta > 0).sum()) >= math.ceil(len(rows) * 2 / 3)
              and min(fold_means) >= 0 and float(np.mean([r['map_delta'] for r in rows])) >= 0)
    return {'mean_rp_delta': float(delta.mean()), 'mean_map_delta': float(np.mean([r['map_delta'] for r in rows])),
            'positive_cells': int((delta > 0).sum()), 'cells': len(rows), 'fold_mean_rp_deltas': fold_means,
            'gate_passed': bool(passed)}


def query_loss(q, videos, positive, negative, anchors, settings):
    from baseline.lora import multi_positive_loss
    contrast = multi_positive_loss(q @ videos.T / settings['temperature'], positive, negative)
    return contrast + settings['anchor_weight'] * (q - anchors).square().sum(1).mean()


def query_inputs(encoder, texts, instruction):
    conversations = [encoder.e.format_model_input(text=text, instruction=instruction) for text in texts]
    text = encoder.e.processor.apply_chat_template(conversations, tokenize=False, add_generation_prompt=True)
    batch = encoder.e.processor(text=text, images=None, videos=None, padding=True,
                                truncation=False, do_resize=False, return_tensors='pt')
    if batch['attention_mask'].sum(1).max() > 2048:
        raise ValueError('Query token budget exceeded')
    return batch


def train_query_lora(out, plan):
    import torch
    from safetensors.torch import save_file
    from baseline.lora import attach_language_lora, adapter_state
    from baseline.native_video import QUERY_INSTRUCTION
    a = plan['adapter']; instruction = plan['instructions']['canonical']
    if instruction != QUERY_INSTRUCTION:
        raise ValueError('Adapter must train under the canonical query instruction')
    parent, queries, qrels = lora_pilot.inputs(plan)
    fm, qref, vref, excluded = frozen(plan)
    ix = fold_indices(parent, qrels, fm['query_ids'], fm['video_ids'], a['held_fold'])
    pos, neg, keep = training_masks(ix['fit_query_ids'], ix['fit_video_ids'], qrels, excluded)
    rows = np.flatnonzero(keep)
    encoder = lora_pilot.TrainingEncoder(plan); encoder.model.eval()
    probe = [queries[ix['fit_query_ids'][i]]['text'] for i in rows[:8]]
    with torch.no_grad():
        before = encoder.encode(query_inputs(encoder, probe, instruction)).detach().clone()
    torch.manual_seed(plan['seed'])  # Controls adapter initialization as well as training order.
    attach_language_lora(encoder.model, a['rank'], a['alpha'])
    with torch.no_grad():
        drift = float((encoder.encode(query_inputs(encoder, probe, instruction)) - before).abs().max())
    if drift > 1e-6:
        raise ValueError('Zero adapter changes the base encoder')
    initial = adapter_state(encoder.model)
    frozen_versions = {n: p._version for n, p in encoder.model.named_parameters() if not p.requires_grad}
    parameters = [p for p in encoder.model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=a['learning_rate'], weight_decay=a['weight_decay'])
    # Held-fold videos are absent from these targets; video vectors stay fixed throughout.
    vt = torch.as_tensor(vref[ix['fit_v']], device='cuda')
    pt = torch.as_tensor(pos, device='cuda'); nt = torch.as_tensor(neg, device='cuda')
    anchors = qref[ix['fit_q']]
    rng = np.random.default_rng(plan['seed']); torch.manual_seed(plan['seed'])
    torch.cuda.reset_peak_memory_stats(); losses, norms = [], []; started = time.monotonic()
    for step in range(a['optimizer_steps']):
        batch = rng.choice(rows, a['query_batch'], replace=False)
        optimizer.zero_grad(set_to_none=True); total = 0.0
        # Per-query mean losses, so share-weighted microbatches reproduce the full-batch objective.
        for chunk in np.array_split(batch, math.ceil(len(batch) / a.get('microbatch', len(batch)))):
            q = encoder.encode(query_inputs(encoder, [queries[ix['fit_query_ids'][i]]['text'] for i in chunk], instruction))
            loss = query_loss(q, vt, pt[chunk], nt[chunk], torch.as_tensor(anchors[chunk], device='cuda'), a)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite query-adapter loss')
            (loss * len(chunk) / len(batch)).backward(); total += float(loss.detach()) * len(chunk) / len(batch)
            del q, loss
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1)
        if not torch.isfinite(norm) or norm <= 0:
            raise ValueError('Missing/nonfinite adapter gradient')
        optimizer.step(); losses.append(total); norms.append(float(norm))
        peak = torch.cuda.max_memory_reserved()
        if peak >= a['max_reserved_gib'] * 1024 ** 3:
            raise ValueError('Query adapter exceeded its memory budget')
        if (step + 1) % 10 == 0:
            atomic_json(out / 'progress.json', {'optimizer_steps': step + 1, 'total': a['optimizer_steps'],
                        'elapsed_seconds': time.monotonic() - started, 'peak_reserved_bytes': peak})
    final = adapter_state(encoder.model)
    changed = any(not torch.equal(value, initial[name]) for name, value in final.items())
    untouched = all(p._version == frozen_versions[n] and p.grad is None
                    for n, p in encoder.model.named_parameters() if n in frozen_versions)
    if not changed or not untouched:
        raise ValueError('Adapter/base-weight correctness gate failed')
    save_file(final, str(out / 'results/adapter.safetensors'))
    finish(out, plan, {'kind': 'fixed-budget query-side-only LoRA over frozen fitting-video vectors', 'gate_passed': True,
           'accuracy_score': False, 'optimizer_steps': a['optimizer_steps'], 'training_queries': int(len(rows)),
           'fitting_videos': len(ix['fit_video_ids']), 'held_fold': a['held_fold'], 'zero_adapter_drift': drift,
           'adapter_changed': changed, 'frozen_weights_unchanged': untouched, 'losses': losses, 'gradient_norms': norms,
           'seconds': time.monotonic() - started, 'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
           'checkpoint_selection': 'Only the prescribed final step; held labels never select a step.'})


def encode_queries(out, plan):
    import torch
    from safetensors.torch import load_file
    from baseline.lora import attach_language_lora, load_adapter_state
    _, queries, _ = lora_pilot.inputs(plan)
    fm, qref, _, _ = frozen(plan)
    texts = [queries[qid]['text'] for qid in fm['query_ids']]
    encoder = lora_pilot.TrainingEncoder(plan); encoder.model.eval()
    size = plan['encode_query_batch']; started = time.monotonic()

    def run(name, instruction):
        parts = []
        with torch.no_grad():
            for i in range(0, len(texts), size):
                parts.append(encoder.encode(query_inputs(encoder, texts[i:i + size], instruction)).cpu().numpy())
                atomic_json(out / 'progress.json', {'arm': name, 'completed': min(i + size, len(texts)), 'total': len(texts)})
        vectors = np.concatenate(parts); save_vectors(out / 'results' / f'queries-{name}.npz', vectors)
        return vectors

    arms = {}
    for name, instruction in plan['instructions'].items():
        vectors = run(name, instruction)
        arms[name] = {'instruction': instruction, 'max_abs_drift_from_frozen': float(np.abs(vectors - qref).max())}
    parity = arms['canonical']['max_abs_drift_from_frozen']
    attach_language_lora(encoder.model, plan['adapter']['rank'], plan['adapter']['alpha'])
    load_adapter_state(encoder.model, load_file(str(verified_result(out.parent, 'train-query-lora', 'adapter.safetensors'))))
    vectors = run('query-lora', plan['instructions']['canonical'])
    arms['query-lora'] = {'instruction': plan['instructions']['canonical'],
                          'max_abs_drift_from_frozen': float(np.abs(vectors - qref).max())}
    atomic_json(out / 'results/query-manifest.json', {'query_ids': fm['query_ids'], 'arms': arms,
                'files': {f'queries-{n}.npz': sha256_file(out / 'results' / f'queries-{n}.npz') for n in arms},
                'device': torch.cuda.get_device_name(0)})
    finish(out, plan, {'kind': 'query-only re-encoding; video vectors reused from the frozen arm', 'accuracy_score': False,
           'gate_passed': parity <= plan['parity_max_abs_drift'], 'canonical_parity_max_abs_drift': parity,
           'arms': arms, 'seconds': time.monotonic() - started})


def arm_cells(name, qarm, v, ixs, qrels, plan, folds):
    cells = []
    for fold in folds:
        ix = ixs[fold]
        labels = positives(ix['fit_query_ids'], ix['fit_video_ids'], qrels)
        for seed in plan['head_seeds']:
            state = fit_heads(qarm[ix['fit_q']], v[ix['fit_v']], labels, plan['head_training'], seed,
                              [plan['head_epochs']], plan['head_device'])[plan['head_epochs']]
            for weight in plan['raw_weights']:
                cells.append({'arm': name, 'fold': fold, 'seed': seed, 'raw_weight': weight,
                              **evaluate(qarm[ix['dev_q']], v[ix['held_v']], ix['dev_query_ids'], ix['held_video_ids'],
                                         qrels, state, weight, plan['head_device'])})
    return cells


def evaluate_query_side(out, plan):
    parent, _, qrels = lora_pilot.inputs(plan)
    fm, qref, v, _ = frozen(plan)
    manifest = json.loads(verified_result(out.parent, 'encode-queries', 'query-manifest.json').read_text())
    if manifest['query_ids'] != fm['query_ids']:
        raise ValueError('Re-encoded query order differs from frozen arm')
    folds = sorted(set(plan['instruction_folds']) | {plan['adapter']['held_fold']})
    ixs = {f: fold_indices(parent, qrels, fm['query_ids'], fm['video_ids'], f) for f in folds}
    arms = {'query-lora': [plan['adapter']['held_fold']]}
    arms.update({n: plan['instruction_folds'] for n in plan['instructions'] if n != 'canonical'})
    cells = arm_cells('frozen', qref, v, ixs, qrels, plan, folds); started = time.monotonic()
    for name, arm_folds in arms.items():
        q = load_vectors(verified_result(out.parent, 'encode-queries', f'queries-{name}.npz'))
        cells += arm_cells(name, q, v, ixs, qrels, plan, arm_folds)
    base = {(c['fold'], c['seed'], c['raw_weight']): c for c in cells if c['arm'] == 'frozen'}
    summary = {}
    for name in arms:
        comparisons = []
        for c in (c for c in cells if c['arm'] == name):
            change = paired_gain(c, base[(c['fold'], c['seed'], c['raw_weight'])], plan['minimum_rp_gain'])
            comparisons.append({'fold': c['fold'], 'seed': c['seed'], 'raw_weight': c['raw_weight'],
                                'r_precision': c['r_precision'], 'map': c['map'], **change})
        summary[name] = {str(w): comparison_gate([r for r in comparisons if r['raw_weight'] == w], plan['minimum_rp_gain'])
                         for w in plan['raw_weights']}
        summary[name]['comparisons'] = comparisons
        summary[name]['gate_passed'] = summary[name][str(plan['primary_raw_weight'])]['gate_passed']
    reference = plan.get('reference_comparison')
    if reference:
        # Pre-registered cross-recipe test: adapted queries with one scoring versus the submitted frozen recipe.
        rows = []
        for c in (c for c in cells if c['arm'] == 'query-lora' and c['raw_weight'] == reference['arm_raw_weight']):
            change = paired_gain(c, base[(c['fold'], c['seed'], reference['frozen_raw_weight'])], plan['minimum_rp_gain'])
            rows.append({'fold': c['fold'], 'seed': c['seed'], **change})
        reference = {**reference, **comparison_gate(rows, plan['minimum_rp_gain']), 'comparisons': rows}
    atomic_json(out / 'results/cells.json', cells)
    finish(out, plan, {'kind': 'paired query-side arms versus frozen queries over held-out videos', 'accuracy_score': True,
           'gate_passed': reference['gate_passed'] if reference else any(s['gate_passed'] for s in summary.values()),
           'arms': summary, 'reference_comparison': reference,
           'primary_raw_weight': plan['primary_raw_weight'], 'seconds': time.monotonic() - started,
           'limits': 'Reused training-side development; the adapter uses one fold and one training seed. Choosing among '
                     'several instruction arms on the same folds is itself a selection; a passing arm needs separate confirmation.'})


JOBS = {'train-query-lora': train_query_lora, 'encode-queries': encode_queries, 'evaluate-query-side': evaluate_query_side}


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
    torch.set_num_threads(plan['cpu_threads'])
    stage_directory(args.out)
    JOBS[args.job](args.out, plan)


if __name__ == '__main__':
    main()
