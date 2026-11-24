"""Portable private cached-head comparisons and bounded LoRA accuracy pilot."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from baseline.io import sha256_file, unique_rows
from baseline.lora_data import sample_batch, FOCUS
from baseline.post_submission import fold_indices, positives, fit_heads, evaluate, select_head_recipe, paired_gain
from scripts.feature_retrieval import atomic_json, load_vectors, save_vectors
from scripts import lora_pilot


def finish(out, plan, report):
    report.update(official_evaluation=False, audit_used=False,
                  code_sha256=sha256_file(Path(__file__)),
                  plan_sha256=sha256_file(Path(plan['plan_path'])))
    report.setdefault('validation_used', False)
    atomic_json(out / 'results/report.json', report)
    subprocess.run([sys.executable, 'scripts/spaces.py', 'upload', '--source', str(out / 'results'),
                    '--prefix', plan['backup_prefix'] + '/' + out.name + '/results',
                    '--receipt', str(out / 'receipt.json')], check=True)
    atomic_json(out / 'status.json', {'state': 'complete', 'backup_verified': True,
                'gate_passed': report['gate_passed'], 'report_sha256': sha256_file(out / 'results/report.json')})
    print(json.dumps(report), flush=True)


def verified_result(root, job, name):
    folder = root / job
    receipt = json.loads((folder / 'receipt.json').read_text())
    path = folder / 'results' / name
    if sha256_file(path) != receipt['files'][name]['sha256']:
        raise ValueError('Stage input differs from its verified receipt: ' + str(path))
    return path


def cached_data(plan):
    parent, queries, qrels = lora_pilot.inputs(plan)
    folder = Path(plan['feature_directory'])
    fm = json.loads((folder / 'feature_manifest.json').read_text())
    if fm.get('attention_backend') != 'math':
        raise ValueError('Canonical math-attention features required')
    for name in ['aggregate.npz', 'queries-train.npz']:
        if sha256_file(folder / name) != fm['files'][name]:
            raise ValueError('Cached feature body changed')
    q, v = load_vectors(folder / 'queries-train.npz'), load_vectors(folder / 'aggregate.npz')
    if q.shape != (len(fm['query_ids']), fm['dimension']) or v.shape != (len(fm['video_ids']), fm['dimension']):
        raise ValueError('Feature dimensions differ from manifest')
    return parent, queries, qrels, fm['query_ids'], fm['video_ids'], q, v


def cached_heads(out, plan):
    from safetensors.torch import save_file
    parent, queries, qrels, qids, vids, q, v = cached_data(plan)
    cells = []; started = time.monotonic()
    for fold in plan['head_folds']:
        ix = fold_indices(parent, qrels, qids, vids, fold)
        labels = positives(ix['fit_query_ids'], ix['fit_video_ids'], qrels)
        for seed in plan['head_seeds']:
            states = fit_heads(q[ix['fit_q']], v[ix['fit_v']], labels, plan['head_training'], seed,
                               plan['head_epochs'], device=plan['head_device'],
                               progress=lambda e: atomic_json(out / 'progress.json',
                                   {'fold': fold, 'seed': seed, 'epoch': e, 'elapsed_seconds': time.monotonic() - started}))
            for epoch, state in states.items():
                save_file(state, str(out / 'results' / f'fold-{fold}-seed-{seed}-epoch-{epoch}.safetensors'))
                for weight in plan['raw_weights']:
                    score = evaluate(q[ix['dev_q']], v[ix['held_v']], ix['dev_query_ids'], ix['held_video_ids'],
                                     qrels, state, weight, plan['head_device'])
                    cells.append({'fold': fold, 'seed': seed, 'epoch': epoch, 'raw_weight': weight, **score})
    recipes, selected = select_head_recipe(cells, plan['head_folds'], plan['head_seeds'],
                                          plan['head_epochs'], plan['raw_weights'], plan['minimum_rp_gain'])
    atomic_json(out / 'results/cells.json', cells)
    finish(out, plan, {'kind': 'canonical cached head duration and scoring comparison',
           'accuracy_score': True, 'gate_passed': selected is not None, 'recipes': recipes, 'selected': selected,
           'baseline': {'epoch': 4, 'raw_weight': .25}, 'seconds': time.monotonic() - started,
           'limits': 'Reused training-side query groups and excluded-video folds; no pristine confirmation or hidden-test tuning.'})


def prepare_frames(out, plan):
    parent, _, _ = lora_pilot.inputs(plan)
    vids = sorted(v for group in parent['development_video_groups'] for v in group)
    def one(vid):
        from baseline.h100_models import FrameStore
        frames, metadata = FrameStore(Path(plan['frame_directory'])).get(vid)
        return {'video_id': vid, 'source_sha256': metadata['source_sha256'],
                'metadata_sha256': sha256_file(Path(plan['frame_directory']) / vid / 'metadata.json')}
    with ThreadPoolExecutor(max_workers=plan['decode_workers']) as pool:
        rows = []
        for row in pool.map(one, vids):
            rows.append(row)
            if len(rows) % 64 == 0:
                atomic_json(out / 'progress.json', {'completed': len(rows), 'total': len(vids)})
    atomic_json(out / 'results/frame-parents.json', rows)
    finish(out, plan, {'kind': 'verified four-frame training-development cache', 'gate_passed': True,
                       'videos': len(rows), 'accuracy_score': False, 'audit_pixels_prepared': False})


def prepared_manifest(root):
    return json.loads(verified_result(root, 'prepare-lora', 'input-manifest.json').read_text())


def encode_features(out, plan, adapted=False):
    import torch
    from safetensors.torch import load_file
    from baseline.lora import attach_language_lora, load_adapter_state
    parent, queries, _ = lora_pilot.inputs(plan)
    manifest = prepared_manifest(out.parent)
    qids = sorted(parent['query_groups']['fit'] + parent['query_groups']['dev'])
    vids = sorted(manifest['fitting_video_ids'] + manifest['held_video_ids'])
    encoder = lora_pilot.TrainingEncoder(plan)
    if adapted:
        attach_language_lora(encoder.model, plan['rank'], plan['alpha'])
        path = verified_result(out.parent, 'train-lora', 'adapter.safetensors')
        load_adapter_state(encoder.model, load_file(str(path)))
    encoder.model.eval(); qparts, vparts = [], []; started = time.monotonic()
    with torch.no_grad():
        for i in range(0, len(vids), plan['encode_video_batch']):
            vi, _ = encoder.video_inputs(vids[i:i + plan['encode_video_batch']])
            vparts.append(encoder.encode(vi).detach().cpu().numpy())
            if i % 64 == 0:
                atomic_json(out / 'progress.json', {'stage': 'video-encoding', 'completed': min(i + plan['encode_video_batch'], len(vids)), 'total': len(vids)})
        for i in range(0, len(qids), plan['encode_query_batch']):
            qi = encoder.query_inputs([queries[qid]['text'] for qid in qids[i:i + plan['encode_query_batch']]])
            qparts.append(encoder.encode(qi).detach().cpu().numpy())
    save_vectors(out / 'results/queries.npz', np.concatenate(qparts))
    save_vectors(out / 'results/videos.npz', np.concatenate(vparts))
    atomic_json(out / 'results/feature-manifest.json', {'query_ids': qids, 'video_ids': vids, 'adapted': adapted,
                'files': {name: sha256_file(out / 'results' / name) for name in ['queries.npz', 'videos.npz']},
                'recipe_sha256': sha256_file(Path(plan['native_recipe'])),
                'model_provenance_sha256': sha256_file(Path(plan['model_directory']) / 'model_provenance.json'),
                'device': torch.cuda.get_device_name(0), 'capability': list(torch.cuda.get_device_capability()),
                'fitting_input_manifest_sha256': sha256_file(out.parent / 'prepare-lora/results/input-manifest.json')})
    finish(out, plan, {'kind': 'same-device ' + ('adapted' if adapted else 'frozen') + ' accuracy features',
                       'gate_passed': True, 'accuracy_score': False, 'queries': len(qids), 'videos': len(vids),
                       'seconds': time.monotonic() - started})


def train_lora(out, plan):
    import torch
    from safetensors.torch import save_file
    from baseline.lora import attach_language_lora, adapter_state, multi_positive_loss
    _, queries, qrels = lora_pilot.inputs(plan)
    root = out.parent; manifest = prepared_manifest(root)
    profile = json.loads(verified_result(root, 'profile-lora', 'report.json').read_text())
    size = plan['training_query_batch']
    if not any(row['query_batch'] == size and row['gate_passed'] for row in profile['profiles']):
        raise ValueError('The fixed training batch did not pass the new-machine profile')
    fm = json.loads(verified_result(root, 'frozen-features', 'feature-manifest.json').read_text())
    qref = load_vectors(verified_result(root, 'frozen-features', 'queries.npz'))
    vref = load_vectors(verified_result(root, 'frozen-features', 'videos.npz'))
    qp = {qid: i for i, qid in enumerate(fm['query_ids'])}; vp = {vid: i for i, vid in enumerate(fm['video_ids'])}
    rng = np.random.default_rng(plan['seed']); torch.manual_seed(plan['seed'])
    encoder = lora_pilot.TrainingEncoder(plan)
    first = sample_batch(manifest['fitting_query_ids'], manifest['fitting_video_ids'], qrels,
                         query_batch=size, excluded=manifest['ambiguous_negative_exclusions'], rng=rng)
    qi, vi, _ = encoder.batch_inputs(first, queries)
    with torch.no_grad():
        before_q, before_v = encoder.encode(qi).detach().clone(), encoder.encode(vi).detach().clone()
    attach_language_lora(encoder.model, plan['rank'], plan['alpha'])
    with torch.no_grad():
        drift = max(float((encoder.encode(qi) - before_q).abs().max()), float((encoder.encode(vi) - before_v).abs().max()))
    if drift > 1e-6:
        raise ValueError('Zero adapter changes the base encoder')
    initial = adapter_state(encoder.model)
    frozen = {name: p._version for name, p in encoder.model.named_parameters() if not p.requires_grad}
    parameters = [p for p in encoder.model.parameters() if p.requires_grad]
    encoder.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    encoder.model.train(); encoder.model.visual.eval()
    optimizer = torch.optim.AdamW(parameters, lr=plan['learning_rate'], weight_decay=.01)
    optimizer.zero_grad(set_to_none=True); torch.cuda.reset_peak_memory_stats()
    losses, norms = [], []; started = time.monotonic()
    for step in range(plan['optimizer_steps']):
        for _ in range(plan['gradient_accumulation']):
            batch = sample_batch(manifest['fitting_query_ids'], manifest['fitting_video_ids'], qrels,
                                 query_batch=size, excluded=manifest['ambiguous_negative_exclusions'], rng=rng)
            qi, vi, _ = encoder.batch_inputs(batch, queries)
            with encoder.native.attention_context():
                q, v = encoder.encode(qi), encoder.encode(vi)
                pos = torch.tensor(batch['positive_mask'], dtype=torch.bool, device='cuda')
                neg = torch.tensor(batch['negative_mask'], dtype=torch.bool, device='cuda')
                contrast = multi_positive_loss(q @ v.T / plan['temperature'], pos, neg)
                rq = torch.as_tensor(qref[[qp[qid] for qid in batch['query_ids']]], device='cuda')
                rv = torch.as_tensor(vref[[vp[vid] for vid in batch['video_ids']]], device='cuda')
                anchor = ((q - rq).square().sum(1).mean() + (v - rv).square().sum(1).mean()) / 2
                loss = contrast + plan['anchor_weight'] * anchor
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite LoRA training loss')
                (loss / plan['gradient_accumulation']).backward(); losses.append(float(loss.detach()))
            del q, v, loss, contrast, anchor, qi, vi
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1)
        if not torch.isfinite(norm) or norm <= 0:
            raise ValueError('Missing/nonfinite LoRA gradient')
        norms.append(float(norm)); optimizer.step(); optimizer.zero_grad(set_to_none=True)
        peak = torch.cuda.max_memory_reserved()
        if peak >= plan['max_reserved_gib'] * 1024 ** 3:
            raise ValueError('LoRA pilot exceeded its memory budget')
        atomic_json(out / 'progress.json', {'optimizer_steps': step + 1, 'total': plan['optimizer_steps'],
                    'elapsed_seconds': time.monotonic() - started, 'peak_reserved_bytes': peak})
        if (step + 1) % plan['checkpoint_interval'] == 0:
            # Diagnostic checkpoint; only the fixed final step is evaluated.
            save_file(adapter_state(encoder.model), str(out / 'results' / f'adapter-step-{step + 1}.safetensors'))
    final = adapter_state(encoder.model)
    changed = any(not torch.equal(value, initial[name]) for name, value in final.items())
    untouched = all(p._version == frozen[name] and p.grad is None for name, p in encoder.model.named_parameters() if name in frozen)
    if not changed or not untouched:
        raise ValueError('Adapter/base-weight correctness gate failed')
    save_file(final, str(out / 'results/adapter.safetensors'))
    finish(out, plan, {'kind': 'fixed-budget training-only LoRA pilot', 'gate_passed': True, 'accuracy_score': False,
           'optimizer_steps': plan['optimizer_steps'], 'query_batch': size, 'losses': losses, 'gradient_norms': norms,
           'adapter_changed': changed, 'frozen_weights_unchanged': untouched, 'zero_adapter_drift': drift,
           'training_query_ids': manifest['fitting_query_ids'], 'fitting_video_ids': manifest['fitting_video_ids'],
           'seconds': time.monotonic() - started, 'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
           'checkpoint_selection': 'Only the prescribed final step; development labels never select an adapter step.'})


def evaluate_lora(out, plan):
    from safetensors.torch import save_file
    parent, queries, qrels = lora_pilot.inputs(plan)
    arms = {}; manifests = []
    for arm, job in [('frozen', 'frozen-features'), ('adapted', 'adapted-features')]:
        fm = json.loads(verified_result(out.parent, job, 'feature-manifest.json').read_text()); manifests.append(fm)
        q = load_vectors(verified_result(out.parent, job, 'queries.npz'))
        v = load_vectors(verified_result(out.parent, job, 'videos.npz'))
        ix = fold_indices(parent, qrels, fm['query_ids'], fm['video_ids'], plan['held_fold'])
        rows = []
        for seed in plan['head_seeds']:
            state = fit_heads(q[ix['fit_q']], v[ix['fit_v']], positives(ix['fit_query_ids'], ix['fit_video_ids'], qrels),
                              plan['head_training'], seed, [4], plan['head_device'])[4]
            save_file(state, str(out / 'results' / f'{arm}-head-seed-{seed}.safetensors'))
            for weight in plan['raw_weights']:
                rows.append({'seed': seed, 'raw_weight': weight, **evaluate(q[ix['dev_q']], v[ix['held_v']],
                             ix['dev_query_ids'], ix['held_video_ids'], qrels, state, weight, plan['head_device'])})
        arms[arm] = rows
    for key in ['query_ids', 'video_ids', 'recipe_sha256', 'model_provenance_sha256', 'device', 'capability', 'fitting_input_manifest_sha256']:
        if manifests[0][key] != manifests[1][key]:
            raise ValueError('Frozen/adapted inference conditions differ: ' + key)
    comparisons = []
    for a, f in zip(arms['adapted'], arms['frozen'], strict=True):
        change = paired_gain(a, f, plan['minimum_rp_gain'])
        focused = {qid for qid in parent['query_groups']['dev'] if FOCUS.search(queries[qid]['text'])}
        ar = [r['r_precision'] for r in a['per_query'] if r['query_id'] in focused and r['r_precision'] is not None]
        fr = [r['r_precision'] for r in f['per_query'] if r['query_id'] in focused and r['r_precision'] is not None]
        comparisons.append({'seed': a['seed'], 'raw_weight': a['raw_weight'], **change,
                            'heuristic_actor_order_rp_delta': float(np.mean(ar) - np.mean(fr)) if ar else None})
    primary = [c for c in comparisons if c['raw_weight'] == .25]
    gate = all(c['gate_passed'] for c in primary)
    atomic_json(out / 'results/arms.json', arms)
    finish(out, plan, {'kind': 'paired LoRA accuracy with independently fitted E4 heads', 'gate_passed': gate,
           'accuracy_score': True, 'comparisons': comparisons, 'primary_raw_weight': .25,
           'limits': 'One reused excluded-video fold, one backbone-training seed, two head seeds. Positive results justify replication, not automatic deployment.'})


def validation_inputs(plan, dimension):
    for name, digest in plan['validation_input_hashes'].items():
        if sha256_file(Path(name)) != digest:
            raise ValueError('Frozen validation input changed: ' + name)
    valq = load_vectors(Path(plan['validation_features']))
    valids = list(unique_rows(Path('data/retrieval/cascade-v1.0/val/queries.jsonl'), 'query_id'))
    if valq.shape != (len(valids), dimension):
        raise ValueError('Validation query feature alignment changed')
    return valq, valids, unique_rows(Path('data/retrieval/cascade-v1.0/val/qrels.jsonl'), 'query_id')


def confirm_head(out, plan):
    from safetensors.torch import load_file, save_file
    selected = json.loads(verified_result(out.parent, 'cached-heads', 'report.json').read_text())['selected']
    if selected is None:
        raise ValueError('No development-selected recipe')
    parent, _, qrels, qids, vids, q, v = cached_data(plan)
    labels = positives(qids, vids, qrels); valid = labels.any(1)
    states = fit_heads(q[valid], v, labels[valid], plan['head_training'], plan['seed'],
                       sorted({4, selected['epoch']}), plan['head_device'])
    # Validation labels are first opened only after train-only selection and fitting.
    valq, valids, valrels = validation_inputs(plan, q.shape[1])
    # Primary gate: validation queries against each fold's held videos, scored with the train-only
    # cached-head checkpoints that never saw those videos. Train/validation share most videos, so a
    # head fitted on the full corpus rewards memorizing familiar clips rather than generalizing.
    cells = []
    for fold in plan['head_folds']:
        ix = fold_indices(parent, qrels, qids, vids, fold)
        for seed in plan['head_seeds']:
            for epoch in plan['head_epochs']:
                state = load_file(str(verified_result(out.parent, 'cached-heads', f'fold-{fold}-seed-{seed}-epoch-{epoch}.safetensors')))
                for weight in plan['raw_weights']:
                    score = evaluate(valq, v[ix['held_v']], valids, ix['held_video_ids'], valrels, state, weight, plan['head_device'])
                    cells.append({'fold': fold, 'seed': seed, 'epoch': epoch, 'raw_weight': weight, **score})
    recipes, _ = select_head_recipe(cells, plan['head_folds'], plan['head_seeds'], plan['head_epochs'],
                                    plan['raw_weights'], plan['minimum_rp_gain'])
    held = next(r for r in recipes if r['epoch'] == selected['epoch'] and r['raw_weight'] == selected['raw_weight'])
    # Secondary, non-gating: the previous shared-video protocol with full-corpus refits.
    baseline = evaluate(valq, v, valids, vids, valrels, states[4], .25, plan['head_device'])
    candidate = evaluate(valq, v, valids, vids, valrels, states[selected['epoch']], selected['raw_weight'], plan['head_device'])
    drift = abs(baseline['r_precision'] - plan['canonical_validation_baseline_rp'])
    save_file(states[selected['epoch']], str(out / 'results/selected-head.safetensors'))
    atomic_json(out / 'results/held-video-cells.json', cells)
    atomic_json(out / 'results/scores.json', {'baseline': baseline, 'selected': candidate})
    finish(out, plan, {'kind': 'one selected cached-head recipe on released validation queries over held-out videos',
           'validation_used': True, 'accuracy_score': True, 'gate_passed': held['gate_passed'], 'selected': selected,
           'held_video_comparison': held, 'held_video_recipes_diagnostic_only': recipes,
           'shared_video_comparison': {**paired_gain(candidate, baseline, plan['minimum_rp_gain']),
                                       'gating': False, 'baseline_rp_drift': drift,
                                       'baseline_reproduced': drift <= plan['maximum_baseline_rp_drift']},
           'limits': 'Released validation has already been examined; this is not a pristine test or a hidden-test estimate. '
                     'Only the development-selected recipe is gated; other validation rows are diagnostics.'})


JOBS = {'prepare-lora': lora_pilot.prepare, 'prepare-frames': prepare_frames, 'cached-heads': cached_heads,
        'profile-lora': lora_pilot.profile, 'frozen-features': encode_features, 'train-lora': train_lora,
        'adapted-features': lambda out, plan: encode_features(out, plan, True),
        'evaluate-lora': evaluate_lora, 'confirm-head': confirm_head}


def stage_directory(out):
    if out.is_symlink() or out.exists() and (not out.is_dir() or any(p.name != 'worker.log' or not p.is_file() or p.is_symlink() for p in out.iterdir())):
        raise ValueError('Use a fresh stage directory; never overwrite an experiment')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'results').mkdir()


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
    # The queue pre-creates only the directory and worker log.
    stage_directory(args.out)
    JOBS[args.job](args.out, plan)


if __name__ == '__main__':
    main()
