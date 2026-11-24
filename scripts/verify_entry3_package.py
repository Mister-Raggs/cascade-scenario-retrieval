"""Offline reconstruction check of the entry-3 package: unfamiliar IDs, no network, read-only inputs, poisoned labels,
Qwen3.8 capped at 80 GB of GPU memory (A100 budget), feature and answer parity with research values, and top-10 parity
with a research-feature reference.

Clips are 32 audit clips, because only they have research Qwen3.8 answers from the packaged adapter; no audit label is
read and no metric is computed. Queries are 64 released validation texts.

Two references (2026-10-21; the entry-2 gate of 60/64 exact top-10 assumed a deterministic pipeline, and bf16 Qwen3.8
answers at question batch 16 differ slightly from the research run at batch 96):
  pipeline  research features + the container's own Qwen3.8 answers: >= 63/64 exact top-10 (everything but the answers is exact)
  answers   research features + research Qwen3.8 answers: mean |log-odds| difference <= 0.1 and mean top-10 overlap >= 0.98;
            its exact top-10 count is reported
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
from baseline.dense import normalize, rank_scores
from baseline.entry3_recipe import fuse, structured_scores, t4_probabilities, with_vlm
from baseline.io import sha256_file, unique_rows, write_json, write_jsonl
from baseline.native_ranking import blended_scores
from baseline.native_run import load_projection
from scripts.feature_retrieval import load_vectors
from scripts.post_submission import verified_result
from scripts.structured_eval import provider
from scripts.verify_native_container import BASE, execute, load

PLAN = ROOT / 'configs/audit-q38/evaluator.json'
VALIDATION = ROOT / 'runs/finalist-canonical-2026-10-11-v2/canonical-refit/results/embed-8b/queries-val.npz'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--context', type=Path, required=True); ap.add_argument('--artifact', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True); ap.add_argument('--gpu-memory-gib', type=float, default=80)
    a = ap.parse_args(); os.chdir(ROOT); a.out.mkdir(parents=True, exist_ok=False)
    plan = json.loads(PLAN.read_text()); top_k = 10
    input_dir, output_dir = a.out / 'input', a.out / 'output'; input_dir.mkdir(); output_dir.mkdir()
    research_vlm = np.load(verified_result(ROOT / 'runs/audit-v2', 'audit-vlm-score', 'vlm-atom-probs.npz'))
    staged = json.loads((a.context / 'native_recipe.json').read_text())
    if staged['vlm'].get('frames') == 32:  # entry 5: the research reference answers are the audit-mode adapter's 32-frame answers
        research_vlm = np.load(ROOT / 'runs/verifier-variants/atoms-f32-faudit/vlm-atom-probs.npz')
        research_props = np.load(ROOT / 'runs/verifier-variants/props32-faudit/props.npz')
    audit_ids = list(research_vlm['video_ids']); ids = audit_ids[::len(audit_ids) // 32][:32]
    val = unique_rows(ROOT / 'data/retrieval/cascade-v1.0/val/queries.jsonl', 'query_id'); val_ids = list(val)
    chosen = [val_ids[i] for i in np.random.default_rng(16).choice(len(val_ids), 64, replace=False)]
    aliases = {v: f'unfamiliar-video-{i:03d}' for i, v in enumerate(ids)}
    for i, v in enumerate(ids):
        os.link(ROOT / 'data/videos' / f'{v}.mp4', input_dir / f'clip-{i:03d}.mp4')
    write_jsonl(input_dir / 'videos.jsonl', ({'video_id': aliases[v], 'path': f'clip-{i:03d}.mp4'} for i, v in enumerate(ids)))
    write_jsonl(input_dir / 'queries.jsonl', ({'query_id': f'unfamiliar-query-{i:03d}', 'text': val[q]['text']} for i, q in enumerate(chosen)))
    (input_dir / 'qrels.jsonl').write_text('Poison labels: inference must never read this file.')
    before = {p.name: sha256_file(p) for p in input_dir.iterdir()}
    recon = load('_entry3_reconstruct', ROOT / 'vendor/cascade-devkit/docker-prep-kit/reconstruct_submission.py')
    manifest = recon.load_manifest(a.artifact); recon.verify_base(BASE, manifest)
    command = recon.docker_run_command(recon.RunSpec(artifact_dir=a.artifact, base_image=BASE, input_dir=input_dir, output_dir=output_dir,
                                                     network='none', gpus='all'))
    command[2:2] = ['--env', f'TOP_K={top_k}', '--env', 'ENTRY3_DEBUG_FEATURES=1', '--env', f'ENTRY3_GPU_MEMORY_GIB={a.gpu_memory_gib:g}']
    started = time.monotonic(); code = execute(command, a.out / 'reconstruction.log', 7200); seconds = time.monotonic() - started
    after = {p.name: sha256_file(p) for p in input_dir.iterdir()}
    validator = load('_entry3_validator', ROOT / 'vendor/cascade-devkit/docker-prep-kit/validate_submission.py')
    errors, warnings = validator.validate_predictions(output_dir / 'predictions.jsonl', input_dir / 'queries.jsonl', input_dir / 'videos.jsonl', top_k) if code == 0 else (['Reconstruction failed'], [])
    if before != after:
        errors.append('Read-only inputs changed')
    parity = {}; run = None
    if code == 0:
        run = json.loads((output_dir / 'run_manifest.json').read_text())
        debug = np.load(output_dir / 'debug_features.npz'); order = [ids[list(aliases.values()).index(x)] for x in debug['video_ids']]
        research = {b: provider(b, plan['blocks'][b], order)[0] for b in ('pooled', 'motion2', 'windows')}
        for b in research:
            x, y = debug[b].astype(np.float64), research[b].astype(np.float64)
            cos = (x * y).sum(1) / (np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1) + 1e-12)
            parity[b] = {'max_abs_diff': float(np.abs(x - y).max()), 'min_cosine': float(cos.min())}
        if parity['motion2']['max_abs_diff'] > 1e-4:
            errors.append('Motion features differ from research features')
        for b in ('pooled', 'windows'):
            if parity[b]['min_cosine'] < 0.999:
                errors.append(f'{b} features drift from research features')
        rows = [audit_ids.index(v) for v in order]
        if list(debug['vlm_atoms']) != list(research_vlm['atoms']):
            errors.append('Qwen3.8 atom order differs from research')
        logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
        d = np.abs(logit(debug['vlm_probs'].astype(np.float64)) - logit(research_vlm['probs'][rows].astype(np.float64)))
        parity['vlm'] = {'max_abs_logit_diff': float(d.max()), 'mean_abs_logit_diff': float(d.mean()),
                         'max_abs_probability_diff': float(np.abs(debug['vlm_probs'] - research_vlm['probs'][rows]).max()),
                         'peak_reserved_gib': run['vlm']['peak_reserved_gib'], 'prefix_logit_drift': run['vlm']['prefix_logit_drift']}
        if run['vlm']['peak_reserved_gib'] > a.gpu_memory_gib:
            errors.append('Qwen3.8 scoring exceeded the GPU memory cap')
        # Reference ranking: research features and research Qwen3.8 answers through the packaged components and head.
        recipe = json.loads((a.context / 'native_recipe.json').read_text())
        project = load_projection(a.context / 'checkpoint', recipe, 'cpu')
        m2 = np.load(ROOT / plan['blocks']['motion2']['path']); series = [m2[v] for v in order]
        vlm = {str(x): np.clip(research_vlm['probs'][rows, j], 1e-4, 1 - 1e-4) for j, x in enumerate(research_vlm['atoms'])}
        own = {str(x): np.clip(debug['vlm_probs'][:, j], 1e-4, 1 - 1e-4) for j, x in enumerate(debug['vlm_atoms'])}
        t4 = t4_probabilities(a.context / recipe['entry3']['atoms'], research, series)
        valq = load_vectors(VALIDATION)[[val_ids.index(q) for q in chosen]]
        q = normalize(valq); raw_docs = research['pooled']; raw = q @ raw_docs.T
        submitted = blended_scores(raw, project(q) @ project(raw_docs).T, recipe['raw_rank_weight'])
        names = [aliases[v] for v in order]; predictions = unique_rows(output_dir / 'predictions.jsonl', 'query_id')
        texts = [val[x]['text'] for x in chosen]

        def compare(answers):
            if 'entry4' in recipe:  # the linear learned scorer ranks (bundle: parser v3 and the container's own verifier answers)
                from baseline.entry4_scorer import LinearScorer, features as scorer_features
                extra = {}
                if recipe['entry4'].get('parser') == 'v3':
                    from baseline.query_parser import Unparsed
                    from baseline.query_parser_v3 import translate as v3
                    from baseline.query_struct import atoms as atoms_of, tree

                    def parse(text):
                        try:
                            return sorted(set(atoms_of(tree(v3(text)))))
                        except Unparsed:
                            return None
                    extra['parse'] = parse
                if recipe['entry4'].get('verifier'):
                    extra['verify'] = debug['verify']
                if recipe['entry4'].get('properties'):  # entry 5: property answers (the container's own, or research for reference)
                    from baseline.property_atoms import query_atoms as prop_atoms
                    from baseline.query_parser import Unparsed
                    from baseline.query_parser_v3 import translate as v3p

                    def prop_parse(text):
                        try:
                            return prop_atoms(v3p(text))
                        except Unparsed:
                            return []
                    extra['props'] = prop_answers; extra['prop_parse'] = prop_parse
                x, mask = scorer_features(texts, t4, answers, raw, submitted, **extra)
                expected = LinearScorer(a.context / recipe['entry4']['weights']).scores(x)
            else:
                scores, mask = structured_scores(texts, with_vlm(t4, answers))
                expected = fuse(raw, submitted, scores, mask, recipe['entry3']['fusion_weight'])
            overlaps = []
            for i in range(len(chosen)):
                got = predictions[f'unfamiliar-query-{i:03d}']['video_ids']; want = rank_scores(names, expected[i])[:top_k]
                overlaps.append((got == want, len(set(got) & set(want)) / top_k))
            return {'top10_exact': int(sum(x for x, _ in overlaps)), 'mean_top10_overlap': float(np.mean([o for _, o in overlaps])),
                    'parsed_queries': int(mask.sum())}
        prop_answers = None
        if recipe.get('entry4', {}).get('properties'):
            prop_answers = {str(k): debug['prop_probs'][:, j] for j, k in enumerate(recipe['vlm']['properties'])}
        parity['pipeline_reference'] = compare(own)
        if recipe.get('entry4', {}).get('properties'):
            if [str(x) for x in research_props['atoms']] != list(recipe['vlm']['properties']):
                errors.append('Property order differs from research')
            prop_answers = {str(k): research_props['probs'][rows, j] for j, k in enumerate(research_props['atoms'])}
        parity['answers_reference'] = compare(vlm)
        if parity['pipeline_reference']['top10_exact'] < 63:
            errors.append(f"Top-{top_k} parity with the container's own answers: {parity['pipeline_reference']['top10_exact']}/64 exact")
        if parity['vlm']['mean_abs_logit_diff'] > 0.1 or parity['answers_reference']['mean_top10_overlap'] < 0.98:
            errors.append('Qwen3.8 answers drift from the research answers')
        per_video = run['vlm']['seconds'] / len(ids)
        parity['projection'] = {'vlm_seconds_per_video_here': per_video, 'native_and_fusion_seconds': run['total_seconds'] - run['vlm_seconds'],
                                'note': 'scale by test size (about 650 videos, 2,000 queries) and A100 speed; see the handoff for the estimate'}
    report = {'kind': 'offline entry-3 package reconstruction', 'passed': code == 0 and not errors, 'exit_code': code, 'network': 'none',
              'input_mounted_read_only': True, 'input_hashes_unchanged': before == after, 'videos': len(ids), 'queries': len(chosen),
              'top_k': top_k, 'gpu_memory_cap_gib': a.gpu_memory_gib, 'seconds_including_reconstruction': seconds, 'parity': parity,
              'validator_errors': errors, 'validator_warnings': warnings, 'run_manifest': run,
              'artifact_manifest_sha256': sha256_file(a.artifact / 'manifest.json'), 'is_validation_score': False, 'audit_labels_read': False,
              'limits': '32 audit clips and 64 validation query texts under unfamiliar IDs; checks offline execution, memory and parity, '
                        'not quality, actual A100 execution or the full six-hour budget.'}
    write_json(a.out / 'report.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'run_manifest'}, indent=1), flush=True)
    if not report['passed']:
        raise SystemExit('Entry-3 package verification failed')


if __name__ == '__main__':
    main()
