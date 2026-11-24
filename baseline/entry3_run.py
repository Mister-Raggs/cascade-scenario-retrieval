"""Offline entry-3/4 entrypoint: submitted native recipe plus t4 atoms late-fused with fine-tuned Qwen3.8-27B atom answers.

Stages: (B) Qwen3.8 atom answers in the Qwen3.8 environment (subprocess, whole GPU) while motion series are decoded on
the CPU; (A) the native 8B pooled, window and query vectors in this process; (C) t4 probabilities, Qwen3.8 late fusion,
parser-v2 atom scores and rank fusion at weight 0.8 with the submitted raw/head ranking; with an 'entry4' recipe
section, the audit-confirmed linear learned scorer (baseline/entry4_scorer.py) ranks instead. No labels are read.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

import numpy as np

from .dense import normalize, rank_scores
from .entry3_recipe import BLOCKS, fuse, motion_series, structured_scores, t4_probabilities, with_vlm
from .environment import environment
from .io import sha256_file, unique_rows, write_json, write_jsonl
from .native_ranking import blended_scores
from .native_run import load_projection
from .native_video import NativeEncoder, validate_recipe
from .structured_run import encode_videos, encode_windows


def input_paths(input_dir, output_dir):
    if output_dir.resolve().is_relative_to(input_dir.resolve()) or input_dir.resolve().is_relative_to(output_dir.resolve()):
        raise ValueError('Input and output directories must be separate')
    videos = unique_rows(input_dir / 'videos.jsonl', 'video_id'); queries = unique_rows(input_dir / 'queries.jsonl', 'query_id')
    if not queries or any(not isinstance(r.get('text'), str) for r in queries.values()):
        raise ValueError('Query set is empty or malformed')
    ids = sorted(videos); paths = []
    for vid in ids:
        relative = videos[vid].get('path')
        if not isinstance(relative, str) or Path(relative).is_absolute() or '..' in Path(relative).parts:
            raise ValueError('Expected a relative video path beneath INPUT_DIR')
        if not (input_dir / relative).is_file():
            raise FileNotFoundError(input_dir / relative)
        paths.append(input_dir / relative)
    return ids, paths, queries


def run(input_dir, output_dir, *, root, top_k=100, device='auto'):
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0:
        raise ValueError('TOP_K must be a nonnegative integer')
    recipe_path = root / 'native_recipe.json'; recipe = json.loads(recipe_path.read_text()); validate_recipe(recipe)
    entry3 = recipe['entry3']; atoms_path = root / entry3['atoms']
    if sha256_file(root / 'vendor/qwen3-vl-embedding/qwen3_vl_embedding.py') != recipe['helper_sha256'] or sha256_file(atoms_path) != entry3['atoms_sha256']:
        raise ValueError('Native helper or entry-3 atom components changed')
    ids, paths, queries = input_paths(input_dir, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True); began = time.perf_counter(); query_ids = list(queries)
    if not (ids and top_k):
        write_jsonl(output_dir / 'predictions.jsonl', ({'query_id': q, 'video_ids': []} for q in query_ids))
        return {'num_queries': len(query_ids), 'total_seconds': 0.0}
    progress = lambda stage, done, total: write_json(output_dir / 'progress.json', {'stage': stage, 'completed': done, 'total': total})
    work = Path(tempfile.mkdtemp(prefix='entry3-')); (work / 'videos.json').write_text(json.dumps([str(p) for p in paths]))
    env = {**os.environ, 'TRITON_CACHE_DIR': str(work / 'triton'), 'PYTHONDONTWRITEBYTECODE': '1',
           'ENTRY5_START': os.environ.get('ENTRY5_START', str(time.time() - (time.perf_counter() - began)))}  # time-guard origin
    entry4 = recipe.get('entry4') or {}; texts = [queries[q]['text'] for q in query_ids]
    # Stage A (this process): native 8B vectors for videos, windows and queries, motion series, t4 probabilities.
    with ThreadPoolExecutor(max_workers=entry3['motion_workers']) as pool:
        motion_future = pool.map(motion_series, paths)
        encoder = NativeEncoder(root / 'model', root / 'vendor/qwen3-vl-embedding/qwen3_vl_embedding.py', recipe, device)
        project = load_projection(root / 'checkpoint', recipe, encoder.device); size = recipe.get('video_batch_size', 4)
        raw_docs, rows = encode_videos(encoder, paths, size); progress('pooled', len(ids), len(ids))
        docs = project(raw_docs)
        windows = encode_windows(encoder, paths, size, lambda d, t: progress('windows', d, t))
        qvec = np.concatenate([normalize(encoder.queries(texts[i:i + 16])) for i in range(0, len(texts), 16)])
        raw = qvec @ raw_docs.T; submitted = blended_scores(raw, project(qvec) @ docs.T, recipe.get('raw_rank_weight', 0))
        series, motion = zip(*motion_future); motion = np.stack(motion); progress('motion', len(ids), len(ids))
    features = {'pooled': raw_docs, 'motion2': motion, 'windows': windows}
    t4 = t4_probabilities(atoms_path, features, list(series)); environment_info = environment(encoder.device, load_torch=True)
    np.savez(work / 'native.npz', raw=raw, submitted=submitted, texts=np.asarray(texts), t4_atoms=np.asarray(list(t4)),
             t4_probs=np.stack(list(t4.values()), 1), fusion_weight=np.asarray(entry3['fusion_weight']))
    del encoder, project
    import gc
    import torch
    gc.collect(); torch.cuda.empty_cache()  # the whole GPU goes to Qwen3.8 in stage B
    native_seconds = time.perf_counter() - began
    # Stage B (Qwen3.8 environment): atom answers, then (entry 4) the whole-query verifier on each query's top-k clips.
    subprocess.run([str(root / entry3['vlm_python']), '-m', 'baseline.entry3_vlm', '--recipe', str(recipe_path), '--videos', str(work / 'videos.json'),
                    '--out', str(work / 'vlm.npz'), '--progress', str(output_dir / 'progress.json'), '--native', str(work / 'native.npz')],
                   cwd=root, env=env, check=True)
    vlm_seconds = time.perf_counter() - began - native_seconds
    v = np.load(work / 'vlm.npz'); vlm = {str(a): np.clip(v['probs'][:, j], 1e-4, 1 - 1e-4) for j, a in enumerate(v['atoms'])}
    vlm_report = json.loads((work / 'vlm.json').read_text())
    props = {str(a): v['prop_probs'][:, j] for j, a in enumerate(v['prop_atoms'])} if 'prop_atoms' in v.files and len(v['prop_atoms']) else None
    verify = None
    if entry4.get('verifier'):
        z = np.load(work / 'verify.npz'); verify = np.full(raw.shape, np.nan, np.float32)
        verify[z['query_index'], z['clip_index']] = z['logits']
        if vlm_report.get('guard_verifier_stopped_at_clip') is None and np.isfinite(verify).sum(1).min() < min(entry4['verifier']['k'], len(ids)):
            raise ValueError('A query lacks its verifier candidates')
    if os.environ.get('ENTRY3_DEBUG_FEATURES') == '1':
        # Verification only: lets offline checks compare container features and answers with research values.
        np.savez_compressed(output_dir / 'debug_features.npz', video_ids=np.asarray(ids), vlm_atoms=v['atoms'], vlm_probs=v['probs'],
                            verify=verify if verify is not None else np.zeros(0), prop_probs=v['prop_probs'] if props else np.zeros(0),
                            **{k: np.asarray(x) for k, x in features.items()})
    # Stage C: ranking.
    if entry4:
        from .entry4_scorer import LinearScorer, features as scorer_features
        if sha256_file(root / entry4['weights']) != entry4['weights_sha256']:
            raise ValueError('Entry-4 scorer weights changed')
        parse = None
        if entry4.get('parser') == 'v3':
            from .query_parser import Unparsed
            from .query_parser_v3 import translate as v3
            from .query_struct import atoms as atoms_of, tree

            def parse(text):
                try:
                    return sorted(set(atoms_of(tree(v3(text)))))
                except Unparsed:
                    return None
        extra = {}
        if entry4.get('properties'):
            from .property_atoms import query_atoms as property_atoms_of
            from .query_parser import Unparsed
            from .query_parser_v3 import translate as v3

            def prop_parse(text):
                try:
                    return property_atoms_of(v3(text))
                except Unparsed:
                    return []
            if props is None:
                raise ValueError('The recipe expects property answers')
            extra = {'props': props, 'prop_parse': prop_parse}
        x, mask = scorer_features(texts, t4, vlm, raw, submitted, **({'parse': parse} if parse else {}), verify=verify, **extra)
        final = LinearScorer(root / entry4['weights']).scores(x)
    else:
        scores, mask = structured_scores(texts, with_vlm(t4, vlm)); final = fuse(raw, submitted, scores, mask, entry3['fusion_weight'])
    write_jsonl(output_dir / 'predictions.jsonl', ({'query_id': q, 'video_ids': rank_scores(ids, score)[:top_k]} for q, score in zip(query_ids, final, strict=True)))
    write_jsonl(output_dir / 'video_processing.jsonl', [{'video_id': v, **r} for v, r in zip(ids, rows, strict=True)])
    report = {'method': 'native-video-shared-projection-plus-t4-atoms-and-qwen3.8-answers' + ('-linear-scorer' if entry4 else '') + ('-verifier' if verify is not None else ''),
              'recipe_sha256': sha256_file(recipe_path), 'is_validation_score': False, 'num_videos': len(ids), 'num_queries': len(query_ids),
              'parsed_queries': int(mask.sum()), 'top_k': top_k, 'native_seconds': native_seconds, 'vlm_seconds': vlm_seconds,
              'total_seconds': time.perf_counter() - began, 'vlm': vlm_report, 'environment': environment_info,
              'predictions_sha256': sha256_file(output_dir / 'predictions.jsonl'), 'feature_reuse': 'none; encode current input pixels and query text'}
    write_json(output_dir / 'run_manifest.json', report); progress('complete', 1, 1)
    return report


def main():
    os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    import torch
    torch.set_num_threads(2); torch.manual_seed(0)
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, default=Path(os.environ.get('INPUT_DIR', '/input')))
    p.add_argument('--output', type=Path, default=Path(os.environ.get('OUTPUT_DIR', '/output')))
    p.add_argument('--top-k', type=int, default=int(os.environ.get('TOP_K', '100')))
    p.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default=os.environ.get('DEVICE', 'auto'))
    a = p.parse_args()
    report = run(a.input, a.output, root=root, top_k=a.top_k, device=a.device)
    print(f"Wrote {report['num_queries']} entry-3 rankings in {report['total_seconds']:.2f}s", flush=True)


if __name__ == '__main__':
    main()
