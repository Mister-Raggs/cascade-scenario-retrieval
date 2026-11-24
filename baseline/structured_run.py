"""Offline structured retrieval entrypoint: submitted native recipe plus parser-driven atom probes (no labels)."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import time

import numpy as np

from .dense import normalize, rank_scores
from .environment import environment
from .io import sha256_file, unique_rows, write_json, write_jsonl
from .native_ranking import blended_scores
from .native_run import load_projection
from .native_video import NativeEncoder, validate_recipe
from .structured_recipe import Probes, WINDOWS, fuse, motion_features, structured_scores, window_feature, window_items


def encode_videos(encoder, paths, size):
    vectors, rows = [], []
    with ThreadPoolExecutor(max_workers=4) as pool:
        pending = [pool.submit(encoder.decode, p) for p in paths[:size]]
        for start in range(0, len(paths), size):
            decoded = [f.result() for f in pending]
            pending = [pool.submit(encoder.decode, p) for p in paths[start + size:start + 2 * size]]
            v, meta = encoder.videos(decoded); vectors.extend(normalize(v)); rows.extend(meta)
    return np.stack(vectors), rows


def encode_windows(encoder, paths, size, progress):
    """Windows of all videos flattened in video order and batched like the research extraction."""
    with ThreadPoolExecutor(max_workers=4) as pool:
        items = [it for its in pool.map(window_items, paths) for it in its]
    vectors = []
    for start in range(0, len(items), size):
        v, _ = encoder.videos(items[start:start + size]); vectors.extend(normalize(v))
        progress(min(start + size, len(items)), len(items))
    vectors = np.stack(vectors).reshape(len(paths), WINDOWS['count'], -1)
    return np.stack([window_feature(v) for v in vectors])


def run(input_dir, output_dir, *, model_path, helper_path, checkpoint, recipe_path, probes_path, top_k=100, device='auto'):
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0:
        raise ValueError('TOP_K must be a nonnegative integer')
    if output_dir.resolve().is_relative_to(input_dir.resolve()) or input_dir.resolve().is_relative_to(output_dir.resolve()):
        raise ValueError('Input and output directories must be separate')
    recipe = json.loads(recipe_path.read_text()); validate_recipe(recipe)
    structured = recipe['structured']
    if sha256_file(helper_path) != recipe['helper_sha256'] or sha256_file(probes_path) != structured['probes_sha256']:
        raise ValueError('Native helper or atom probes changed')
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
    output_dir.mkdir(parents=True, exist_ok=True); began = time.perf_counter(); query_ids = list(queries)
    if not (ids and top_k):
        write_jsonl(output_dir / 'predictions.jsonl', ({'query_id': q, 'video_ids': []} for q in query_ids))
        return {'num_queries': len(query_ids), 'total_seconds': 0.0}
    encoder = NativeEncoder(model_path, helper_path, recipe, device)
    project = load_projection(checkpoint, recipe, encoder.device); size = recipe.get('video_batch_size', 4)
    progress = lambda stage, done, total: write_json(output_dir / 'progress.json', {'stage': stage, 'completed': done, 'total': total})
    with ThreadPoolExecutor(max_workers=structured['motion_workers']) as pool:
        motion_future = pool.map(motion_features, paths)
        raw_docs, rows = encode_videos(encoder, paths, size); progress('pooled', len(ids), len(ids))
        docs = project(raw_docs)
        windows = encode_windows(encoder, paths, size, lambda d, t: progress('windows', d, t))
        motion = np.stack(list(motion_future)); progress('motion', len(ids), len(ids))
    if os.environ.get('STRUCTURED_DEBUG_FEATURES') == '1':
        # Verification only: lets offline checks compare container features with research features.
        np.savez_compressed(output_dir / 'debug_features.npz', video_ids=np.asarray(ids), pooled=raw_docs, motion2=motion, windows=windows)
    probes = Probes(probes_path)
    probabilities = probes.probabilities({'pooled': raw_docs, 'motion2': motion, 'windows': windows})
    encoded_seconds = time.perf_counter() - began; parsed = 0

    def predictions():
        nonlocal parsed
        for start in range(0, len(query_ids), 16):
            selected = query_ids[start:start + 16]; texts = [queries[q]['text'] for q in selected]
            q = normalize(encoder.queries(texts))
            raw = q @ raw_docs.T
            submitted = blended_scores(raw, project(q) @ docs.T, recipe.get('raw_rank_weight', 0))
            scores, mask = structured_scores(texts, probabilities, probes); parsed += int(mask.sum())
            for qid, score in zip(selected, fuse(raw, submitted, scores, mask, structured['fusion_weight']), strict=True):
                yield {'query_id': qid, 'video_ids': rank_scores(ids, score)[:top_k]}
    write_jsonl(output_dir / 'predictions.jsonl', predictions())
    write_jsonl(output_dir / 'video_processing.jsonl', [{'video_id': v, **r} for v, r in zip(ids, rows, strict=True)])
    report = {'method': 'native-video-shared-projection-plus-structured-atoms', 'recipe_sha256': sha256_file(recipe_path),
              'is_validation_score': False, 'num_videos': len(ids), 'num_queries': len(query_ids), 'parsed_queries': parsed,
              'top_k': top_k, 'feature_seconds': encoded_seconds, 'total_seconds': time.perf_counter() - began,
              'environment': environment(encoder.device, load_torch=True), 'predictions_sha256': sha256_file(output_dir / 'predictions.jsonl'),
              'feature_reuse': 'none; encode current input pixels and query text'}
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
    p.add_argument('--model-path', type=Path, default=root / 'model')
    p.add_argument('--helper-path', type=Path, default=root / 'vendor/qwen3-vl-embedding/qwen3_vl_embedding.py')
    p.add_argument('--checkpoint', type=Path, default=root / 'checkpoint')
    p.add_argument('--recipe', type=Path, default=root / 'native_recipe.json')
    p.add_argument('--probes', type=Path, default=root / 'probes/atom_probes.npz')
    p.add_argument('--top-k', type=int, default=int(os.environ.get('TOP_K', '100')))
    p.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default=os.environ.get('DEVICE', 'auto'))
    a = p.parse_args()
    report = run(a.input, a.output, model_path=a.model_path, helper_path=a.helper_path, checkpoint=a.checkpoint,
                 recipe_path=a.recipe, probes_path=a.probes, top_k=a.top_k, device=a.device)
    print(f"Wrote {report['num_queries']} structured rankings in {report['total_seconds']:.2f}s", flush=True)


if __name__ == '__main__':
    main()
