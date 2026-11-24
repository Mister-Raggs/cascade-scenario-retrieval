"""Direction 3: per-window 8B embeddings and time-localized attribute probes (GPU encode, CPU evaluate)."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from baseline.annotation_events import load_events, window_labels
from baseline.io import sha256_file
from baseline.probe import auc, cross_validated, fit_logistic, standardize
from baseline.windows import WindowFrameStore
from scripts.egomotion_run import development_videos
from scripts.feature_retrieval import atomic_json, load_vectors, save_vectors
from scripts.post_submission import stage_directory, verified_result
from scripts.query_side import finish


def store(plan):
    return WindowFrameStore(ROOT / plan['window_directory'], plan['windows'], plan['frames_per_window'], ROOT / 'data/videos')


def _prepare(args):
    directory, windows, frames, vid = args
    return vid, WindowFrameStore(directory, windows, frames, ROOT / 'data/videos').prepare(vid)


def prepare_windows(out, plan):
    vids = development_videos(plan); started = time.monotonic(); rows = {}
    tasks = [(ROOT / plan['window_directory'], plan['windows'], plan['frames_per_window'], v) for v in vids]
    with ProcessPoolExecutor(max_workers=plan['decode_workers']) as pool:
        for i, (vid, meta) in enumerate(pool.map(_prepare, tasks, chunksize=2), 1):
            rows[vid] = [{'window': m['window'], 'bounds': m['window_bounds'], 'timestamps': m['timestamps']} for m in meta]
            if i % 32 == 0 or i == len(vids):
                atomic_json(out / 'progress.json', {'completed': i, 'total': len(vids), 'elapsed_seconds': time.monotonic() - started})
    atomic_json(out / 'results/windows.json', rows)
    finish(out, plan, {'kind': 'source-timed per-window frame cache', 'gate_passed': True, 'accuracy_score': False,
           'videos': len(rows), 'windows': plan['windows'], 'seconds': time.monotonic() - started})


def encode_windows(out, plan):
    import torch
    from scripts import lora_pilot
    vids = development_videos(plan); windows = store(plan)
    keys = [k for v in vids for k in windows.keys(v)]
    encoder = lora_pilot.TrainingEncoder(plan); encoder.frames = windows; encoder.model.eval()
    parts = []; started = time.monotonic()
    with torch.no_grad():
        for i in range(0, len(keys), plan['encode_video_batch']):
            batch, _ = encoder.video_inputs(keys[i:i + plan['encode_video_batch']])
            parts.append(encoder.encode(batch).cpu().numpy())
            if i % 128 == 0:
                atomic_json(out / 'progress.json', {'completed': min(i + plan['encode_video_batch'], len(keys)), 'total': len(keys),
                            'elapsed_seconds': time.monotonic() - started})
    save_vectors(out / 'results/window-vectors.npz', np.concatenate(parts))
    atomic_json(out / 'results/window-manifest.json', {'keys': keys, 'windows': plan['windows'],
                'files': {'window-vectors.npz': sha256_file(out / 'results/window-vectors.npz')},
                'device': torch.cuda.get_device_name(0)})
    finish(out, plan, {'kind': 'per-window frozen 8B video vectors', 'gate_passed': True, 'accuracy_score': False,
           'windows_encoded': len(keys), 'seconds': time.monotonic() - started,
           'peak_reserved_bytes': torch.cuda.max_memory_reserved()})


def label_set(events, vids, minimum):
    counts = {}
    for v in vids:
        for label in {e[0] for e in events[v]['events']}:
            counts[label] = counts.get(label, 0) + 1
    return sorted(label for label, n in counts.items() if n >= minimum)


def evaluate_windows(out, plan):
    vids = development_videos(plan); k = plan['windows']
    manifest = json.loads(verified_result(out.parent, 'encode-windows', 'window-manifest.json').read_text())
    if manifest['keys'] != [f'{v}@{w}' for v in vids for w in range(k)]:
        raise ValueError('Window vector order changed')
    windowed = load_vectors(verified_result(out.parent, 'encode-windows', 'window-vectors.npz')).reshape(len(vids), k, -1)
    bounds = json.loads(verified_result(out.parent, 'prepare-windows', 'windows.json').read_text())
    fm = json.loads((ROOT / plan['feature_directory'] / 'feature_manifest.json').read_text())
    index = {v: i for i, v in enumerate(fm['video_ids'])}
    pooled = load_vectors(ROOT / plan['feature_directory'] / 'aggregate.npz')[[index[v] for v in vids]]
    events = load_events(ROOT / plan['annotation_directory'])
    labels = label_set(events, vids, plan['minimum_positive_clips'])
    window_y = np.asarray([window_labels(events[v]['events'], [tuple(b['bounds']) for b in bounds[v]], labels) for v in vids])
    clip_y = window_y.any(1)
    folds = np.random.default_rng(plan['seed']).permutation(len(vids)) % plan['folds']
    flat = windowed.reshape(len(vids) * k, -1); flat_folds = np.repeat(folds, k)
    rows = []; started = time.monotonic()
    for j, label in enumerate(labels):
        yw = window_y[:, :, j].reshape(-1)
        window_prob = np.zeros(len(yw))
        for f in range(plan['folds']):
            train, test = flat_folds != f, flat_folds == f
            if yw[train].sum() < 3:
                window_prob[test] = yw[train].mean(); continue
            x = standardize(flat, train); w, b = fit_logistic(x[train], yw[train])
            window_prob[test] = 1 / (1 + np.exp(-(x[test] @ w + b)))
        clip_windowed = window_prob.reshape(len(vids), k).max(1)
        rows.append({'label': label, 'positive_clips': int(clip_y[:, j].sum()),
                     'clip_auc_windowed': auc(clip_windowed, clip_y[:, j]),
                     'clip_auc_pooled': auc(cross_validated(pooled, clip_y[:, j], folds), clip_y[:, j]),
                     'window_auc': auc(window_prob, yw)})
        atomic_json(out / 'progress.json', {'labels_done': j + 1, 'total': len(labels), 'elapsed_seconds': time.monotonic() - started})
    groups = {}
    for r in rows:
        groups.setdefault(r['label'].split(':')[0], []).append(r)
    by_group = {g: {'labels': len(rs), 'windowed': float(np.mean([r['clip_auc_windowed'] for r in rs])),
                    'pooled': float(np.mean([r['clip_auc_pooled'] for r in rs]))} for g, rs in groups.items()}
    gain = float(np.mean([r['clip_auc_windowed'] - r['clip_auc_pooled'] for r in rows]))
    finish(out, plan, {'kind': 'windowed versus pooled attribute probes', 'accuracy_score': True,
           'gate_passed': gain >= plan['minimum_auc_gain'], 'mean_clip_auc_gain': gain, 'by_group': by_group,
           'labels': rows, 'seconds': time.monotonic() - started,
           'limits': 'Random video folds over development videos; frozen encoder, linear probes. Window AUC also measures '
                     'temporal localization needed for while/before scoring.'})


JOBS = {'prepare-windows': prepare_windows, 'encode-windows': encode_windows, 'evaluate-windows': evaluate_windows}


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
