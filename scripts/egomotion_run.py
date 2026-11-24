"""Direction 2: ego-motion features from video and a pre-registered ego-action AUC gate (CPU droplet)."""
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
from baseline.annotation_events import load_events
from baseline.egomotion import decode_gray, grid_flow, signals, clip_vector
from baseline.io import sha256_file
from baseline.probe import auc, cross_validated
from scripts.feature_retrieval import atomic_json, load_vectors
from scripts.post_submission import stage_directory, verified_result
from scripts.query_side import finish


def development_videos(plan):
    parent = json.loads(Path(plan['partition_parent']).read_text())
    if sha256_file(Path(plan['partition_parent'])) != plan['partition_parent_sha256']:
        raise ValueError('Partition changed')
    groups = 'audit_video_groups' if plan.get('video_set') == 'audit' else 'development_video_groups'
    return sorted(v for group in parent[groups] for v in group)


def one(args):
    vid, fps, width, rows, cols = args
    series = signals(grid_flow(decode_gray(ROOT / 'data/videos' / (vid + '.mp4'), fps, width), rows, cols))
    if not np.isfinite(series).all():
        raise ValueError('Nonfinite motion signal: ' + vid)
    return vid, series


def extract(out, plan):
    vids = development_videos(plan); started = time.monotonic(); series = {}
    tasks = [(v, plan['fps'], plan['width'], plan['grid_rows'], plan['grid_cols']) for v in vids]
    with ProcessPoolExecutor(max_workers=plan['workers']) as pool:
        for i, (vid, s) in enumerate(pool.map(one, tasks, chunksize=4), 1):
            series[vid] = s
            if i % 64 == 0 or i == len(vids):
                atomic_json(out / 'progress.json', {'completed': i, 'total': len(vids), 'elapsed_seconds': time.monotonic() - started})
    np.savez_compressed(out / 'results/motion-series.npz', **series)
    finish(out, plan, {'kind': 'ego-motion phase-correlation signals for development videos', 'gate_passed': True,
           'accuracy_score': False, 'videos': len(series), 'pairs_per_clip': sorted({len(s) for s in series.values()}),
           'seconds': time.monotonic() - started})


def ego_labels(plan, vids):
    events = load_events(ROOT / plan['annotation_directory'])
    labels = {}
    for name, prefixes in plan['ego_actions'].items():
        labels[name] = np.asarray([any(e[0].startswith(tuple('ego_act:' + p for p in prefixes)) for e in events[v]['events'])
                                   for v in vids], dtype=bool)
    return labels


def evaluate(out, plan):
    vids = development_videos(plan)
    store = np.load(verified_result(out.parent, plan.get('job_names', {}).get('extract', 'extract'), 'motion-series.npz'))
    motion = np.stack([clip_vector(store[v], plan['profile_bins']) for v in vids])
    manifest = json.loads((ROOT / plan['feature_directory'] / 'feature_manifest.json').read_text())
    index = {v: i for i, v in enumerate(manifest['video_ids'])}
    embedding = load_vectors(ROOT / plan['feature_directory'] / 'aggregate.npz')[[index[v] for v in vids]]
    folds = np.random.default_rng(plan['seed']).permutation(len(vids)) % plan['folds']
    labels = ego_labels(plan, vids); rows = []; started = time.monotonic()
    for name, y in labels.items():
        row = {'action': name, 'positives': int(y.sum())}
        for arm, x in [('embedding', embedding), ('motion', motion), ('combined', np.concatenate([embedding, motion], 1))]:
            row[arm] = auc(cross_validated(x, y, folds), y)
        rows.append(row)
    primary = [r for r in rows if r['action'] in plan['primary_actions']]
    mean = {arm: float(np.mean([r[arm] for r in primary])) for arm in ('embedding', 'motion', 'combined')}
    gate = mean['combined'] >= mean['embedding'] + plan['minimum_auc_gain'] and mean['combined'] >= plan['minimum_combined_auc']
    finish(out, plan, {'kind': 'ego-action AUC: frozen embedding versus motion signals versus combined', 'accuracy_score': True,
           'gate_passed': bool(gate), 'primary_mean_auc': mean, 'actions': rows, 'videos': len(vids),
           'seconds': time.monotonic() - started,
           'limits': 'Clip-level ego-action detection on development videos with random video folds; '
                     'a detection gain is necessary but not sufficient for retrieval gains.'})


JOBS = {'extract': extract, 'evaluate': evaluate}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', choices=JOBS, required=True)
    parser.add_argument('--plan', type=Path, required=True); parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); os.chdir(ROOT)
    import torch
    plan = json.loads(args.plan.read_text()); plan['plan_path'] = str(args.plan)
    if not plan['development_only'] or plan['official_submission_allowed']:
        raise ValueError('Only private local experiments are supported')
    torch.set_num_threads(plan['cpu_threads'])
    stage_directory(args.out)
    JOBS[args.job](args.out, plan)


if __name__ == '__main__':
    main()
