"""32-frame Qwen3.8 atom answers through the learned scorer (2026-10-22/23; CPU; entry 5; rule fixed before results).

Each fold's 16-frame atom adapter answered its held clips on 32 frames (scripts/verifier_variants.py atoms). Arms,
cross-fitted over folds 0-2 x head seeds 43/44, with the scorer retrained on the same kind of answers it is tested on:
  dev   the entry-4 bundle (parser v3, zero-shot top-30 verifier) on the development queries: 16 versus 32 frames
  val   the linear scorer without verifier, trained on development queries, tested on validation queries (unseen
        scenarios): 16 versus 32 frames
Adopt 32-frame atoms if both dev and val pass the standing gate: mean RP gain > 0.001, at least 4/6 cells improve, no
fold regresses, mean MAP change >= 0.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.entry5_headroom import per_query_rp
from scripts.learned_scorer import OUT, load, logit, train_scorer, verify_matrix


def with32(d, f):
    z = np.load(ROOT / f'runs/verifier-variants/atoms-f32-f{f}/vlm-atom-probs.npz')
    if [str(v) for v in z['video_ids']] != d['held']:
        raise ValueError('32-frame answers not aligned with fold %d' % f)
    new = {str(a): logit(z['probs'][:, j]) for j, a in enumerate(z['atoms'])}
    return {**d, 'qwen_logit': {a: new.get(a, l) for a, l in d['qwen_logit'].items()}}


def gate(rows, arm, base):
    pair = {(r['fold'], r['seed']): r for r in rows if r['arm'] == base}
    x = [(r, pair[(r['fold'], r['seed'])]) for r in rows if r['arm'] == arm]
    d = [a['rp'] - b['rp'] for a, b in x]; dm = [a['map'] - b['map'] for a, b in x]
    per_fold = {f: float(np.mean([a['rp'] - b['rp'] for a, b in x if a['fold'] == f])) for f in (0, 1, 2)}
    out = {'mean_rp_delta': float(np.mean(d)), 'mean_map_delta': float(np.mean(dm)), 'cells_improved': int(sum(v > 0 for v in d)),
           'per_fold_rp_delta': per_fold}
    out['passes'] = bool(out['mean_rp_delta'] > .001 and out['cells_improved'] >= 4 and min(per_fold.values()) >= 0 and out['mean_map_delta'] >= 0)
    return out


def main():
    dev16 = {f: load(f, OUT / 'p3') for f in (0, 1, 2)}
    for f, d in dev16.items():
        d['verify'] = verify_matrix(d, ROOT / f'runs/gpu-2026-10-22/verify-f{f}/verify.npz', 30)
    val16 = {f: load(f, OUT / 'val') for f in (0, 1, 2)}
    dev32 = {f: with32(d, f) for f, d in dev16.items()}; val32 = {f: with32(d, f) for f, d in val16.items()}
    rows = []
    for f in (0, 1, 2):
        others = [g for g in (0, 1, 2) if g != f]
        for seed in (43, 44):
            for arm, dv, tv in (('dev-16', dev16, None), ('dev-32', dev32, None), ('val-16', dev16, val16), ('val-32', dev32, val32)):
                train = [dv[g] for g in others]
                if tv is None:
                    test = dv[f]
                else:  # no verifier answers exist for validation queries yet: drop the verifier features
                    train = [{k: v for k, v in d.items() if k != 'verify'} for d in train]; test = tv[f]
                rp, ap_, _, _ = per_query_rp(test, train_scorer(train, seed, False, 10, linear=True)(test))
                rows.append({'fold': f, 'seed': seed, 'arm': arm, 'rp': float(np.nanmean(rp)), 'map': float(np.nanmean(ap_))})
            print(f, seed, {r['arm']: round(r['rp'], 4) for r in rows if r['fold'] == f and r['seed'] == seed}, flush=True)
    report = {'dev': gate(rows, 'dev-32', 'dev-16'), 'val': gate(rows, 'val-32', 'val-16'), 'rows': rows, 'audit_used': False}
    report['adopt'] = report['dev']['passes'] and report['val']['passes']
    (OUT / 'report-atoms32.json').write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != 'rows'}, indent=1))


if __name__ == '__main__':
    main()
