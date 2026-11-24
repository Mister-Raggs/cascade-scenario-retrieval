"""Entry-5 joint recipe J2: J1p with the fine-tuned verifier (2026-10-23; CPU; rule fixed before results).

The 2026-10-22 multi-task fine-tuned verifier (mtv adapters, one per development fold, trained on fitting clips) passed on
validation queries at top 30 (+0.0088, 6/6), and the zero-shot verifier at top 50 passed too (+0.0071, 6/6). J2 combines
them, with candidates ranked as the container will (entry-3 recipe with 32-frame atom answers).
Arms (32-frame atoms, Qwen3.8 32-frame properties, linear scorer on development + 25% validation queries, seeds 43/44):
  J1p     zero-shot verifier on top 30 (16-frame-ranked candidates; the measured J1p)
  J2-30   fine-tuned verifier on the 32-frame-ranked top 30 (top-50 answers restricted to each query's top 30)
  J2-50   fine-tuned verifier on the 32-frame-ranked top 50
Pass (each J2 arm versus J1p): validation test (unseen scenario halves) standing gate (mean RP gain > 0.001, >= 4/6
cells, no fold regresses, MAP >= 0) and development mean RP change >= -0.005. Choice: J2-50 only if it passes and beats
J2-30 by more than +0.001 on validation (it costs about 20 more A100 minutes); otherwise J2-30 if it passes; else J1p.
"""
from __future__ import annotations

import collections
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.atoms_views_eval import view
from scripts.candidates32 import top_k
from scripts.entry5_headroom import per_query_rp
from scripts.learned_scorer import OUT, load, train_scorer, verify_matrix
from scripts.props_val import VV, with_props
from scripts.val_eval import scenarios, subset


def restrict(d, m, d_v2, f):
    """Keep only answers on each query's 32-frame-ranked top 30 (the same ranking that produced the top 50)."""
    top = top_k(view(d_v2, f, '32'), 30); keep = np.zeros(m.shape, bool); pos = {v: j for j, v in enumerate(d['held'])}; qi = {q: i for i, q in enumerate(d['dev'])}
    for v, qs in top.items():
        for q in qs:
            keep[qi[q], pos[v]] = True
    return np.where(keep, m, np.nan)


def main():
    half, _ = scenarios()
    base_dev = {f: with_props(view(load(f, OUT / 'p3'), f, '32'), f, 'qwen') for f in (0, 1, 2)}
    base_val = {f: with_props(view(load(f, OUT / 'val'), f, '32'), f, 'qwen') for f in (0, 1, 2)}
    arms = {}
    j1 = ({f: {**d, 'verify': verify_matrix(d, ROOT / f'runs/gpu-2026-10-22/verify-f{f}/verify.npz', 30)} for f, d in base_dev.items()},
          {f: {**d, 'verify': verify_matrix(d, VV / f'val-base-f{f}/verify.npz', 30)} for f, d in base_val.items()})
    arms['J1p'] = j1
    ft_dev = {f: verify_matrix(d, VV / f'ft-c50-f{f}/verify.npz', 50) for f, d in base_dev.items()}
    ft_val = {f: verify_matrix(d, VV / f'val-ft-c50-f{f}/verify.npz', 50) for f, d in base_val.items()}
    arms['J2-50'] = ({f: {**d, 'verify': ft_dev[f]} for f, d in base_dev.items()}, {f: {**d, 'verify': ft_val[f]} for f, d in base_val.items()})
    arms['J2-30'] = ({f: {**d, 'verify': restrict(d, ft_dev[f], load(f), f)} for f, d in base_dev.items()},
                     {f: {**d, 'verify': restrict(d, ft_val[f], load(f, OUT / 'val-v2'), f)} for f, d in base_val.items()})
    keep = lambda d, m: {**subset(d, m), 'prop_score': d['prop_score'][np.asarray(m, bool)], 'prop_mask': d['prop_mask'][np.asarray(m, bool)]}
    rows = []
    for arm, (dv, vl) in arms.items():
        for f in (0, 1, 2):
            others = [g for g in (0, 1, 2) if g != f]
            for seed in (43, 44):
                rng = np.random.default_rng(100 + f)
                for h in (0, 1):
                    extra = [keep(vl[g], np.asarray([half[q] != h for q in vl[g]['dev']]) & (rng.random(len(vl[g]['dev'])) < .25)) for g in others]
                    test = keep(vl[f], [half[q] == h for q in vl[f]['dev']])
                    rp, ap_, _, _ = per_query_rp(test, train_scorer([dv[g] for g in others] + extra, seed, False, 10, linear=True)(test))
                    rows.append({'test': 'val', 'arm': arm, 'fold': f, 'seed': seed, 'half': h, 'rp': float(np.nanmean(rp)), 'map': float(np.nanmean(ap_))})
                extra = [keep(vl[g], rng.random(len(vl[g]['dev'])) < .25) for g in others]
                rp, ap_, _, _ = per_query_rp(dv[f], train_scorer([dv[g] for g in others] + extra, seed, False, 10, linear=True)(dv[f]))
                rows.append({'test': 'dev', 'arm': arm, 'fold': f, 'seed': seed, 'half': -1, 'rp': float(np.nanmean(rp)), 'map': float(np.nanmean(ap_))})
        print(arm, {t: round(float(np.mean([r['rp'] for r in rows if r['arm'] == arm and r['test'] == t])), 4) for t in ('val', 'dev')}, flush=True)
    out = {}
    for arm in ('J2-30', 'J2-50'):
        res = {}
        for t in ('val', 'dev'):
            ref = {(r['fold'], r['seed'], r['half']): r for r in rows if r['arm'] == 'J1p' and r['test'] == t}
            x = [r for r in rows if r['arm'] == arm and r['test'] == t]; d = [r['rp'] - ref[(r['fold'], r['seed'], r['half'])]['rp'] for r in x]
            cells = collections.defaultdict(list)
            for r, v in zip(x, d):
                cells[(r['fold'], r['seed'])].append(v)
            res[t] = {'rp': float(np.mean([r['rp'] for r in x])), 'rp_delta': float(np.mean(d)),
                      'map_delta': float(np.mean([r['map'] - ref[(r['fold'], r['seed'], r['half'])]['map'] for r in x])),
                      'cells_improved': int(sum(np.mean(v) > 0 for v in cells.values())),
                      'per_fold_rp_delta': {f: float(np.mean([v for r, v in zip(x, d) if r['fold'] == f])) for f in (0, 1, 2)}}
        v = res['val']; res['passes'] = bool(v['rp_delta'] > .001 and v['cells_improved'] >= 4 and min(v['per_fold_rp_delta'].values()) >= 0
                                             and v['map_delta'] >= 0 and res['dev']['rp_delta'] >= -0.005)
        out[arm] = res
    if out['J2-50']['passes'] and (not out['J2-30']['passes'] or out['J2-50']['val']['rp_delta'] > out['J2-30']['val']['rp_delta'] + 0.001):
        choice = 'J2-50'
    else:
        choice = 'J2-30' if out['J2-30']['passes'] else 'J1p'
    (OUT / 'report-joint2.json').write_text(json.dumps({'arms': out, 'choice': choice, 'rows': rows, 'audit_used': False}, indent=1))
    print(json.dumps({'arms': out, 'choice': choice}, indent=1))


if __name__ == '__main__':
    main()
