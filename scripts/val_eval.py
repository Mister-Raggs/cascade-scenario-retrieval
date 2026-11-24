"""Learned-scorer evaluation on validation queries (2026-10-22; CPU; entry 5; rules fixed before results).

Uses runs/learned-scorer/val (scripts/val_queries.py). Scenarios: queries with the identical relevant-clip set over the
full corpus are paraphrases of one scenario; scenarios are split into halves A/B (seeded). Every test is on clips held
out of the fold and, for scorers trained on validation queries, on validation scenarios not seen in training.

Arms, tested on validation queries of held fold f (head seeds 43/44), scorers trained on folds g != f:
  entry3          the entry-3 recipe (t4 + Qwen3.8 late fusion, rank fusion 0.8), no learning
  linear-dev      the entry-4 scorer recipe without verifier: trained on the 711 development queries (current practice)
  linear-devval   trained on development queries + validation scenarios of the other half
  mlp-devval      2-layer MLP (64 hidden, 10 epochs) on the same training data
  verify arms     the same with the verifier features, when runs/verifier-variants/val-<variant>-f*/verify.npz exist
Reported by half and overall, as RP/MAP deltas against linear-dev on the same cells. Standing gate for adopting a change:
mean RP gain > 0.001, at least 4/6 fold-seed cells improve, no fold regresses, MAP >= 0.
Usage: val_eval.py [--verifier VARIANT ...]
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.entry5_headroom import per_query_rp
from scripts.learned_scorer import OUT, combined, load, summed, train_scorer, verify_matrix

SEEDS = (43, 44)


def scenarios():
    from baseline.io import unique_rows
    rels = unique_rows(ROOT / 'data/retrieval/cascade-v1.0/val/qrels.jsonl', 'query_id')
    key = {q: tuple(sorted(r['relevant_video_ids'])) for q, r in rels.items()}
    groups = sorted(set(key.values())); rng = np.random.default_rng(5)
    half = {g: int(b) for g, b in zip(groups, rng.permutation(len(groups)) % 2)}
    return {q: half[key[q]] for q in key}, len(groups)


def subset(d, keep):
    keep = np.asarray(keep, bool); out = dict(d)
    for k in ('rel', 'qvec', *[x for x in d if x.startswith(('raw_', 'sub_'))]):
        out[k] = d[k][keep]
    out['dev'] = [q for q, k in zip(d['dev'], keep) if k]
    if 'verify' in d:
        out['verify'] = d['verify'][keep]
    if 'only' in d:
        out['only'] = d['only'][keep]
    return out


def entry3(d, seed):
    from baseline.native_ranking import blended_scores
    s, m = summed(d, combined(d)); raw, sub = d[f'raw_{seed}'], d[f'sub_{seed}']
    return np.where(m[:, None], blended_scores(np.where(m[:, None], s, raw), sub, .8), sub)


def main():
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('--verifier', nargs='*', default=[])
    a = ap.parse_args()
    half, n_groups = scenarios()
    dev = {f: load(f, OUT / 'p3') for f in (0, 1, 2)}; val = {f: load(f, OUT / 'val') for f in (0, 1, 2)}
    vv = ROOT / 'runs/verifier-variants'
    with_v = {}
    for name in a.verifier:
        if all((vv / f'val-{name}-f{f}/verify.npz').exists() for f in (0, 1, 2)):
            with_v[name] = ({f: {**dev[f], 'verify': verify_matrix(dev[f], ROOT / ('runs/gpu-2026-10-22/verify-f%d/verify.npz' % f) if name == 'base'
                                                                  else vv / f'{name}-f{f}/verify.npz', 30)} for f in dev},
                            {f: {**val[f], 'verify': verify_matrix(val[f], vv / f'val-{name}-f{f}/verify.npz', 30)} for f in val})
    rows = []
    for f in (0, 1, 2):
        for h in (0, 1):
            test_keep = [half[q] == h for q in val[f]['dev']]; train_keep = lambda d: [half[q] != h for q in d['dev']]
            for seed in SEEDS:
                tests = {}
                t = subset(val[f], test_keep)
                tests['entry3'] = (t, entry3(t, seed))
                others = [g for g in (0, 1, 2) if g != f]
                tests['linear-dev'] = (t, train_scorer([dev[g] for g in others], seed, False, 10, linear=True)(t))
                both = [dev[g] for g in others] + [subset(val[g], train_keep(val[g])) for g in others]
                tests['linear-devval'] = (t, train_scorer(both, seed, False, 10, linear=True)(t))
                tests['mlp-devval'] = (t, train_scorer(both, seed, False, 10)(t))
                for name, (dv, vl) in with_v.items():
                    tv = subset(vl[f], test_keep)
                    tests[f'linear-dev+verify-{name}'] = (tv, train_scorer([dv[g] for g in others], seed, False, 10, linear=True)(tv))
                    bv = [dv[g] for g in others] + [subset(vl[g], train_keep(vl[g])) for g in others]
                    tests[f'linear-devval+verify-{name}'] = (tv, train_scorer(bv, seed, False, 10, linear=True)(tv))
                for arm, (td, sc) in tests.items():
                    rp, ap_, _, _ = per_query_rp(td, sc)
                    rows.append({'fold': f, 'half': h, 'seed': seed, 'arm': arm, 'rp': float(np.nanmean(rp)), 'map': float(np.nanmean(ap_))})
                print(f, h, seed, {r['arm']: round(r['rp'], 4) for r in rows if r['fold'] == f and r['half'] == h and r['seed'] == seed}, flush=True)
    summary = {}
    for arm in sorted({r['arm'] for r in rows}):
        x = [r for r in rows if r['arm'] == arm]
        base = {(r['fold'], r['half'], r['seed']): r for r in rows if r['arm'] == 'linear-dev'}
        d = [r['rp'] - base[(r['fold'], r['half'], r['seed'])]['rp'] for r in x]
        cells = collections.defaultdict(list)
        for r, dd in zip(x, d):
            cells[(r['fold'], r['seed'])].append(dd)
        cell = [np.mean(v) for v in cells.values()]
        per_fold = {f: float(np.mean([dd for r, dd in zip(x, d) if r['fold'] == f])) for f in (0, 1, 2)}
        summary[arm] = {'mean_rp': float(np.mean([r['rp'] for r in x])), 'mean_map': float(np.mean([r['map'] for r in x])),
                        'rp_delta_vs_linear_dev': float(np.mean(d)),
                        'map_delta_vs_linear_dev': float(np.mean([r['map'] - base[(r['fold'], r['half'], r['seed'])]['map'] for r in x])),
                        'cells_improved': int(sum(c > 0 for c in cell)), 'per_fold_rp_delta': per_fold}
    report = {'kind': 'learned scorer on validation queries (unseen scenarios), development folds 0-2', 'scenarios': n_groups,
              'audit_used': False, 'summary': summary, 'rows': rows}
    (OUT / 'report-val.json').write_text(json.dumps(report, indent=1))
    print(f"\n{n_groups} validation scenarios")
    for arm, s in sorted(summary.items(), key=lambda kv: -kv[1]['mean_rp']):
        print(f"{arm:34s} RP {s['mean_rp']:.4f}  dRP {s['rp_delta_vs_linear_dev']:+.4f}  dMAP {s['map_delta_vs_linear_dev']:+.4f}  cells {s['cells_improved']}/6  folds {[round(v, 4) for v in s['per_fold_rp_delta'].values()]}")


if __name__ == '__main__':
    main()
