"""How many validation queries should train the final scorer? (2026-10-23; CPU; entry 5; rule fixed before results.)

The linear scorer (entry-4 features with the zero-shot verifier) is trained on development queries of folds g != f plus a
seeded fraction p of the available validation queries of folds g != f, for p in {0, 0.25, 0.5, 1}. Tests on fold f:
  val  validation queries of scenario half h, with that half's scenarios excluded from training (unseen scenarios)
  dev  development queries
Choice: the p with the largest mean relative RP change over the two tests (each relative to p = 0), provided neither test
loses more than 0.005 RP on average. Head seeds 43/44.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.entry5_headroom import per_query_rp
from scripts.learned_scorer import OUT, load, train_scorer, verify_matrix
from scripts.val_eval import scenarios, subset

FRACTIONS = (0.0, 0.25, 0.5, 1.0)


def sample(d, keep, p, rng):
    keep = np.asarray(keep, bool) & (rng.random(len(keep)) < p)
    return subset(d, keep) if keep.any() else None


def main():
    half, _ = scenarios()
    dev = {f: load(f, OUT / 'p3') for f in (0, 1, 2)}; val = {f: load(f, OUT / 'val') for f in (0, 1, 2)}
    for f in (0, 1, 2):
        dev[f]['verify'] = verify_matrix(dev[f], ROOT / f'runs/gpu-2026-10-22/verify-f{f}/verify.npz', 30)
        val[f]['verify'] = verify_matrix(val[f], ROOT / f'runs/verifier-variants/val-base-f{f}/verify.npz', 30)
    res = {p: {'val': [], 'dev': []} for p in FRACTIONS}
    for f in (0, 1, 2):
        others = [g for g in (0, 1, 2) if g != f]
        for seed in (43, 44):
            for p in FRACTIONS:
                rng = np.random.default_rng(100 + f)
                for h in (0, 1):
                    extra = [sample(val[g], [half[q] != h for q in val[g]['dev']], p, rng) for g in others]
                    train = [dev[g] for g in others] + [e for e in extra if e is not None]
                    test = subset(val[f], [half[q] == h for q in val[f]['dev']])
                    res[p]['val'].append(float(np.nanmean(per_query_rp(test, train_scorer(train, seed, False, 10, linear=True)(test))[0])))
                extra = [sample(val[g], [True] * len(val[g]['dev']), p, rng) for g in others]
                train = [dev[g] for g in others] + [e for e in extra if e is not None]
                res[p]['dev'].append(float(np.nanmean(per_query_rp(dev[f], train_scorer(train, seed, False, 10, linear=True)(dev[f]))[0])))
            print(f, seed, {p: (round(res[p]['val'][-1], 4), round(res[p]['dev'][-1], 4)) for p in FRACTIONS}, flush=True)
    base = {t: np.mean(res[0.0][t]) for t in ('val', 'dev')}
    summary = {}
    for p in FRACTIONS:
        m = {t: float(np.mean(res[p][t])) for t in ('val', 'dev')}
        summary[str(p)] = {'val_rp': m['val'], 'dev_rp': m['dev'], 'val_delta': m['val'] - base['val'], 'dev_delta': m['dev'] - base['dev'],
                           'mean_relative': float(np.mean([(m[t] - base[t]) / base[t] for t in m])),
                           'eligible': bool(min(m['val'] - base['val'], m['dev'] - base['dev']) >= -0.005)}
    best = max((p for p in summary if summary[p]['eligible']), key=lambda p: summary[p]['mean_relative'])
    (OUT / 'report-scorer-mix.json').write_text(json.dumps({'summary': summary, 'choice': best, 'raw': {str(k): v for k, v in res.items()}}, indent=1))
    for p, s in summary.items():
        print(p, {k: round(v, 4) if isinstance(v, float) else v for k, v in s.items()})
    print('choice', best)


if __name__ == '__main__':
    main()
