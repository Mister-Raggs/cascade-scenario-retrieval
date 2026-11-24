"""Atom answers from two frame samplings (16 and 32 frames) through the learned scorer (2026-10-23; CPU; entry 5).

Rules fixed before results. Same adapter per fold, two views: the research 16-frame answers and the 32-frame answers
(runs/verifier-variants/atoms-f32-f*). Arms, cross-fitted over folds 0-2 x head seeds 43/44, scorer trained on the
development queries of the other folds, with the zero-shot verifier features on both query sets:
  16      reference (entry-4 features)
  32      32-frame answers in place of the 16-frame ones
  avg     mean of the two views' log-odds
Primary test: validation queries (298 unseen scenarios); secondary: development queries. Adopt an arm over 16 if the
primary passes the standing gate (mean RP gain > 0.001, >= 4/6 cells, no fold regresses, MAP >= 0) and the secondary does
not lose more than 0.005 on average. If both 32 and avg pass, the one with the larger primary gain.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.atoms32_eval import gate
from scripts.entry5_headroom import per_query_rp
from scripts.learned_scorer import OUT, load, logit, train_scorer, verify_matrix


def view(d, f, how):
    if how == '16':
        return d
    z = np.load(ROOT / f'runs/verifier-variants/atoms-f32-f{f}/vlm-atom-probs.npz')
    if [str(v) for v in z['video_ids']] != d['held']:
        raise ValueError('32-frame answers not aligned with fold %d' % f)
    new = {str(a): logit(z['probs'][:, j]) for j, a in enumerate(z['atoms'])}
    if how == '32':
        return {**d, 'qwen_logit': {a: new.get(a, l) for a, l in d['qwen_logit'].items()}}
    return {**d, 'qwen_logit': {a: (l + new[a]) / 2 if a in new else l for a, l in d['qwen_logit'].items()}}


def main():
    dev = {f: load(f, OUT / 'p3') for f in (0, 1, 2)}; val = {f: load(f, OUT / 'val') for f in (0, 1, 2)}
    for f in (0, 1, 2):
        dev[f]['verify'] = verify_matrix(dev[f], ROOT / f'runs/gpu-2026-10-22/verify-f{f}/verify.npz', 30)
        val[f]['verify'] = verify_matrix(val[f], ROOT / f'runs/verifier-variants/val-base-f{f}/verify.npz', 30)
    rows = []
    for how in ('16', '32', 'avg'):
        dv = {f: view(dev[f], f, how) for f in dev}; vl = {f: view(val[f], f, how) for f in val}
        for f in (0, 1, 2):
            others = [g for g in (0, 1, 2) if g != f]
            for seed in (43, 44):
                pr = train_scorer([dv[g] for g in others], seed, False, 10, linear=True)
                for test, d in (('dev', dv[f]), ('val', vl[f])):
                    rp, ap_, _, _ = per_query_rp(d, pr(d))
                    rows.append({'test': test, 'fold': f, 'seed': seed, 'arm': f'{test}-{how}', 'rp': float(np.nanmean(rp)), 'map': float(np.nanmean(ap_))})
        print(how, {r['arm']: round(float(np.mean([x['rp'] for x in rows if x['arm'] == r['arm']])), 4) for r in rows if r['arm'].endswith(how)}, flush=True)
    out = {}
    for how in ('32', 'avg'):
        out[how] = {'val': gate(rows, f'val-{how}', 'val-16'), 'dev': gate(rows, f'dev-{how}', 'dev-16')}
        out[how]['adopt'] = bool(out[how]['val']['passes'] and out[how]['dev']['mean_rp_delta'] >= -0.005)
    (OUT / 'report-atoms-views.json').write_text(json.dumps({**out, 'rows': rows, 'audit_used': False}, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
