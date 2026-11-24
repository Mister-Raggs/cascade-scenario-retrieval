"""Property atoms re-tested where they occur: validation queries (2026-10-23; entry 5; rules fixed before results).

The 2026-10-22 property test failed on development queries, where only 7 of 64 scenarios use a property (signalling,
moving slowly, double-parked, flashing or ego-affecting lights). Validation queries use them in 16% of queries (45 of 278
scenarios), and with perfect property labels their AND-set overlap with the relevant set rises from 0.27 to 0.64.

  probes        (CPU) per fold, t4-style late-fused linear probes (pooled, motion2, windows; scripts/structured_v2.COMBO)
                fitted on the fold's fitting clips for every property atom used by development or validation queries,
                predicting its held clips: runs/learned-scorer/props-all-f{f}.npz
  atoms-list    writes that atom list for the GPU job (runs/learned-scorer/props-all-atoms.json)
  eval          (CPU) the J1 recipe (32-frame atoms, linear scorer on development + 25% validation queries, zero-shot
                verifier) with two extra features, each query's summed property log-probability (z within the query, 0
                without properties) and a has-property flag, from: probes, qwen (zero-shot Qwen3.8 at 32 frames,
                runs/verifier-variants/props32-f*), and both (mean of the two log-odds). Versus J1.
Pass: the validation test (unseen scenario halves) passes the standing gate against J1 and the development mean RP change
is >= -0.005. Choose the passing arm with the largest validation gain.
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
from scripts.learned_scorer import OUT, PLAN, load, logit, query_texts, train_scorer, verify_matrix

VV = ROOT / 'runs/verifier-variants'


def prop_atoms_of(q):
    from baseline.property_atoms import query_atoms
    from baseline.query_parser import Unparsed
    from baseline.query_parser_v3 import translate
    try:
        return query_atoms(translate(query_texts()[q]))
    except Unparsed:
        return []


def all_atoms():
    dev = load(0, OUT / 'p3'); val = load(0, OUT / 'val')
    return sorted({a for d in (dev, val) for q in d['dev'] for a in prop_atoms_of(q)})


def probes():
    from baseline.probe import fit_logistic, standardize
    from baseline.property_atoms import load_labels, present
    from scripts.structured_eval import provider
    from scripts.structured_v2 import COMBO, late
    from scripts.tracks_eval import common
    plan = json.loads(PLAN.read_text()); lab = load_labels(ROOT / plan['annotation_directory']); atoms = all_atoms()
    (OUT / 'props-all-atoms.json').write_text(json.dumps(atoms))
    for f in (0, 1, 2):
        _, _, _, _, vids, ix, _, _ = common({**plan, 'held_fold': f})
        fit, held = np.asarray(ix['fit_v']), np.asarray(ix['held_v'])
        blocks = {b: provider(b, plan['blocks'][b], vids)[0] for b in COMBO}; probs = np.zeros((len(held), len(atoms)), np.float32)
        for j, a in enumerate(atoms):
            y = np.asarray([present(a, lab[v]) for v in vids])
            parts = []
            for b in COMBO:
                if y[fit].sum() < 3 or y[fit].sum() > len(fit) - 3:
                    parts.append(np.full(len(held), float(np.clip(y[fit].mean(), 1e-3, 1 - 1e-3)))); continue
                z = standardize(blocks[b], fit); w, c = fit_logistic(z[fit], y[fit])
                parts.append(np.clip(1 / (1 + np.exp(-(z[held] @ w + c))), 1e-4, 1 - 1e-4))
            probs[:, j] = late(parts)
        np.savez_compressed(OUT / f'props-all-f{f}.npz', atoms=np.asarray(atoms), video_ids=np.asarray([vids[i] for i in held]), probs=probs)
        print(f'fold {f}: {len(atoms)} property atoms x {len(held)} held clips', flush=True)


def clip_logits(f, source, held):
    z = {}
    if source in ('probes', 'both'):
        p = np.load(OUT / f'props-all-f{f}.npz'); assert [str(v) for v in p['video_ids']] == held
        z['probes'] = {str(a): logit(p['probs'][:, j]) for j, a in enumerate(p['atoms'])}
    if source in ('qwen', 'both'):
        p = np.load(VV / f'props32-f{f}/props.npz'); assert [str(v) for v in p['video_ids']] == held
        z['qwen'] = {str(a): logit(p['probs'][:, j]) for j, a in enumerate(p['atoms'])}
    if source == 'both':
        return {a: (z['probes'][a] + z['qwen'][a]) / 2 for a in z['probes'] if a in z['qwen']}
    return z[source]


def with_props(d, f, source):
    lo = clip_logits(f, source, d['held']); score = np.zeros(d['rel'].shape); mask = np.zeros(len(d['dev']), bool)
    for i, q in enumerate(d['dev']):
        x = [a for a in prop_atoms_of(q) if a in lo]
        if x:
            score[i] = np.sum([np.log(1 / (1 + np.exp(-lo[a]))) for a in x], 0); mask[i] = True
    return {**d, 'prop_score': score, 'prop_mask': mask}


def evaluate():
    from scripts.atoms_views_eval import view
    from scripts.entry5_headroom import per_query_rp
    from scripts.scorer_mix_eval import sample
    from scripts.val_eval import scenarios, subset
    half, _ = scenarios()
    dev = {f: view(load(f, OUT / 'p3'), f, '32') for f in (0, 1, 2)}; val = {f: view(load(f, OUT / 'val'), f, '32') for f in (0, 1, 2)}
    for f in (0, 1, 2):
        dev[f]['verify'] = verify_matrix(dev[f], ROOT / f'runs/gpu-2026-10-22/verify-f{f}/verify.npz', 30)
        val[f]['verify'] = verify_matrix(val[f], VV / f'val-base-f{f}/verify.npz', 30)
    sources = ['probes'] + (['qwen', 'both'] if all((VV / f'props32-f{f}/props.npz').exists() for f in (0, 1, 2)) else [])
    arms = {'J1': (dev, val)}
    for s in sources:
        arms[f'J1+props-{s}'] = ({f: with_props(dev[f], f, s) for f in dev}, {f: with_props(val[f], f, s) for f in val})
    keep_mask = lambda d, m: {**subset(d, m), **({'prop_score': d['prop_score'][np.asarray(m, bool)], 'prop_mask': d['prop_mask'][np.asarray(m, bool)]} if 'prop_score' in d else {})}
    rows = []
    for arm, (dv, vl) in arms.items():
        for f in (0, 1, 2):
            others = [g for g in (0, 1, 2) if g != f]
            for seed in (43, 44):
                rng = np.random.default_rng(100 + f)
                for h in (0, 1):
                    extra = []
                    for g in others:
                        m = np.asarray([half[q] != h for q in vl[g]['dev']]) & (rng.random(len(vl[g]['dev'])) < .25)
                        extra.append(keep_mask(vl[g], m))
                    test = keep_mask(vl[f], [half[q] == h for q in vl[f]['dev']])
                    rp, ap_, _, _ = per_query_rp(test, train_scorer([dv[g] for g in others] + extra, seed, False, 10, linear=True)(test))
                    rows.append({'test': 'val', 'arm': arm, 'fold': f, 'seed': seed, 'half': h, 'rp': float(np.nanmean(rp)), 'map': float(np.nanmean(ap_))})
                extra = [keep_mask(vl[g], rng.random(len(vl[g]['dev'])) < .25) for g in others]
                rp, ap_, _, _ = per_query_rp(dv[f], train_scorer([dv[g] for g in others] + extra, seed, False, 10, linear=True)(dv[f]))
                rows.append({'test': 'dev', 'arm': arm, 'fold': f, 'seed': seed, 'half': -1, 'rp': float(np.nanmean(rp)), 'map': float(np.nanmean(ap_))})
        print(arm, {t: round(float(np.mean([r['rp'] for r in rows if r['arm'] == arm and r['test'] == t])), 4) for t in ('val', 'dev')}, flush=True)
    out = {}
    for arm in arms:
        if arm == 'J1':
            continue
        res = {}
        for t in ('val', 'dev'):
            ref = {(r['fold'], r['seed'], r['half']): r for r in rows if r['arm'] == 'J1' and r['test'] == t}
            x = [r for r in rows if r['arm'] == arm and r['test'] == t]; d = [r['rp'] - ref[(r['fold'], r['seed'], r['half'])]['rp'] for r in x]
            cells = collections.defaultdict(list)
            for r, v in zip(x, d):
                cells[(r['fold'], r['seed'])].append(v)
            res[t] = {'rp_delta': float(np.mean(d)), 'map_delta': float(np.mean([r['map'] - ref[(r['fold'], r['seed'], r['half'])]['map'] for r in x])),
                      'cells_improved': int(sum(np.mean(v) > 0 for v in cells.values())),
                      'per_fold_rp_delta': {f: float(np.mean([v for r, v in zip(x, d) if r['fold'] == f])) for f in (0, 1, 2)}}
        v = res['val']; res['passes'] = bool(v['rp_delta'] > .001 and v['cells_improved'] >= 4 and min(v['per_fold_rp_delta'].values()) >= 0
                                             and v['map_delta'] >= 0 and res['dev']['rp_delta'] >= -0.005)
        out[arm] = res
    (OUT / 'report-props-val.json').write_text(json.dumps({'arms': out, 'rows': rows, 'audit_used': False}, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('job', choices=['probes', 'eval'])
    a = ap.parse_args()
    probes() if a.job == 'probes' else evaluate()
