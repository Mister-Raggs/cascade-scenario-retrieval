"""Entry-5 headroom diagnostics on the entry-4 bundle (2026-10-22; development folds 0-2, CPU; diagnostic only).

No gate and no audit look: this measures where the remaining R-Precision is, to choose entry 5's target. Reference is the
audit-confirmed bundle (linear scorer, parser-v3 caches, zero-shot top-30 verifier features), cross-fitted per fold and
head seed exactly as learned_scorer.py eval-verify2 ('linear-p3-verify').

Perception arms replace the held clips' t4 and Qwen3.8 log-odds for one atom group:
  oracle         ground truth (probability 0.02 / 0.98)
  shift-D        current log-odds + D * (y - 0.5): the current model made better, keeping its error structure
  indep-A        independent equal-variance Gaussian logits at per-atom AUC A (the 2026-10-19 ceiling model)
Each runs in two modes: retrain (training folds transformed too, so the scorer learns to trust the better input) and
plug (the reference scorer applied to transformed test features).
Verifier arms replace the whole-query verifier log-odds within each query's candidates:
  perfect-K      +4 relevant / -4 not, on the top-K candidates of the entry-3 recipe (K = 30 is the current set)
  shift-D        current verifier log-odds + D * (rel - 0.5) on the current top-30
Also reported: per-atom held AUCs, candidate coverage and optimal top-K reorder ceilings, the pure parse-and-truth ceiling
(AND of ground-truth atoms per parsed query), RP by query relation, and the queries that stay poor with perfect atoms.
Usage: entry5_headroom.py [--workers N] [--quick]
"""
from __future__ import annotations

import argparse
import collections
import json
from multiprocessing import get_context
from pathlib import Path
from statistics import NormalDist
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.learned_scorer import OUT, combined, load, logit, query_texts, sigmoid, summed, train_scorer, verify_matrix, zrow

VERIFY = ROOT / 'runs/gpu-2026-10-22'
RESULT = ROOT / 'runs/entry5-headroom'
SEEDS = (43, 44)
HARD = ('decel', 'nudge', 'change_lane')
GROUPS = {
    'agent-actions': lambda a: a.startswith('agent|action='),
    'agent-types': lambda a: a.startswith('agent|type='),
    'agent-all': lambda a: a.startswith('agent|'),
    'ego-hard': lambda a: a.startswith('ego|') and any(k in a for k in HARD),
    'ego-other': lambda a: a.startswith('ego|') and not any(k in a for k in HARD),
    'ego-all': lambda a: a.startswith('ego|'),
    'light': lambda a: a.startswith('light|'),
    'obj': lambda a: a.startswith('obj|'),
    'all-atoms': lambda a: True,
    'agent-stop': lambda a: a.startswith('agent|action=stop'),
    'agent-turn': lambda a: a.startswith('agent|action=turn'),
    'agent-ped': lambda a: a.startswith('agent|action=') and 'type=ped' in a,
    'agent-lane': lambda a: a.startswith('agent|action=change_lane'),
    'agent-move-other': lambda a: a.startswith('agent|action=') and not a.startswith(('agent|action=stop', 'agent|action=turn', 'agent|action=change_lane')) and 'type=ped' not in a,
}
FOLDS = {}


def auc(score, label):
    label = np.asarray(label, bool); n1, n0 = label.sum(), (~label).sum()
    if not n1 or not n0:
        return None
    ranks = np.empty(len(score)); ranks[np.argsort(score, kind='stable')] = np.arange(1, len(score) + 1)
    return float((ranks[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def per_query_rp(d, scores):
    """R-Precision per query (NaN without relevant clips), ties broken as baseline.hubness.ranking_metrics."""
    order = np.argsort(np.asarray(d['held']), kind='stable'); rel = d['rel'][:, order]
    ranking = np.argsort(-np.asarray(scores)[:, order], axis=1, kind='stable'); hits = np.take_along_axis(rel, ranking, 1)
    counts = rel.sum(1); pos = np.arange(1, rel.shape[1] + 1)
    rp = np.divide((hits & (pos <= counts[:, None])).sum(1), counts, out=np.full(len(counts), np.nan), where=counts > 0)
    ap = np.divide((hits * hits.cumsum(1) / pos).sum(1), counts, out=np.full(len(counts), np.nan), where=counts > 0)
    return rp, ap, hits, counts


def topk_stats(d, scores, ks=(10, 30, 50, 100)):
    """Recall of relevant clips in the top K, and RP if only the top K were reordered perfectly."""
    _, _, hits, counts = per_query_rp(d, scores); ok = counts > 0; out = {}
    for k in ks:
        found = hits[:, :k].sum(1)
        out[k] = {'recall': float(np.mean(found[ok] / counts[ok])),
                  'reorder_ceiling_rp': float(np.mean(np.minimum(found[ok], counts[ok]) / counts[ok]))}
        # reordering the top K moves every relevant clip found there to the front; RP counts min(found, R) of R
    return out


def entry3_scores(f, seed=43):
    from baseline.native_ranking import blended_scores
    d = load(f); s, m = summed(d, combined(d)); raw, sub = d[f'raw_{seed}'], d[f'sub_{seed}']
    return d, np.where(m[:, None], blended_scores(np.where(m[:, None], s, raw), sub, .8), sub)


def base():
    folds = {}
    for f in (0, 1, 2):
        d = load(f, OUT / 'p3'); d['verify'] = verify_matrix(d, VERIFY / f'verify-f{f}/verify.npz', 30)
        d2, e3 = entry3_scores(f)
        if d2['dev'] != d['dev'] or d2['held'] != d['held']:
            raise ValueError('v2 and v3 caches are not aligned')
        d['entry3'] = e3; d['fold'] = f; folds[f] = d
    return folds


def usage(d):
    return collections.Counter(a for q in d['dev'] for a in (d['parsed'].get(q) or []))


# ---------- transforms (each returns a new fold dict; deterministic per fold) ----------

def perception(kind, group, param):
    pick = GROUPS[group]

    def apply(d):
        rng = np.random.default_rng(1000 + d['fold']); t4, qw = dict(d['t4_logit']), dict(d['qwen_logit'])
        for a in d['atoms']:
            if not pick(a):
                continue
            y = d['truth_of'][a].astype(float)
            if kind == 'oracle':
                t4[a] = logit(np.clip(y, .02, .98))
                if a in qw:
                    qw[a] = t4[a]
            elif kind == 'shift':
                t4[a] = t4[a] + param * (y - .5)
                if a in qw:
                    qw[a] = qw[a] + param * (y - .5)
            elif kind == 'indep':
                z = 2 ** .5 * NormalDist().inv_cdf(param) * (y - .5) + rng.normal(size=len(y))
                t4[a] = z
                if a in qw:
                    qw[a] = z
        return {**d, 't4_logit': t4, 'qwen_logit': qw}
    return apply


def verifier(kind, param):
    def apply(d):
        v = d['verify']; rel = d['rel'].astype(float)
        if kind == 'perfect':
            if param == 30:
                cand = np.isfinite(v)
            else:
                top = np.argsort(-d['entry3'], 1, kind='stable')[:, :param]; cand = np.zeros(v.shape, bool)
                np.put_along_axis(cand, top, True, 1)
            new = np.where(cand, 8 * (rel - .5), np.nan)
        elif kind == 'shift':
            new = v + param * (rel - .5)
        return {**d, 'verify': new.astype(np.float32)}
    return apply


def chain(*fs):
    def apply(d):
        for f in fs:
            d = f(d)
        return d
    return apply


def arms(quick=False):
    out = {}
    groups = ('agent-actions', 'all-atoms') if quick else tuple(GROUPS)
    for g in groups:
        out[f'oracle:{g}'] = perception('oracle', g, None)
    for g in (('agent-actions',) if quick else ('agent-actions', 'agent-types', 'ego-hard', 'ego-other', 'light', 'obj', 'all-atoms')):
        out[f'shift2:{g}'] = perception('shift', g, 2.0)
    if not quick:
        for dlt in (0.5, 1.0, 3.0, 4.0, 6.0):
            out[f'shift{dlt:g}:agent-actions'] = perception('shift', 'agent-actions', dlt)
        for dlt in (1.0, 3.0):
            out[f'shift{dlt:g}:all-atoms'] = perception('shift', 'all-atoms', dlt)
        for lv in (.85, .90, .95):
            out[f'indep{lv:.2f}:agent-actions'] = perception('indep', 'agent-actions', lv)
        for k in (30, 50, 100):
            out[f'verify-perfect{k}'] = verifier('perfect', k)
        for dlt in (0.25, 0.5, 1.0, 2.0, 4.0):
            out[f'verify-shift{dlt:g}'] = verifier('shift', dlt)
        for dlt in (1.0, 2.0):
            out[f'shift{dlt:g}:ego-hard'] = perception('shift', 'ego-hard', dlt)
        out['oracle:agent-actions+verify-perfect30'] = chain(perception('oracle', 'agent-actions', None), verifier('perfect', 30))
        out['shift2:agent-actions+verify-shift2'] = chain(perception('shift', 'agent-actions', 2.0), verifier('shift', 2.0))
        out['oracle:all-atoms+verify-perfect30'] = chain(perception('oracle', 'all-atoms', None), verifier('perfect', 30))
    else:
        out['verify-perfect30'] = verifier('perfect', 30)
    return out


# ---------- jobs ----------

def run_job(job):
    import torch
    torch.set_num_threads(1)
    name, mode, f, seed = job; t = time.monotonic(); table = ARMS
    test = FOLDS[f]; train = [FOLDS[g] for g in FOLDS if g != f]; result = {}
    if mode == 'reference':
        predict = train_scorer(train, seed, False, 10, linear=True)
        result['bundle'] = predict(test)
        for arm, tf in table.items():
            result[f'{arm} [plug]'] = predict(tf(test))
    else:
        tf = table[name]; result[f'{name} [retrain]'] = train_scorer([tf(d) for d in train], seed, False, 10, linear=True)(tf(test))
    rows = {}
    for arm, scores in result.items():
        rp, ap, _, _ = per_query_rp(test, scores)
        rows[arm] = {'rp': float(np.nanmean(rp)), 'map': float(np.nanmean(ap)), 'per_query': rp.tolist()}
        if arm == 'bundle':
            rows[arm]['topk'] = topk_stats(test, scores)
    return {'fold': f, 'seed': seed, 'rows': rows, 'seconds': time.monotonic() - t}


ARMS = {}


def static_analyses(folds):
    from baseline.query_parser import Unparsed
    from baseline.query_parser_v3 import translate
    from baseline.query_struct import tree
    texts = query_texts(); out = {'per_atom_auc': {}, 'group_auc': {}, 'entry3_topk': {}, 'verifier_auc': {}, 'parse_truth': {}}
    relation = {}
    for d in folds.values():
        for q in d['dev']:
            try:
                relation[q] = tree(translate(texts[q]))[0] if d['parsed'].get(q) else 'unparsed'
            except (Unparsed, Exception):
                relation[q] = 'unparsed'
    out['relation_counts'] = dict(collections.Counter(relation.values()))
    for f, d in folds.items():
        use = usage(d); comb = combined(d); rows = {}
        for a in d['atoms']:
            y = d['truth_of'][a]
            rows[a] = {'uses': use[a], 'positives': int(y.sum()), 'auc_combined': auc(comb[a], y), 'auc_t4': auc(d['t4_logit'][a], y),
                       'auc_qwen': auc(d['qwen_logit'][a], y) if a in d['qwen_logit'] else None}
        out['per_atom_auc'][f] = rows
        for g, pick in GROUPS.items():
            vals = [(r['auc_combined'], r['uses']) for a, r in rows.items() if pick(a) and r['auc_combined'] is not None and r['uses']]
            out['group_auc'].setdefault(g, {})[f] = (float(np.average([v for v, _ in vals], weights=[w for _, w in vals])) if vals else None)
        # perception arms' resulting AUCs for the transformed atoms
        for name, tf in ARMS.items():
            if name.startswith(('shift', 'indep')) and '+' not in name:
                g = name.split(':')[1]; td = tf(d); c2 = combined(td)
                vals = [(auc(c2[a], d['truth_of'][a]), use[a]) for a in d['atoms'] if GROUPS[g](a) and use[a]]
                vals = [(v, w) for v, w in vals if v is not None]
                out['group_auc'].setdefault(name, {})[f] = float(np.average([v for v, _ in vals], weights=[w for _, w in vals]))
        out['entry3_topk'][f] = topk_stats(d, d['entry3'])
        cand = np.isfinite(d['verify']); top = np.zeros(cand.shape, bool)
        np.put_along_axis(top, np.argsort(-d['entry3'], 1, kind='stable')[:, :30], True, 1)
        out.setdefault('verifier_candidates_match_entry3_top30', {})[f] = float((cand & top).sum() / cand.sum())
        for name, v in (('current', d['verify']), *[(f'shift{x:g}', verifier('shift', x)(d)['verify']) for x in (0.25, 0.5, 1.0, 2.0, 4.0)]):
            per = [auc(v[i, cand[i]], d['rel'][i, cand[i]]) for i in range(len(d['dev']))]
            per = [x for x in per if x is not None]
            out['verifier_auc'].setdefault(name, {})[f] = {'mean_within_candidates': float(np.mean(per)), 'queries': len(per)}
        # pure parse-and-truth ceiling: AND of ground-truth atoms for each parsed query, raw similarity breaking ties
        tie = 1e-3 * zrow(d['raw_43']); score = np.array(d['raw_43'], float); parsed = np.zeros(len(d['dev']), bool); jac = []
        for i, q in enumerate(d['dev']):
            known = [a for a in (d['parsed'].get(q) or []) if a in d['truth_of']]
            if known:
                both = np.all([d['truth_of'][a] for a in known], 0); score[i] = both + tie[i]; parsed[i] = True
                r = d['rel'][i]
                if r.any():
                    jac.append((both & r).sum() / max((both | r).sum(), 1))
        rp, _, _, _ = per_query_rp(d, score)
        out['parse_truth'][f] = {'rp_parsed_queries': float(np.nanmean(rp[parsed])), 'parsed_queries': int(parsed.sum()),
                                 'queries': len(parsed), 'mean_jaccard_truth_and_vs_relevant': float(np.mean(jac))}
    return out, relation


def main():
    global FOLDS, ARMS
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('--workers', type=int, default=22); ap.add_argument('--quick', action='store_true')
    a = ap.parse_args(); RESULT.mkdir(parents=True, exist_ok=True); t0 = time.monotonic()
    FOLDS = base(); ARMS = arms(a.quick)
    static, relation = static_analyses(FOLDS)
    print('static analyses done', round(time.monotonic() - t0), 's', flush=True)
    jobs = [('bundle', 'reference', f, s) for f in FOLDS for s in SEEDS] + [(n, 'retrain', f, s) for n in ARMS for f in FOLDS for s in SEEDS]
    results = []
    with get_context('fork').Pool(a.workers) as pool:
        for r in pool.imap_unordered(run_job, jobs):
            results.append(r); print(f'{len(results)}/{len(jobs)} fold {r["fold"]} seed {r["seed"]} {list(r["rows"])[:1]} {r["seconds"]:.0f}s', flush=True)
    cells = collections.defaultdict(dict); per_query = collections.defaultdict(lambda: collections.defaultdict(list)); topk = {}
    for r in results:
        for arm, row in r['rows'].items():
            cells[arm][(r['fold'], r['seed'])] = (row['rp'], row['map'])
            for q, v in zip(FOLDS[r['fold']]['dev'], row['per_query']):
                per_query[arm][q].append(v)
            if 'topk' in row:
                topk[f'{r["fold"]}-{r["seed"]}'] = row['topk']
    ref = cells['bundle']; summary = {}
    for arm, c in cells.items():
        keys = sorted(c); d_rp = [c[k][0] - ref[k][0] for k in keys]; d_map = [c[k][1] - ref[k][1] for k in keys]
        summary[arm] = {'mean_rp': float(np.mean([c[k][0] for k in keys])), 'mean_map': float(np.mean([c[k][1] for k in keys])),
                        'rp_delta_vs_bundle': float(np.mean(d_rp)), 'map_delta_vs_bundle': float(np.mean(d_map)),
                        'per_fold_rp': {f: float(np.mean([c[k][0] for k in keys if k[0] == f])) for f in FOLDS}, 'cells': len(keys)}
    texts = query_texts(); by_relation = {}
    for arm in summary:
        groups = collections.defaultdict(list)
        for q, vals in per_query[arm].items():
            v = [x for x in vals if x is not None and np.isfinite(x)]
            if v:
                groups[relation[q]].append(np.mean(v)); groups['all'].append(np.mean(v))
        by_relation[arm] = {g: {'mean_rp': float(np.mean(v)), 'queries': len(v)} for g, v in groups.items()}
    stuck = []
    oracle_arm = 'oracle:all-atoms [retrain]'
    for q, vals in per_query.get(oracle_arm, {}).items():
        v = [x for x in vals if x is not None and np.isfinite(x)]
        if v and np.mean(v) < .25:
            parsed = next(d['parsed'].get(q) for d in FOLDS.values() if q in d['dev'])
            stuck.append({'query_id': q, 'text': texts.get(q), 'relation': relation[q], 'parsed_atoms': parsed,
                          'rp_perfect_atoms': float(np.mean(v)),
                          'rp_bundle': float(np.nanmean([x for x in per_query['bundle'][q] if x is not None]))})
    report = {'kind': 'entry-5 headroom diagnostics on the entry-4 bundle, development folds 0-2 x head seeds 43/44',
              'audit_used': False, 'gate': None, 'summary': dict(sorted(summary.items(), key=lambda kv: -kv[1]['mean_rp'])),
              'by_relation': by_relation, 'bundle_topk': topk, 'static': static, 'stuck_with_perfect_atoms': sorted(stuck, key=lambda x: x['rp_perfect_atoms']),
              'seconds': time.monotonic() - t0}
    (RESULT / ('report-quick.json' if a.quick else 'report.json')).write_text(json.dumps(report, indent=1, default=str))
    print(f"{'arm':48s} {'RP':>7s} {'dRP':>7s} {'dMAP':>7s}")
    for arm, s in report['summary'].items():
        print(f"{arm:48s} {s['mean_rp']:.4f} {s['rp_delta_vs_bundle']:+.4f} {s['map_delta_vs_bundle']:+.4f}")
    print('group AUC (combined, query-weighted):', json.dumps({g: {f: (round(v, 3) if v else v) for f, v in x.items()} for g, x in static['group_auc'].items()}))
    print('entry-3 top-K:', json.dumps(static['entry3_topk']['0']) if '0' in static['entry3_topk'] else json.dumps(static['entry3_topk'][0]))
    print('verifier AUC:', json.dumps(static['verifier_auc'])); print('parse truth:', json.dumps(static['parse_truth']))
    print('relations:', static['relation_counts']); print('stuck with perfect atoms:', len(stuck), 'queries; seconds', round(report['seconds']))


if __name__ == '__main__':
    main()
