"""Released validation queries as extra development queries (2026-10-22; CPU; entry 5).

The 3,108 validation queries (cascade-v1.0/val) are labelled over the same 2,066 clips. They never fitted the 8B heads,
the learned scorer or any adapter (their texts did shape parsers v3/v4, so parse coverage on them is slightly flattered).
For each development fold this writes a learned-scorer cache in the same format as learned_scorer.save_state: the fold's
held clips with their out-of-fold t4 and Qwen3.8 atom answers (copied from the parser-v3 fold cache), and the validation
queries' raw and head similarities (heads fitted on the fold's fitting queries and clips, seeds 43/44), parses and qrels.

  cache        runs/learned-scorer/val/fold-{f}.npz (parser v3) and runs/learned-scorer/val-v2/fold-{f}.npz (parser v2)
  candidates   runs/learned-scorer/verify-candidates-val.json: top 30 held clips per validation query under the
               entry-3 recipe (parser v2, seed-43 head), the rule of the entry-4 verifier
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.learned_scorer import OUT, PLAN, combined, load, summed

VAL_VECTORS = ROOT / 'runs/finalist-canonical-2026-10-11-v2/canonical-refit/results/embed-8b/queries-val.npz'


def cache():
    from baseline.io import unique_rows
    from baseline.query_parser_v2 import translate as v2
    from baseline.query_parser_v3 import translate as v3
    from scripts.feature_retrieval import load_vectors
    from scripts.tracks_eval import common, parse, submitted_scores
    plan = json.loads(PLAN.read_text())
    val = unique_rows(ROOT / 'data/retrieval/cascade-v1.0/val/queries.jsonl', 'query_id'); val_ids = list(val)
    vrels = unique_rows(ROOT / 'data/retrieval/cascade-v1.0/val/qrels.jsonl', 'query_id')
    qtrain = load_vectors(ROOT / plan['feature_directory'] / 'queries-train.npz'); qval = load_vectors(VAL_VECTORS)
    vvec = load_vectors(ROOT / plan['feature_directory'] / 'aggregate.npz')
    if len(qval) != len(val_ids):
        raise ValueError('Validation vectors and queries differ in length')
    qall = np.concatenate([qtrain, qval]); texts = {q: val[q]['text'] for q in val_ids}
    parses = {'val': parse(texts, v3), 'val-v2': parse(texts, v2)}
    for f in (0, 1, 2):
        p = {**plan, 'held_fold': f}; _, _, qrels, _, vids, ix, _, _ = common(p)
        old = dict(np.load(OUT / 'p3' / f'fold-{f}.npz', allow_pickle=False))
        held = [str(v) for v in old['held']]
        if held != list(ix['held_video_ids']):
            raise ValueError('Held clips differ from the fold cache')
        ix2 = {**ix, 'dev_q': np.arange(len(qtrain), len(qall)), 'dev_query_ids': val_ids}
        subs = {s: submitted_scores(p, qall, vvec, ix2, qrels, s) for s in plan['head_seeds']}
        if f == 0:  # the heads must be the fold caches' heads: recompute the development queries' head scores
            dev_sub = submitted_scores(p, qall, vvec, ix, qrels, 43)[1]
            print('head parity on development queries, max abs diff', float(np.abs(dev_sub - old['sub_43']).max()), flush=True)
        pos = {v: j for j, v in enumerate(held)}; rel = np.zeros((len(val_ids), len(held)), bool)
        for i, q in enumerate(val_ids):
            for v in vrels[q]['relevant_video_ids']:
                if v in pos:
                    rel[i, pos[v]] = True
        clip_side = {k: old[k] for k in ('held', 'atoms', 't4', 'qwen_atoms', 'qwen', 'truth')}
        for name, parsed in parses.items():
            (OUT / name).mkdir(parents=True, exist_ok=True)
            np.savez_compressed(OUT / name / f'fold-{f}.npz', **clip_side, dev=np.asarray(val_ids), rel=rel, qvec=qval,
                                parsed=np.asarray(json.dumps(parsed)),
                                **{f'raw_{s}': subs[s][0] for s in subs}, **{f'sub_{s}': subs[s][1] for s in subs})
        print(f'fold {f}: {len(val_ids)} validation queries x {len(held)} clips; queries with relevant held clips {int(rel.any(1).sum())}', flush=True)
    print('parsed v3', sum(v is not None for v in parses['val'].values()), 'v2', sum(v is not None for v in parses['val-v2'].values()), 'of', len(val_ids))


def candidates(k=30):
    from baseline.io import unique_rows
    from baseline.native_ranking import blended_scores
    val = unique_rows(ROOT / 'data/retrieval/cascade-v1.0/val/queries.jsonl', 'query_id'); out = {'rule': f'top-{k} held clips per validation query under the entry-3 recipe (parser v2, seed-43 head)', 'k': k, 'folds': {}}
    for f in (0, 1, 2):
        d = load(f, OUT / 'val-v2'); s, m = summed(d, combined(d)); raw, sub = d['raw_43'], d['sub_43']
        e3 = np.where(m[:, None], blended_scores(np.where(m[:, None], s, raw), sub, .8), sub)
        top = np.argsort(-e3, 1, kind='stable')[:, :k]; per = {}
        for i, q in enumerate(d['dev']):
            for j in top[i]:
                per.setdefault(d['held'][j], []).append(q)
        out['folds'][str(f)] = per
    out['texts'] = {q: val[q]['text'] for q in val}
    (OUT / ('verify-candidates-val.json' if k == 30 else f'verify-candidates-val{k}.json')).write_text(json.dumps(out))
    print('candidates:', {f: (len(v), sum(map(len, v.values()))) for f, v in out['folds'].items()})


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('job', choices=['cache', 'candidates']); ap.add_argument('--k', type=int, default=30)
    a = ap.parse_args()
    cache() if a.job == 'cache' else candidates(a.k)
