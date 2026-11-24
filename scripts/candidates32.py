"""Verifier candidates from the 32-frame atom answers, as the entry-5 container will compute them (2026-10-23; CPU).

In the container the verifier's top-30 candidates come from the entry-3 recipe (parser v2, t4 + Qwen3.8 late fusion,
rank fusion 0.8, seed-43 head) using the atom answers it has, which in entry 5 are 32-frame answers. They overlap the
16-frame candidates by about 88%, so the training features must use the same rule. Writes, in the format of
runs/learned-scorer/verify-candidates.json:
  verify-candidates-32.json        development queries, folds 0-2
  verify-candidates-val-32.json    validation queries, folds 0-2
  verify-candidates-audit-32.json  audit queries on the audit clips (needs runs/verifier-variants/atoms-f32-faudit)
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.atoms_views_eval import view
from scripts.learned_scorer import OUT, combined, load, query_texts, summed


def top_k(d, k=30):
    from baseline.native_ranking import blended_scores
    s, m = summed(d, combined(d)); raw, sub = d['raw_43'], d['sub_43']
    e3 = np.where(m[:, None], blended_scores(np.where(m[:, None], s, raw), sub, .8), sub)
    per = {}
    for i, q in enumerate(d['dev']):
        for j in np.argsort(-e3[i], kind='stable')[:k]:
            per.setdefault(d['held'][j], []).append(q)
    return per


def main(which, k=30):
    texts = query_texts()
    specs = {'dev': ('verify-candidates-32.json', [(f, lambda f: load(f)) for f in (0, 1, 2)]),
             'val': ('verify-candidates-val-32.json', [(f, lambda f: load(f, OUT / 'val-v2')) for f in (0, 1, 2)]),
             'audit': ('verify-candidates-audit-32.json', [('audit', lambda f: load('audit'))])}
    name, folds = specs[which]; name = name if k == 30 else name.replace('.json', f'-{k}.json')
    out = {'rule': f'top-{k} per query under the entry-3 recipe with 32-frame Qwen3.8 answers (parser v2, seed-43 head)', 'k': k, 'folds': {}}
    qs = set()
    for f, loader in folds:
        d = view(loader(f), f, '32'); out['folds'][str(f)] = top_k(d, k); qs |= set(d['dev'])
    out['texts'] = {q: texts[q] for q in qs}
    (OUT / name).write_text(json.dumps(out)); print(name, {f: (len(v), sum(map(len, v.values()))) for f, v in out['folds'].items()})


if __name__ == '__main__':
    k = int(sys.argv[1]) if sys.argv[1].isdigit() else 30
    for w in [a for a in sys.argv[1:] if not a.isdigit()]:
        main(w, k)
