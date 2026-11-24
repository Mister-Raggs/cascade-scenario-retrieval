"""Fit and save the entry-5 linear scorer (J1p, all-clip build) and prove runtime feature parity (2026-10-23).

Features (25): the entry-4 bundle's 23 with 32-frame Qwen3.8 atom answers, plus the Qwen3.8 32-frame property score and
flag, and the verifier of the chosen recipe (--arm J1p | J2-30 | J2-50; scripts/entry5_features.py) on candidates ranked
with the 32-frame answers (the container's rule). Training (head
seed 43, 10 epochs, seed 0): out-of-fold features of development folds 0-2 and the audit set with their development and
audit queries, plus a seeded 25% of the validation queries over folds 0-2 (scripts/scorer_mix_eval.py choice).
Parity: baseline/entry4_scorer.features (the container's code) equals the research pair features on every group.
Writes runs/entry5/scorer/{linear-scorer.npz, manifest.json} with the property list and the runtime guard settings.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from baseline.entry4_scorer import features
from baseline.io import sha256_file
from scripts.atoms_views_eval import view
from scripts.feature_retrieval import atomic_json
from scripts.learned_scorer import OUT, load, pair_features, query_texts, train_scorer, verify_matrix
from scripts.props_val import VV, with_props
from scripts.entry5_features import ARMS, group as arm_group
from scripts.val_eval import subset

SEED = 43
GUARD = {'atoms_switch_seconds': 3.0 * 3600, 'verifier_stop_seconds': 4.25 * 3600, 'fallback_frames': 16}


def runtime_features(d, f):
    from baseline.property_atoms import query_atoms as prop_atoms
    from baseline.query_parser import Unparsed
    from baseline.query_parser_v3 import translate as v3
    from baseline.query_struct import atoms, tree

    def parse(text):
        try:
            return sorted(set(atoms(tree(v3(text)))))
        except Unparsed:
            return None

    def prop_parse(text):
        try:
            return prop_atoms(v3(text))
        except Unparsed:
            return []
    z = np.load(VV / f'atoms-f32-f{f}/vlm-atom-probs.npz'); p = np.load(VV / f'props32-f{f}/props.npz')
    assert [str(v) for v in z['video_ids']] == d['held'] == [str(v) for v in p['video_ids']]
    vlm = {str(a): np.clip(z['probs'][:, j], 1e-4, 1 - 1e-4) for j, a in enumerate(z['atoms'])}
    t4 = {a: d['t4'][:, j] for j, a in enumerate(d['atoms'])}; props = {str(a): p['probs'][:, j] for j, a in enumerate(p['atoms'])}
    texts = query_texts()
    return features([texts[q] for q in d['dev']], t4, vlm, d[f'raw_{SEED}'], d[f'sub_{SEED}'], parse, d['verify'], props, prop_parse)[0]


def main():
    arm = sys.argv[sys.argv.index('--arm') + 1] if '--arm' in sys.argv else 'J1p'
    provisional = '--provisional' in sys.argv  # smoke tests only: no audit group (its answers may not exist yet)
    out = ROOT / ('runs/entry5/scorer-provisional' if provisional else 'runs/entry5/scorer'); out.mkdir(parents=True, exist_ok=True)
    dev = [arm_group(arm, 'dev', f) for f in (0, 1, 2)]; audit = None if provisional else arm_group(arm, 'audit', 'audit')
    val = [arm_group(arm, 'val', f) for f in (0, 1, 2)]; rng = np.random.default_rng(2026)
    val25 = []
    for d in val:
        m = rng.random(len(d['dev'])) < .25
        val25.append({**subset(d, m), 'prop_score': d['prop_score'][m], 'prop_mask': d['prop_mask'][m]})
    parity = {}
    for d, f in [*zip(dev, (0, 1, 2)), *([(audit, 'audit')] if audit else []), (val[0], 0)]:
        parity[d['group']] = float(np.abs(runtime_features(d, f) - pair_features(d, SEED)[0]).max())
    print('parity', parity, flush=True)
    predict = train_scorer([*dev, *([audit] if audit else []), *val25], SEED, False, 10, linear=True)
    layer = predict.model; weight = layer.weight.detach().numpy().reshape(-1).astype(np.float32); bias = float(layer.bias.detach().numpy()[0])
    np.savez(out / 'linear-scorer.npz', weight=weight, bias=np.asarray(bias, np.float32))
    properties = json.loads((OUT / 'props-all-atoms.json').read_text())
    manifest = {'kind': f'entry-5 linear learned scorer ({arm}, all-clip build)', 'arm': arm, 'verifier_k': ARMS[arm][1],
                'verifier': 'fine-tuned multi-task adapter' if ARMS[arm][0].startswith('ft') else 'zero-shot (atom adapter)', 'weights_sha256': sha256_file(out / 'linear-scorer.npz'),
                'dimension': len(weight), 'properties': properties, 'guard': GUARD, 'question_batch': 8,
                'training': ('linear arm, head seed 43, 10 epochs, seed 0; out-of-fold features of development folds 0-2 and the audit set '
                             '(their development and audit queries) plus a seeded 25% of the validation queries over folds 0-2; 32-frame '
                             f'atoms, Qwen3.8 32-frame property answers, {arm} verifier on 32-frame-ranked candidates'),
                'training_queries': int(sum(len(d['dev']) for d in [*dev, *([audit] if audit else []), *val25])), 'provisional': provisional,
                'feature_parity_max_abs_diff': parity, 'passed': max(parity.values()) < 1e-5,
                'confirmation': 'see runs/learned-scorer/report-audit-j1p.json and runs/entry5-reports'}
    atomic_json(out / 'manifest.json', manifest); print(json.dumps({k: v for k, v in manifest.items() if k != 'properties'}, indent=1))
    if not manifest['passed']:
        raise SystemExit('Entry-5 scorer feature parity failed')


if __name__ == '__main__':
    main()
