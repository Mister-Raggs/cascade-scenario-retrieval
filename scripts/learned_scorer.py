"""Learned query-clip scorer on out-of-fold features (2026-10-22; development folds only, CPU).

Rules fixed before any result. Cross-fitted: the scorer for fold f is trained on the other two folds' dev queries and
held clips and evaluated on fold f. Every feature is out of fold: each fold's Qwen3.8 answers come from the adapter that
did not train on that fold (runs/tracks-v8 q38-score-f0, runs/tracks-v9 q38-score-f1/f2) and its t4 probabilities are
the held-out ones of scripts/perception_v3_eval.py fold_state. Head seeds 43/44 give the submitted ranking.

Arms (all at fusion weight 0.8 where a structured score is rank-fused with the submitted ranking):
  current    t4 + Qwen3.8 late fusion, parser-v2 atom log-probabilities summed (the entry-3 recipe; must reproduce)
  calib      per-family (agent/light/obj/ego) temperature and bias on Qwen3.8 log-odds and on t4 log-odds, fitted on the
             training folds' atom labels by logistic loss, then the current recipe
  mlp        a learned pair scorer replacing the summed atom score: per-atom-slot combined/Qwen/t4 log-odds (two slots,
             atoms sorted), their sum, atom count and family flags, raw and head similarities and the parsed flag, each
             similarity and score standardized within the query; 2-layer MLP (64 hidden), listwise softmax loss
  bilinear   mlp plus query-text x answers: 8B query vector (PCA 64, fitted on training queries) times the clip's 95
             Qwen3.8 log-odds through a rank-8 bilinear map
Learned scores are used directly as the final ranking for every query (they include the submitted similarities).
Fixed hyperparameters: Adam lr 1e-3, weight decay 1e-4, 30 epochs, 32 queries per batch, seed 0.

Revision 2 (2026-10-22, after the first run, reported as such): the first run had two problems. The calibration fit
diverged on nearly separable families (Qwen light scale 2e8), and the 30-epoch MLP overfitted its training clip pools
(training-fold RP 0.42, held 0.235). Diagnostics then looked at fold 1, seed 43 for four variants, so that cell is mildly
contaminated. Revised arms, fixed before the full rerun:
  calib      the same fit with a ridge penalty (lambda 1, towards scale 1 and bias 0) and scale capped at 10
  linear     all pair features, one linear layer, Adam lr 1e-2, 10 epochs (fixed before its diagnostic was seen)
  mlp        epochs chosen inside the training folds only, from {2, 5, 10, 30}: train on one training fold, validate on
             the other, both directions, best mean RP; then retrain on both training folds
  bilinear   mlp's selection procedure, with the bilinear term
The gate below is unchanged.
Gate per arm versus current: mean RP gain > 0.001 over the 6 fold-seed cells, at least 4/6 improve, no fold regresses on
average, mean MAP change >= 0.
Revision 3 (2026-10-22, job eval2; rules fixed before its results):
  calib2         per-family scale and bias on Qwen3.8 and t4 log-odds, ridge 1e-3 x points towards (1, 0), scale in
                 [0.05, 3], fitted on the training folds; then the current recipe. Compared against current.
  linear+calib2  linear with calib2 log-odds as its atom features. Compared against linear.
  linear-fam     linear plus per-family copies of the slot features (2 slots x 4 families x 3 log-odds). Compared
                 against linear.
Same gate. These are further looks at folds 0-2; a pass needs audit confirmation before an entry.
Audit confirmation (2026-10-22, job audit; rule fixed before any audit number; third look at the audit set): the linear
arm trained on all three folds' out-of-fold features (dev queries x held clips, 10 epochs, per head seed) scores the 773
audit queries x 517 audit clips with the audit run's Qwen3.8 answers (runs/audit-v2, the all-development adapter) and the
audit-mode t4 probabilities and heads. Confirmed if its mean RP gain over current on the audit set is > +0.001, both head
seeds improve and the mean MAP change is >= 0.
Revision 4 (2026-10-22, jobs cache-p3 and eval3; rules fixed before their results; reference is linear, the
audit-confirmed scorer; passes go to a later entry, not entry 4):
  linear-p3    folds re-cached with parser v3 (baseline/query_parser_v3.py, coverage-only, written from validation texts)
               in place of v2; the new atoms get t4 probabilities and the ego CNN's atom set may grow
  linear-rank  linear plus reciprocal-rank features (1/(60+rank) of raw, head and summed scores within the query) and
               per-clip hubness (the clip's mean within-query z of raw and head similarity over the set's queries)
  linear-bag5  mean of five linear scorers trained with seeds 0-4
  diagnostics  per-query RP gain of linear over current by query type (no gate)
Same gate, each versus linear.
Mirror (2026-10-22, job eval-mirror; rule in scripts/gpu_batch_2026_10_22.py, fixed before results): each fold's
Qwen3.8 answers become the mean log-odds of the original and mirrored views, then linear versus linear, standing gate.
Verifier (2026-10-22, job eval-verify; rule in scripts/gpu_batch_2026_10_22.py, fixed before results): two extra
features, the verifier log-odds z-scored within each query's top-30 candidates (others at the minimum minus 1) and an
in-candidates flag; linear-verify versus linear, standing gate.
Revision 5 (2026-10-22, jobs cache-p4 and eval4; rules fixed before results; reference linear-p3):
  linear-p4       folds re-cached with parser v4 (v3 plus time-limited sequences, written from validation texts)
  linear-p3-only  on the v3 cache, one extra feature for queries containing "only" or "nothing but": the sum of
                  log(1 - p) over the ego atoms the query does not mention, standardized within the query (0 otherwise)
Same gate, each versus linear-p3.
Ego pilot (2026-10-22, job eval-ego-pilot; rule in scripts/gpu_batch_2026_10_22.py): on fold 0, the ego specialist's
answers replace Qwen3.8's ego answers in the entry-3 recipe; pass if RP improves by >= +0.005 on both head seeds.
Revision 6 (2026-10-22, job eval-verify2; rules fixed before results; reference linear-verify, the zero-shot top-30
verifier on the v2 cache):
  linear-p3-verify    parser v3 cache plus the same verifier features (does the parser gain add to the verifier gain?)
  linear-verify50     the zero-shot verifier on each query's top-50 candidates (runs/gpu-2026-10-22/verify50-f*)
  linear-verifyft     a fine-tuned verifier's answers in place of the zero-shot ones (runs/gpu-2026-10-22/verifyft-f*),
                      evaluated when present
Same gate, each versus linear-verify.
Sweep pilots (2026-10-22, job eval-sweep-pilot --name N): as the ego pilot, with the sweep point's answers for all atoms.
Property atoms (2026-10-22, jobs prop-probs and eval-prop; rules fixed before results): per fold, t4-style late-fused
linear probes (pooled, motion2, windows) for the property atoms of baseline/property_atoms.py, fitted on the other folds'
clips and predicting the held clips; each query's summed property-atom log-probability (z within the query, 0 without
property atoms) and a has-property flag become two linear-scorer features. linear-prop versus linear, standing gate.
Property atoms from Qwen3.8 (job eval-prop-qwen; rule fixed before results): the same two features with Qwen3.8's
zero-shot property answers (runs/gpu-2026-10-22/props-f*) in place of the t4 probes; linear-prop-qwen versus linear.
Bundle audit (2026-10-22, jobs audit-prep and audit-bundle; rule fixed before any audit number; fourth look at the audit
set): the linear scorer with parser v3 and the zero-shot top-30 verifier features, trained on the three development folds
(p3 caches + runs/gpu-2026-10-22/verify-f*), scores the 773 audit queries x 517 audit clips with the audit run's Qwen3.8
adapter answering both the atom and the verifier questions (verifier candidates: top 30 per audit query under the entry-3
recipe, seed-43 head, from the v2 audit cache). Confirmed if the mean RP gain over the audited linear scorer is > +0.001,
both head seeds improve and the mean MAP change is >= 0.
Usage: learned_scorer.py cache | cache-p3 | cache-p4 | eval | eval2 | eval3 | eval4 | eval-mirror | eval-verify | eval-verify2 | eval-ego-pilot | audit   (cache writes runs/learned-scorer/fold-{0,1,2}.npz)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PLAN = ROOT / 'configs/audit-q38/evaluator.json'
OUT = ROOT / 'runs/learned-scorer'
QWEN = {0: ('runs/tracks-v8', 'q38-score-f0'), 1: ('runs/tracks-v9', 'q38-score-f1'), 2: ('runs/tracks-v9', 'q38-score-f2')}
FAMILIES = ('agent', 'light', 'obj', 'ego')
WEIGHT = 0.8


def cache(out=None, parser=None):
    from scripts.feature_retrieval import load_vectors
    from scripts.perception_v3_eval import fold_state, with_ego
    from scripts.post_submission import verified_result
    import scripts.perception_v3_eval as pe
    if parser is not None:
        pe.translate_v2 = parser  # fold_state's scoring parse (st['p2']) becomes this parser
    out = out or OUT; plan = json.loads(PLAN.read_text()); out.mkdir(parents=True, exist_ok=True)
    qvec = load_vectors(ROOT / plan['feature_directory'] / 'queries-train.npz')
    for f in (0, 1, 2):
        t = time.monotonic(); st = fold_state(plan, f, None)
        q = np.load(verified_result(ROOT / QWEN[f][0], QWEN[f][1], 'vlm-atom-probs.npz'))
        save_state(st, q, qvec, out / f'fold-{f}.npz')
        print(f'fold {f}: {len(st["ix"]["dev_query_ids"])} queries x {len(st["ix"]["held_video_ids"])} clips, {round(time.monotonic() - t)} s', flush=True)


def save_state(st, q, qvec, path):
    from scripts.perception_v3_eval import with_ego
    ix = st['ix']; held, dev = list(ix['held_video_ids']), list(ix['dev_query_ids'])
    if list(q['video_ids']) != held:
        raise ValueError('Qwen3.8 answers not aligned with the held clips')
    t4 = with_ego(st, 'm2'); atoms = sorted(t4)
    rel = np.zeros((len(dev), len(held)), bool); pos = {v: i for i, v in enumerate(held)}
    for i, qid in enumerate(dev):
        for v in st['qrels'][qid]['relevant_video_ids']:
            if v in pos:
                rel[i, pos[v]] = True
    held_i = np.asarray(ix['held_v'])
    truth = np.stack([st['truth'][a][held_i] for a in atoms], 1)
    parsed = json.dumps({qid: st['p2'].get(qid) for qid in dev})
    np.savez_compressed(path, held=np.asarray(held), dev=np.asarray(dev), atoms=np.asarray(atoms),
                        t4=np.stack([t4[a] for a in atoms], 1), qwen_atoms=q['atoms'], qwen=q['probs'], rel=rel, truth=truth,
                        qvec=qvec[ix['dev_q']], parsed=np.asarray(parsed),
                        **{f'raw_{s}': st['subs'][s][0] for s in st['subs']}, **{f'sub_{s}': st['subs'][s][1] for s in st['subs']})


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def sigmoid(z):
    return np.clip(1 / (1 + np.exp(-z)), 1e-4, 1 - 1e-4)


def load(f, root=None):
    d = dict(np.load((root or OUT) / (f'fold-{f}.npz' if isinstance(f, int) else f'{f}.npz'), allow_pickle=False))
    d['parsed'] = json.loads(str(d['parsed'])); d['atoms'] = [str(a) for a in d['atoms']]
    d['qwen_atoms'] = [str(a) for a in d['qwen_atoms']]; d['dev'] = [str(x) for x in d['dev']]; d['held'] = [str(x) for x in d['held']]
    d['t4_logit'] = {a: logit(d['t4'][:, j]) for j, a in enumerate(d['atoms'])}
    d['qwen_logit'] = {a: logit(d['qwen'][:, j]) for j, a in enumerate(d['qwen_atoms'])}
    d['truth_of'] = {a: d['truth'][:, j] for j, a in enumerate(d['atoms'])}
    d['only'] = np.asarray([bool(ONLY.search(query_texts().get(q, '').lower())) for q in d['dev']])
    return d


ONLY = __import__('re').compile(r'\bonly\b|\bnothing but\b')
_TEXTS = {}


def query_texts():
    if not _TEXTS:
        from baseline.io import unique_rows
        for split in ('train', 'val'):
            _TEXTS.update({q: r['text'] for q, r in unique_rows(ROOT / f'data/retrieval/cascade-v1.0/{split}/queries.jsonl', 'query_id').items()})
    return _TEXTS


def family(a):
    return a.split('|')[0]


def combined(d, cal=None):
    """Atom -> held log-odds of the current late fusion (mean of t4 and Qwen3.8 log-odds), optionally calibrated."""
    out = {}
    for a, t in d['t4_logit'].items():
        if cal:
            t = cal['t4'][family(a)][0] * t + cal['t4'][family(a)][1]
        if a in d['qwen_logit']:
            q = d['qwen_logit'][a]
            if cal:
                q = cal['qwen'][family(a)][0] * q + cal['qwen'][family(a)][1]
            t = logit(sigmoid((t + q) / 2))
        out[a] = logit(sigmoid(t))
    return out


def summed(d, comb):
    s = np.zeros_like(d['raw_43']); m = np.zeros(len(d['dev']), bool)
    for i, q in enumerate(d['dev']):
        known = [a for a in (d['parsed'].get(q) or []) if a in comb]
        if known:
            s[i] = np.sum([np.log(sigmoid(comb[a])) for a in known], axis=0); m[i] = True
    return s, m


def metrics(d, scores):
    from baseline.hubness import ranking_metrics
    qrels = {q: {'relevant_video_ids': [d['held'][j] for j in np.flatnonzero(d['rel'][i])]} for i, q in enumerate(d['dev'])}
    m = ranking_metrics(scores, d['dev'], d['held'], qrels)
    return m['r_precision'], m['map']


def fit_platt(x, y, ridge=1.0, bounds=(0.0, 10.0)):
    """1-D logistic fit (Newton) of y on a*x + b, ridge-penalized towards (1, 0) so separable data cannot diverge."""
    theta = np.array([1.0, 0.0]); prior = theta.copy()
    for _ in range(50):
        p = sigmoid(theta[0] * x + theta[1]); g = np.array([np.sum((p - y) * x), np.sum(p - y)]) + ridge * (theta - prior)
        w = p * (1 - p); h = np.array([[np.sum(w * x * x), np.sum(w * x)], [np.sum(w * x), np.sum(w)]]) + ridge * np.eye(2)
        theta = theta - np.linalg.solve(h, g); theta[0] = np.clip(theta[0], *bounds)
    return float(theta[0]), float(theta[1])


def calibration(train, scaled=False):
    cal = {'t4': {}, 'qwen': {}}
    for kind in cal:
        for fam in FAMILIES:
            xs, ys = [], []
            for d in train:
                src = d['t4_logit'] if kind == 't4' else d['qwen_logit']
                for a, x in src.items():
                    if family(a) == fam and a in d['truth_of']:
                        xs.append(x); ys.append(d['truth_of'][a].astype(float))
            x, y = (np.concatenate(xs), np.concatenate(ys)) if xs else (None, None)
            cal[kind][fam] = ((fit_platt(x, y, 1e-3 * len(x), (0.05, 3.0)) if scaled else fit_platt(x, y)) if xs else (1.0, 0.0))
    return cal


def zrow(x):
    return (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-6)


def pair_features(d, seed, cal=None, per_family=False, ranks=False, only=False):
    """[queries, clips, features] for the learned scorer, plus the Qwen answer matrix for the bilinear arm."""
    comb = combined(d, cal); s, m = summed(d, comb); n_q, n_v = s.shape
    slot = np.zeros((n_q, n_v, 2, 4 + len(FAMILIES)), np.float32); count = np.zeros(n_q, np.float32)
    for i, q in enumerate(d['dev']):
        known = sorted(a for a in (d['parsed'].get(q) or []) if a in comb)[:2]; count[i] = len(known)
        for k, a in enumerate(known):
            fam = family(a); t4l = d['t4_logit'][a]
            slot[i, :, k, 0] = comb[a]; slot[i, :, k, 1] = cal['t4'][fam][0] * t4l + cal['t4'][fam][1] if cal else t4l
            if a in d['qwen_logit']:
                ql = d['qwen_logit'][a]
                slot[i, :, k, 2] = cal['qwen'][fam][0] * ql + cal['qwen'][fam][1] if cal else ql; slot[i, :, k, 3] = 1
            slot[i, :, k, 4 + FAMILIES.index(family(a))] = 1
    raw, sub = d[f'raw_{seed}'], d[f'sub_{seed}']
    per_q = np.stack([zrow(raw), zrow(sub), np.where(m[:, None], zrow(np.where(m[:, None], s, 0)), 0),
                      np.broadcast_to(m[:, None], s.shape).astype(np.float32),
                      np.broadcast_to(count[:, None], s.shape) / 2], -1)
    parts = [slot.reshape(n_q, n_v, -1) / 5, per_q]
    if per_family:  # slot log-odds (combined, t4, Qwen) copied into the slot's family
        parts.append((slot[..., :3, None] * slot[..., None, 4:]).reshape(n_q, n_v, -1) / 5)
    if only:  # exclusivity: evidence against the ego actions an "only" query does not mention
        ex = np.zeros(s.shape, np.float32)
        for i, q in enumerate(d['dev']):
            mentioned = set(d['parsed'].get(q) or [])
            others = [a for a in comb if a.startswith('ego') and a not in mentioned]
            if d['only'][i] and mentioned and others:
                ex[i] = zrow(np.sum([np.log(1 - sigmoid(comb[a])) for a in others], 0)[None])[0]
        parts.append(ex[..., None])
    if 'prop_score' in d:  # property atoms (signal, slow, double-parked, light state and flags)
        pm = d['prop_mask']
        parts.append(np.stack([np.where(pm[:, None], zrow(np.where(pm[:, None], d['prop_score'], 0)), 0),
                               np.broadcast_to(pm[:, None], s.shape).astype(np.float32)], -1))
    if 'verify' in d:  # whole-query verifier on the top-30 candidates
        v = d['verify']; cand = np.isfinite(v); mu = np.nanmean(np.where(cand, v, np.nan), 1, keepdims=True)
        sd = np.nanstd(np.where(cand, v, np.nan), 1, keepdims=True) + 1e-6; z = np.where(cand, (v - mu) / sd, np.nan)
        lowest = np.nanmin(np.where(cand, z, np.nan), 1, keepdims=True) - 1
        parts.append(np.stack([np.where(cand, z, lowest), cand.astype(np.float32)], -1))
    if ranks:  # reciprocal ranks within the query, and per-clip hubness over the set's queries
        rr = lambda v: 60 / (60 + np.argsort(np.argsort(-v, 1), 1) + 1)
        zr, zs = zrow(raw), zrow(sub)
        parts.append(np.stack([rr(raw), rr(sub), np.where(m[:, None], rr(np.where(m[:, None], s, 0)), 0),
                               np.broadcast_to(zr.mean(0)[None], s.shape), np.broadcast_to(zs.mean(0)[None], s.shape)], -1))
    x = np.concatenate(parts, -1).astype(np.float32)
    answers = logit(d['qwen']).astype(np.float32)
    return x, (answers - answers.mean(0)) / (answers.std(0) + 1e-6)


def train_scorer(train, seed, bilinear, epochs=30, linear=False, cal=None, per_family=False, ranks=False, init=0, only=False):
    import torch
    torch.manual_seed(init); rng = np.random.default_rng(init)
    data = [(*pair_features(d, seed, cal, per_family, ranks, only), d['qvec'], d['rel']) for d in train]
    qall = np.concatenate([q for _, _, q, _ in data]); mean = qall.mean(0)
    _, _, vt = np.linalg.svd(qall - mean, full_matrices=False); pca = vt[:64].T.astype(np.float32)
    dim = data[0][0].shape[-1]
    mlp = (torch.nn.Linear(dim, 1) if linear else
           torch.nn.Sequential(torch.nn.Linear(dim, 64), torch.nn.ReLU(), torch.nn.Linear(64, 64), torch.nn.ReLU(), torch.nn.Linear(64, 1)))
    w1 = torch.nn.Parameter(torch.zeros(64, 8)); w2 = torch.nn.Parameter(torch.randn(95, 8) * .01)
    params = list(mlp.parameters()) + ([w1, w2] if bilinear else [])
    opt = torch.optim.Adam(params, lr=1e-2 if linear else 1e-3, weight_decay=1e-4)

    def score(x, answers, qv):
        out = mlp(x).squeeze(-1)
        if bilinear:
            out = out + ((qv @ w1) @ (answers @ w2).T)
        return out
    rows = [(k, i) for k, (_, _, _, r) in enumerate(data) for i in range(len(r)) if r[i].any()]
    tensors = [(torch.as_tensor(x), torch.as_tensor(a), torch.as_tensor((q - mean) @ pca), torch.as_tensor(r, dtype=torch.float32)) for x, a, q, r in data]
    for _ in range(epochs):
        order = rng.permutation(len(rows))
        for start in range(0, len(order), 32):
            loss = 0
            for j in order[start:start + 32]:
                k, i = rows[j]; x, a, q, r = tensors[k]
                logp = torch.log_softmax(score(x[i:i + 1], a, q[i:i + 1])[0], 0)
                loss = loss - (r[i] / r[i].sum() * logp).sum()
            opt.zero_grad(); (loss / 32).backward(); opt.step()

    def predict(d):
        x, a = pair_features(d, seed, cal, per_family, ranks, only)
        with torch.no_grad():
            return score(torch.as_tensor(x), torch.as_tensor(a), torch.as_tensor((d['qvec'] - mean) @ pca)).numpy()
    predict.model = mlp  # the trained layer(s), for packaging
    return predict


def select_epochs(train, seed, bilinear, grid=(2, 5, 10, 30)):
    """Epochs chosen inside the training folds only: fit on one, validate on the other, both directions."""
    a, b = train; score = {}
    for e in grid:
        score[e] = np.mean([metrics(v, train_scorer([t], seed, bilinear, e)(v))[0] for t, v in ((a, b), (b, a))])
    return max(score, key=score.get)


def evaluate():
    from baseline.native_ranking import blended_scores
    folds = {f: load(f) for f in (0, 1, 2)}; rows = []; chosen = {}
    fused = lambda raw, sub, s, m: np.where(m[:, None], blended_scores(np.where(m[:, None], s, raw), sub, WEIGHT), sub)
    for f, d in folds.items():
        train = [folds[g] for g in folds if g != f]; cal = calibration(train)
        for seed in (43, 44):
            raw, sub = d[f'raw_{seed}'], d[f'sub_{seed}']
            arms = {'current': fused(raw, sub, *summed(d, combined(d))), 'calib': fused(raw, sub, *summed(d, combined(d, cal)))}
            arms['linear'] = train_scorer(train, seed, False, 10, linear=True)(d)
            for name, bil in (('mlp', False), ('bilinear', True)):
                t = time.monotonic(); epochs = select_epochs(train, seed, bil); chosen[(f, seed, name)] = epochs
                arms[name] = train_scorer(train, seed, bil, epochs)(d)
                print(f'fold {f} seed {seed} {name}: {epochs} epochs, {round(time.monotonic() - t)} s', flush=True)
            for name, scores in arms.items():
                rp, ap = metrics(d, scores); rows.append({'fold': f, 'seed': seed, 'arm': name, 'r_precision': rp, 'map': ap})
            print(f, seed, {r['arm']: round(r['r_precision'], 4) for r in rows if r['fold'] == f and r['seed'] == seed}, flush=True)
    from scripts.perception_v3_eval import compare, summarise
    from scripts.feature_retrieval import atomic_json
    gate = lambda c: bool(c['mean_rp_delta'] > .001 and c['cells_improved'] >= 4 and min(c['per_fold_rp_delta'].values()) >= 0 and c['mean_map_delta'] >= 0)
    result = {}
    for arm in ('calib', 'linear', 'mlp', 'bilinear'):
        c = compare(rows, arm, 'current'); c['passes'] = gate(c); result[arm] = c
    report = {'kind': 'learned query-clip scorer on out-of-fold features, development folds 0-2', 'summary': summarise(rows),
              'versus_current': result, 'epochs_chosen': {f'fold{f}-seed{s}-{n}': e for (f, s, n), e in chosen.items()},
              'calibration_note': 'fitted on training folds per held fold', 'rule': __doc__.split('Gate per arm')[1].split('Usage')[0].strip(), 'audit_used': False}
    atomic_json(OUT / 'report.json', report); atomic_json(OUT / 'cells.json', rows); print(json.dumps(result, indent=1))


def evaluate2():
    from baseline.native_ranking import blended_scores
    from scripts.feature_retrieval import atomic_json
    from scripts.perception_v3_eval import compare, summarise
    folds = {f: load(f) for f in (0, 1, 2)}; rows = []; params = {}
    fused = lambda raw, sub, s, m: np.where(m[:, None], blended_scores(np.where(m[:, None], s, raw), sub, WEIGHT), sub)
    for f, d in folds.items():
        train = [folds[g] for g in folds if g != f]; cal = calibration(train, scaled=True); params[f] = cal
        for seed in (43, 44):
            raw, sub = d[f'raw_{seed}'], d[f'sub_{seed}']
            arms = {'current': fused(raw, sub, *summed(d, combined(d))), 'calib2': fused(raw, sub, *summed(d, combined(d, cal))),
                    'linear': train_scorer(train, seed, False, 10, linear=True)(d),
                    'linear+calib2': train_scorer(train, seed, False, 10, linear=True, cal=cal)(d),
                    'linear-fam': train_scorer(train, seed, False, 10, linear=True, per_family=True)(d)}
            for name, scores in arms.items():
                rp, ap = metrics(d, scores); rows.append({'fold': f, 'seed': seed, 'arm': name, 'r_precision': rp, 'map': ap})
            print(f, seed, {r['arm']: round(r['r_precision'], 4) for r in rows if r['fold'] == f and r['seed'] == seed}, flush=True)
    gate = lambda c: bool(c['mean_rp_delta'] > .001 and c['cells_improved'] >= 4 and min(c['per_fold_rp_delta'].values()) >= 0 and c['mean_map_delta'] >= 0)
    result = {}
    for arm, base in (('calib2', 'current'), ('linear', 'current'), ('linear+calib2', 'linear'), ('linear-fam', 'linear')):
        c = compare(rows, arm, base); c['versus'] = base; c['passes'] = gate(c); result[arm] = c
    report = {'kind': 'learned scorer revision 3 (calibration and per-family linear), development folds 0-2', 'summary': summarise(rows),
              'comparisons': result, 'calibration': {str(f): c for f, c in params.items()}, 'audit_used': False}
    atomic_json(OUT / 'report-eval2.json', report); atomic_json(OUT / 'cells-eval2.json', rows); print(json.dumps(result, indent=1))


def evaluate3():
    import re
    from baseline.native_ranking import blended_scores
    from scripts.feature_retrieval import atomic_json
    from scripts.perception_v3_eval import compare, summarise
    folds = {f: load(f) for f in (0, 1, 2)}; folds3 = {f: load(f, OUT / 'p3') for f in (0, 1, 2)}; rows = []; per_query = []
    from baseline.io import unique_rows
    texts = {q: r['text'] for q, r in unique_rows(ROOT / 'data/retrieval/cascade-v1.0/train/queries.jsonl', 'query_id').items()}
    fused = lambda raw, sub, s, m: np.where(m[:, None], blended_scores(np.where(m[:, None], s, raw), sub, WEIGHT), sub)

    def rp_each(d, scores):
        order = np.argsort(-scores, 1, kind='stable'); hits = np.take_along_axis(d['rel'], order, 1); n = d['rel'].sum(1)
        return np.array([hits[i, :n[i]].mean() if n[i] else np.nan for i in range(len(n))])
    for f, d in folds.items():
        train = [folds[g] for g in folds if g != f]; train3 = [folds3[g] for g in folds3 if g != f]
        for seed in (43, 44):
            raw, sub = d[f'raw_{seed}'], d[f'sub_{seed}']
            arms = {'current': fused(raw, sub, *summed(d, combined(d))), 'linear': train_scorer(train, seed, False, 10, linear=True)(d),
                    'linear-rank': train_scorer(train, seed, False, 10, linear=True, ranks=True)(d),
                    'linear-bag5': np.mean([train_scorer(train, seed, False, 10, linear=True, init=k)(d) for k in range(5)], 0)}
            p3 = train_scorer(train3, seed, False, 10, linear=True)(folds3[f])
            for name, scores in arms.items():
                rp, ap = metrics(d, scores); rows.append({'fold': f, 'seed': seed, 'arm': name, 'r_precision': rp, 'map': ap})
            rp, ap = metrics(folds3[f], p3); rows.append({'fold': f, 'seed': seed, 'arm': 'linear-p3', 'r_precision': rp, 'map': ap})
            gain = rp_each(d, arms['linear']) - rp_each(d, arms['current'])
            for i, q in enumerate(d['dev']):
                atoms = d['parsed'].get(q) or []
                per_query.append({'fold': f, 'seed': seed, 'query': q, 'gain': None if np.isnan(gain[i]) else float(gain[i]),
                                  'parsed': bool(atoms), 'families': sorted({family(a) for a in atoms}), 'atoms': len(atoms),
                                  'relevant': int(d['rel'][i].sum()), 'relation': next((w for w in ('because', 'since', 'so', 'after', 'before', 'while', 'then', 'consequently', 'leading', 'causing') if re.search(r'\b' + w + r'\b', texts.get(q, '').lower())), 'none')})
            print(f, seed, {r['arm']: round(r['r_precision'], 4) for r in rows if r['fold'] == f and r['seed'] == seed}, flush=True)
    gate = lambda c: bool(c['mean_rp_delta'] > .001 and c['cells_improved'] >= 4 and min(c['per_fold_rp_delta'].values()) >= 0 and c['mean_map_delta'] >= 0)
    result = {}
    for arm in ('linear-p3', 'linear-rank', 'linear-bag5'):
        c = compare(rows, arm, 'linear'); c['passes'] = gate(c); result[arm] = c
    groups = {}
    for key in ('parsed', 'families', 'atoms', 'relation', 'relevant'):
        agg = {}
        for r in per_query:
            if r['gain'] is None:
                continue
            k = str(r[key]) if key != 'relevant' else ('1' if r['relevant'] == 1 else '2-3' if r['relevant'] <= 3 else '4+')
            agg.setdefault(k, []).append(r['gain'])
        groups[key] = {k: {'queries_x_seeds': len(v), 'mean_gain': float(np.mean(v))} for k, v in sorted(agg.items())}
    report = {'kind': 'learned scorer revision 4 (parser v3, rank/hubness features, bagging) and diagnostics, folds 0-2',
              'summary': summarise(rows), 'versus_linear': result, 'diagnostics_linear_minus_current': groups, 'audit_used': False}
    atomic_json(OUT / 'report-eval3.json', report); atomic_json(OUT / 'cells-eval3.json', rows); print(json.dumps(result, indent=1))


def evaluate4():
    from scripts.feature_retrieval import atomic_json
    from scripts.perception_v3_eval import compare, summarise
    p3 = {f: load(f, OUT / 'p3') for f in (0, 1, 2)}; p4 = {f: load(f, OUT / 'p4') for f in (0, 1, 2)}; rows = []
    for f in (0, 1, 2):
        for seed in (43, 44):
            for name, folds, only in (('linear-p3', p3, False), ('linear-p3-only', p3, True), ('linear-p4', p4, False)):
                scores = train_scorer([folds[g] for g in folds if g != f], seed, False, 10, linear=True, only=only)(folds[f])
                rp, ap = metrics(folds[f], scores); rows.append({'fold': f, 'seed': seed, 'arm': name, 'r_precision': rp, 'map': ap})
            print(f, seed, {r['arm']: round(r['r_precision'], 4) for r in rows if r['fold'] == f and r['seed'] == seed}, flush=True)
    gate = lambda c: bool(c['mean_rp_delta'] > .001 and c['cells_improved'] >= 4 and min(c['per_fold_rp_delta'].values()) >= 0 and c['mean_map_delta'] >= 0)
    result = {}
    for arm in ('linear-p3-only', 'linear-p4'):
        c = compare(rows, arm, 'linear-p3'); c['passes'] = gate(c); result[arm] = c
    only_queries = int(sum(p3[f]['only'].sum() for f in p3)); timed = sum(1 for f in p4 for q in p4[f]['dev'] if p4[f]['parsed'].get(q) and not p3[f]['parsed'].get(q))
    report = {'kind': 'learned scorer revision 5 (parser v4 time limits, "only" exclusivity), folds 0-2', 'summary': summarise(rows),
              'versus_linear_p3': result, 'only_queries_over_folds': only_queries, 'newly_parsed_by_v4_over_folds': timed, 'audit_used': False}
    atomic_json(OUT / 'report-eval4.json', report); print(json.dumps({**result, 'only_queries': only_queries, 'newly_parsed_by_v4': timed}, indent=1))


def evaluate_ego_pilot(root=ROOT / 'runs/gpu-2026-10-22', job='ego-score-f0', name='ego-pilot'):
    from baseline.native_ranking import blended_scores
    from scripts.feature_retrieval import atomic_json
    from scripts.post_submission import verified_result
    d = load(0); e = np.load(verified_result(root, job, 'vlm-atom-probs.npz'))
    if [str(v) for v in e['video_ids']] != d['held']:
        raise ValueError(f'{job} answers not aligned with fold 0')
    swapped = dict(d); swapped['qwen_logit'] = dict(d['qwen_logit'])
    for j, a in enumerate(str(x) for x in e['atoms']):
        if a in swapped['qwen_logit']:
            swapped['qwen_logit'][a] = logit(e['probs'][:, j])
    fused = lambda x, seed: (lambda raw, sub, s, m: np.where(m[:, None], blended_scores(np.where(m[:, None], s, raw), sub, WEIGHT), sub))(
        x[f'raw_{seed}'], x[f'sub_{seed}'], *summed(x, combined(x)))
    cells = {seed: {'current': metrics(d, fused(d, seed))[0], name: metrics(swapped, fused(swapped, seed))[0]} for seed in (43, 44)}
    gains = [c[name] - c['current'] for c in cells.values()]
    report = {'kind': f'{name} pilot on fold 0 (entry-3 recipe; its answers replace Qwen3.8 answers for the atoms it covers)', 'cells': {str(k): v for k, v in cells.items()},
              'rp_gain_per_seed': gains, 'passes': bool(min(gains) >= 0.005), 'rule': 'RP gain >= +0.005 on both head seeds', 'audit_used': False}
    atomic_json(OUT / f'report-{name}.json', report); print(json.dumps(report, indent=1))


def evaluate_mirror(root=ROOT / 'runs/gpu-2026-10-22'):
    from scripts.feature_retrieval import atomic_json
    from scripts.perception_v3_eval import compare, summarise
    base = {f: load(f) for f in (0, 1, 2)}; mirrored = {}
    for f, d in base.items():
        m = np.load(root / f'mirror-f{f}/vlm-atom-probs-mirror.npz')
        if [str(v) for v in m['video_ids']] != d['held'] or [str(a) for a in m['atoms']] != d['qwen_atoms']:
            raise ValueError('Mirror answers not aligned with fold %d' % f)
        e = dict(d); e['qwen'] = sigmoid((logit(d['qwen']) + logit(m['probs'])) / 2)
        e['qwen_logit'] = {a: logit(e['qwen'][:, j]) for j, a in enumerate(e['qwen_atoms'])}; mirrored[f] = e
    rows = []
    for f in (0, 1, 2):
        for seed in (43, 44):
            for name, folds in (('linear', base), ('linear-mirror', mirrored)):
                scores = train_scorer([folds[g] for g in folds if g != f], seed, False, 10, linear=True)(folds[f])
                rp, ap = metrics(folds[f], scores); rows.append({'fold': f, 'seed': seed, 'arm': name, 'r_precision': rp, 'map': ap})
            print(f, seed, {r['arm']: round(r['r_precision'], 4) for r in rows if r['fold'] == f and r['seed'] == seed}, flush=True)
    c = compare(rows, 'linear-mirror', 'linear')
    c['passes'] = bool(c['mean_rp_delta'] > .001 and c['cells_improved'] >= 4 and min(c['per_fold_rp_delta'].values()) >= 0 and c['mean_map_delta'] >= 0)
    report = {'kind': 'mirror-averaged Qwen3.8 answers under the linear learned scorer, folds 0-2', 'summary': summarise(rows),
              'mirror_vs_linear': c, 'audit_used': False}
    atomic_json(OUT / 'report-mirror.json', report); print(json.dumps(c, indent=1))


def verify_matrix(d, path, k):
    z = np.load(path); qi = {q: i for i, q in enumerate(d['dev'])}; vi = {v: j for j, v in enumerate(d['held'])}
    m = np.full(d['rel'].shape, np.nan, np.float32)
    for q, v, l in zip(z['query_ids'], z['video_ids'], z['logits']):
        m[qi[str(q)], vi[str(v)]] = l
    if np.isfinite(m).sum(1).min() < k:
        raise ValueError(f'A query lacks its {k} verifier candidates: {path}')
    return m


def prop_probs():
    from baseline.probe import fit_logistic, standardize
    from baseline.property_atoms import load_labels, present, query_atoms as prop_atoms
    from baseline.query_parser import Unparsed
    from baseline.query_parser_v2 import translate as v2
    from scripts.structured_eval import provider
    from scripts.structured_v2 import COMBO, late
    plan = json.loads(PLAN.read_text()); lab = load_labels(ROOT / plan['annotation_directory']); texts = query_texts()
    folds = {f: load(f) for f in (0, 1, 2)}; dev = sorted({v for d in folds.values() for v in d['held']}); pos = {v: i for i, v in enumerate(dev)}
    blocks = {b: provider(b, plan['blocks'][b], dev)[0] for b in COMBO}
    def atoms_of(q):
        try:
            return prop_atoms(v2(texts[q]))
        except Unparsed:
            return []
    for f, d in folds.items():
        held = np.asarray([pos[v] for v in d['held']]); fit = np.setdiff1d(np.arange(len(dev)), held)
        per_q = [atoms_of(q) for q in d['dev']]; atoms = sorted({a for x in per_q for a in x})
        probs = {}
        for a in atoms:
            y = np.asarray([present(a, lab[v]) for v in dev])
            parts = []
            for b in COMBO:
                if y[fit].sum() < 3 or y[fit].sum() > len(fit) - 3:
                    parts.append(np.full(len(held), float(np.clip(y[fit].mean(), 1e-3, 1 - 1e-3)))); continue
                z = standardize(blocks[b], fit); w, c = fit_logistic(z[fit], y[fit])
                parts.append(np.clip(1 / (1 + np.exp(-(z[held] @ w + c))), 1e-4, 1 - 1e-4))
            probs[a] = late(parts)
        score = np.zeros((len(d['dev']), len(held))); mask = np.zeros(len(d['dev']), bool)
        for i, x in enumerate(per_q):
            if x:
                score[i] = np.sum([np.log(probs[a]) for a in x], 0); mask[i] = True
        np.savez_compressed(OUT / f'props-f{f}.npz', score=score, mask=mask, atoms=np.asarray(atoms))
        print(f'fold {f}: {int(mask.sum())} of {len(mask)} queries with property atoms, {len(atoms)} atoms', flush=True)


def qwen_prop_scores(d, f):
    from baseline.property_atoms import query_atoms as prop_atoms
    from baseline.query_parser import Unparsed
    from baseline.query_parser_v2 import translate as v2
    z = np.load(ROOT / f'runs/gpu-2026-10-22/props-f{f}/props.npz')
    if [str(v) for v in z['video_ids']] != d['held']:
        raise ValueError('Property answers not aligned with fold %d' % f)
    p = {str(a): np.clip(z['probs'][:, j], 1e-4, 1 - 1e-4) for j, a in enumerate(z['atoms'])}; texts = query_texts()
    score = np.zeros(d['rel'].shape); mask = np.zeros(len(d['dev']), bool)
    for i, q in enumerate(d['dev']):
        try:
            x = [a for a in prop_atoms(v2(texts[q])) if a in p]
        except Unparsed:
            x = []
        if x:
            score[i] = np.sum([np.log(p[a]) for a in x], 0); mask[i] = True
    return score, mask


def evaluate_prop(source='t4'):
    from scripts.feature_retrieval import atomic_json
    from scripts.perception_v3_eval import compare, summarise
    base = {f: load(f) for f in (0, 1, 2)}; with_p = {}
    for f, d in base.items():
        if source == 't4':
            z = np.load(OUT / f'props-f{f}.npz'); score, mask = z['score'], z['mask']
        else:
            score, mask = qwen_prop_scores(d, f)
        with_p[f] = {**d, 'prop_score': score, 'prop_mask': mask}
    arm = 'linear-prop' if source == 't4' else 'linear-prop-qwen'
    rows = []
    for f in (0, 1, 2):
        for seed in (43, 44):
            for name, folds in (('linear', base), (arm, with_p)):
                scores = train_scorer([folds[g] for g in folds if g != f], seed, False, 10, linear=True)(folds[f])
                rp, ap = metrics(folds[f], scores); rows.append({'fold': f, 'seed': seed, 'arm': name, 'r_precision': rp, 'map': ap})
            print(f, seed, {r['arm']: round(r['r_precision'], 4) for r in rows if r['fold'] == f and r['seed'] == seed}, flush=True)
    c = compare(rows, arm, 'linear')
    c['passes'] = bool(c['mean_rp_delta'] > .001 and c['cells_improved'] >= 4 and min(c['per_fold_rp_delta'].values()) >= 0 and c['mean_map_delta'] >= 0)
    report = {'kind': f'property-atom features ({source}) under the linear learned scorer, folds 0-2', 'summary': summarise(rows),
              'prop_vs_linear': c, 'audit_used': False}
    atomic_json(OUT / ('report-prop.json' if source == 't4' else 'report-prop-qwen.json'), report); print(json.dumps(c, indent=1))


def evaluate_verify2(root=ROOT / 'runs/gpu-2026-10-22'):
    from scripts.feature_retrieval import atomic_json
    from scripts.perception_v3_eval import compare, summarise
    v2 = {f: load(f) for f in (0, 1, 2)}; p3 = {f: load(f, OUT / 'p3') for f in (0, 1, 2)}
    arms = {'linear-verify': {f: {**v2[f], 'verify': verify_matrix(v2[f], root / f'verify-f{f}/verify.npz', 30)} for f in v2},
            'linear-p3-verify': {f: {**p3[f], 'verify': verify_matrix(p3[f], root / f'verify-f{f}/verify.npz', 30)} for f in p3}}
    for name, folder, k in (('linear-verify50', 'verify50', 50), ('linear-verifyft', 'verifyft', 30)):
        if all((root / f'{folder}-f{f}/verify.npz').exists() for f in v2):
            arms[name] = {f: {**v2[f], 'verify': verify_matrix(v2[f], root / f'{folder}-f{f}/verify.npz', k)} for f in v2}
    rows = []
    for f in (0, 1, 2):
        for seed in (43, 44):
            for name, folds in arms.items():
                scores = train_scorer([folds[g] for g in folds if g != f], seed, False, 10, linear=True)(folds[f])
                rp, ap = metrics(folds[f], scores); rows.append({'fold': f, 'seed': seed, 'arm': name, 'r_precision': rp, 'map': ap})
            print(f, seed, {r['arm']: round(r['r_precision'], 4) for r in rows if r['fold'] == f and r['seed'] == seed}, flush=True)
    gate = lambda c: bool(c['mean_rp_delta'] > .001 and c['cells_improved'] >= 4 and min(c['per_fold_rp_delta'].values()) >= 0 and c['mean_map_delta'] >= 0)
    result = {a: {**compare(rows, a, 'linear-verify')} for a in arms if a != 'linear-verify'}
    for c in result.values():
        c['passes'] = gate(c)
    report = {'kind': 'learned scorer revision 6 (verifier variants), folds 0-2', 'summary': summarise(rows), 'versus_linear_verify': result,
              'arms_evaluated': sorted(arms), 'audit_used': False}
    atomic_json(OUT / 'report-verify2.json', report); print(json.dumps(result, indent=1))


def evaluate_verify(root=ROOT / 'runs/gpu-2026-10-22'):
    from scripts.feature_retrieval import atomic_json
    from scripts.perception_v3_eval import compare, summarise
    base = {f: load(f) for f in (0, 1, 2)}; with_v = {}
    for f, d in base.items():
        z = np.load(root / f'verify-f{f}/verify.npz'); qi = {q: i for i, q in enumerate(d['dev'])}; vi = {v: j for j, v in enumerate(d['held'])}
        m = np.full(d['rel'].shape, np.nan, np.float32)
        for q, v, l in zip(z['query_ids'], z['video_ids'], z['logits']):
            m[qi[str(q)], vi[str(v)]] = l
        if np.isfinite(m).sum(1).min() < 30:
            raise ValueError('A query lacks its 30 verifier candidates on fold %d' % f)
        with_v[f] = {**d, 'verify': m}
    rows = []
    for f in (0, 1, 2):
        for seed in (43, 44):
            for name, folds in (('linear', base), ('linear-verify', with_v)):
                scores = train_scorer([folds[g] for g in folds if g != f], seed, False, 10, linear=True)(folds[f])
                rp, ap = metrics(folds[f], scores); rows.append({'fold': f, 'seed': seed, 'arm': name, 'r_precision': rp, 'map': ap})
            print(f, seed, {r['arm']: round(r['r_precision'], 4) for r in rows if r['fold'] == f and r['seed'] == seed}, flush=True)
    c = compare(rows, 'linear-verify', 'linear')
    c['passes'] = bool(c['mean_rp_delta'] > .001 and c['cells_improved'] >= 4 and min(c['per_fold_rp_delta'].values()) >= 0 and c['mean_map_delta'] >= 0)
    report = {'kind': 'whole-query verifier feature under the linear learned scorer, folds 0-2', 'summary': summarise(rows),
              'verify_vs_linear': c, 'audit_used': False}
    atomic_json(OUT / 'report-verify.json', report); print(json.dumps(c, indent=1))


def audit_prep():
    """Parser-v3 audit cache and the audit verifier candidates (no audit metric is computed here)."""
    import scripts.perception_v3_eval as pe
    from baseline.native_ranking import blended_scores
    from baseline.query_parser_v3 import translate
    from scripts.feature_retrieval import load_vectors
    from scripts.perception_v3_eval import fold_state
    from scripts.post_submission import verified_result
    pe.translate_v2 = translate
    plan = json.loads(PLAN.read_text()); qvec = load_vectors(ROOT / plan['feature_directory'] / 'queries-train.npz')
    st = fold_state(plan, 'audit', None); (OUT / 'p3').mkdir(parents=True, exist_ok=True)
    save_state(st, np.load(verified_result(ROOT / 'runs/audit-v2', 'audit-vlm-score', 'vlm-atom-probs.npz')), qvec, OUT / 'p3/audit.npz')
    d = load('audit'); s, m = summed(d, combined(d))
    cur = np.where(m[:, None], blended_scores(np.where(m[:, None], s, d['raw_43']), d['sub_43'], WEIGHT), d['sub_43']); per = {}
    for i, q in enumerate(d['dev']):
        for j in np.argsort(-cur[i], kind='stable')[:30]:
            per.setdefault(d['held'][j], []).append(q)
    texts = query_texts()
    (OUT / 'verify-candidates-audit.json').write_text(json.dumps({'rule': 'top-30 per audit query under the entry-3 recipe (seed-43 head)', 'k': 30,
        'folds': {'audit': per}, 'texts': {q: texts[q] for q in d['dev']}}))
    print('audit p3 cache and verifier candidates written:', len(per), 'clips', flush=True)


def audit_bundle(root=ROOT / 'runs/gpu-2026-10-22'):
    from scripts.feature_retrieval import atomic_json
    from scripts.perception_v3_eval import compare, summarise
    train = [{**load(f, OUT / 'p3'), 'verify': verify_matrix(load(f, OUT / 'p3'), root / f'verify-f{f}/verify.npz', 30)} for f in (0, 1, 2)]
    d = load('audit', OUT / 'p3'); d = {**d, 'verify': verify_matrix(d, root / 'verify-audit-faudit/verify.npz', 30)}
    recorded = {r['seed']: r for r in json.loads((OUT / 'cells-audit.json').read_text()) if r['arm'] == 'linear'}
    rows = [{**recorded[s], 'arm': 'linear'} for s in (43, 44)]
    for seed in (43, 44):
        rp, ap = metrics(d, train_scorer(train, seed, False, 10, linear=True)(d))
        rows.append({'fold': 'audit', 'seed': seed, 'arm': 'bundle', 'r_precision': rp, 'map': ap})
    c = compare(rows, 'bundle', 'linear')
    confirmed = bool(c['mean_rp_delta'] > .001 and c['cells_improved'] == c['cells'] and c['mean_map_delta'] >= 0)
    report = {'kind': 'audit confirmation of the bundle (linear scorer + parser v3 + zero-shot verifier); fourth look at the audit set',
              'summary': summarise(rows), 'bundle_vs_audited_linear': c, 'confirmed': confirmed, 'audit_used': True,
              'rule': __doc__.split('Bundle audit')[1].split('Usage')[0].strip()}
    atomic_json(OUT / 'report-audit-bundle.json', report); print(json.dumps(report, indent=1))


def audit():
    from baseline.native_ranking import blended_scores
    from scripts.feature_retrieval import atomic_json, load_vectors
    from scripts.perception_v3_eval import compare, fold_state, summarise
    from scripts.post_submission import verified_result
    plan = json.loads(PLAN.read_text()); qvec = load_vectors(ROOT / plan['feature_directory'] / 'queries-train.npz')
    st = fold_state(plan, 'audit', None)
    save_state(st, np.load(verified_result(ROOT / 'runs/audit-v2', 'audit-vlm-score', 'vlm-atom-probs.npz')), qvec, OUT / 'audit.npz')
    d = load('audit'); train = [load(f) for f in (0, 1, 2)]; rows = []
    fused = lambda raw, sub, s, m: np.where(m[:, None], blended_scores(np.where(m[:, None], s, raw), sub, WEIGHT), sub)
    for seed in (43, 44):
        arms = {'current': fused(d[f'raw_{seed}'], d[f'sub_{seed}'], *summed(d, combined(d))),
                'linear': train_scorer(train, seed, False, 10, linear=True)(d)}
        for name, scores in arms.items():
            rp, ap = metrics(d, scores); rows.append({'fold': 'audit', 'seed': seed, 'arm': name, 'r_precision': rp, 'map': ap})
    c = compare(rows, 'linear', 'current')
    confirmed = bool(c['mean_rp_delta'] > .001 and c['cells_improved'] == c['cells'] and c['mean_map_delta'] >= 0)
    report = {'kind': 'audit confirmation of the linear learned scorer (third look at the audit set)', 'summary': summarise(rows),
              'linear_vs_current': c, 'confirmed': confirmed, 'audit_used': True, 'audit_queries': len(d['dev']), 'audit_clips': len(d['held']),
              'rule': __doc__.split('Audit confirmation')[1].split('Usage')[0].strip()}
    atomic_json(OUT / 'report-audit.json', report); atomic_json(OUT / 'cells-audit.json', rows); print(json.dumps(report, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('job', choices=['cache', 'cache-p3', 'cache-p4', 'eval', 'eval2', 'eval3', 'eval4', 'eval-mirror', 'eval-verify', 'eval-verify2', 'eval-ego-pilot', 'eval-sweep-pilot', 'prop-probs', 'eval-prop', 'eval-prop-qwen', 'audit', 'audit-prep', 'audit-bundle'])
    ap.add_argument('--name')
    a = ap.parse_args()
    if a.job == 'cache':
        cache()
    elif a.job == 'cache-p3':
        from baseline.query_parser_v3 import translate
        cache(OUT / 'p3', translate)
    elif a.job == 'cache-p4':
        from baseline.query_parser_v4 import translate
        cache(OUT / 'p4', translate)
    elif a.job == 'eval3':
        evaluate3()
    elif a.job == 'eval4':
        evaluate4()
    elif a.job == 'eval-mirror':
        evaluate_mirror()
    elif a.job == 'eval-verify':
        evaluate_verify()
    elif a.job == 'eval-verify2':
        evaluate_verify2()
    elif a.job == 'eval-ego-pilot':
        evaluate_ego_pilot()
    elif a.job == 'prop-probs':
        prop_probs()
    elif a.job == 'eval-prop':
        evaluate_prop()
    elif a.job == 'eval-prop-qwen':
        evaluate_prop('qwen')
    elif a.job == 'eval-sweep-pilot':
        evaluate_ego_pilot(job=f'sweep-{a.name}-score-f0', name=f'sweep-{a.name}')
    elif a.job == 'eval':
        evaluate()
    elif a.job == 'audit':
        audit()
    elif a.job == 'audit-prep':
        audit_prep()
    elif a.job == 'audit-bundle':
        audit_bundle()
    else:
        evaluate2()


if __name__ == '__main__':
    main()
