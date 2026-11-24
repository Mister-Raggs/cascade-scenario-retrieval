"""Multi-task 30B-class answerer (2026-10-21): atom questions + query-level verification + mirror augmentation.

One LoRA adapter on the shared-video-prefix scorer learns two tasks per training clip: atom yes/no questions (as in the
scaled recipe) and whole-query verification ("Does this video match the following description? ..."), with positives from
the fitting qrels and near-miss negatives (every parsed atom present, clip still not relevant). With probability
`mirror_probability` the clip is mirrored left-right and every question is mirrored too (atoms swap left/right, query
words swap), keeping the original labels. Scoring answers every atom question on the original and the mirrored clip (test-
time augmentation within one model) and every development query's verification question on the original clip.

Jobs: 'train' and 'score' (GPU). Development folds only; fitting queries and fitting clips only for training.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from baseline.atom_questions import question
from baseline.mirror import mirror_atom, mirror_text
from baseline.query_pairs import clip_query_pools, sample_queries, verifier_question
from baseline.query_struct import atom_present
from baseline.track_data import atom_question_batches
from scripts.feature_retrieval import atomic_json
from scripts.post_submission import stage_directory, verified_result
from scripts.query_side import finish
from scripts.reranker_finetune import fold_data
from scripts.tracks_eval import parse, translate_v2
from scripts.vlm_atom_finetune import labels
from scripts.vlm_atom_v2 import SUFFIX, SharedPrefixAnswerer, plan_atoms


class MirrorFrames:
    """Frame store wrapper returning left-right mirrored frames while `mirror` is set (cache holds originals)."""

    def __init__(self, store):
        self.store = store; self.mirror = False

    def get(self, vid):
        from PIL import ImageOps
        frames, meta = self.store.get(vid)
        return ([ImageOps.mirror(f) for f in frames] if self.mirror else frames), meta


class MultiTaskAnswerer(SharedPrefixAnswerer):
    """Shared-prefix scorer over free-text questions, with an optional mirrored view."""

    def __init__(self, plan, adapter=None):
        super().__init__(plan, adapter); self.frames = MirrorFrames(self.frames)

    def suffix_ids(self, text):
        return self.processor.tokenizer.encode(text + SUFFIX + self.post, add_special_tokens=False)

    def check_split(self, vid, batch, frames, meta, text):
        from baseline.h100_models import visual_batch
        full, _ = visual_batch(self.processor, [self.conversation(frames, text + SUFFIX)], [meta], self.plan['max_length'],
                               template_kwargs=self.template)
        if full['input_ids'][0].tolist() != batch['input_ids'][0].tolist() + self.suffix_ids(text):
            raise ValueError('Shared-prefix tokenization differs from the full prompt: ' + vid)
        return full

    def full_logits(self, vid, texts):
        import torch
        from baseline.h100_models import visual_batch
        frames, meta = self.frames.get(vid); values = []
        for t in texts:
            batch, _ = visual_batch(self.processor, [self.conversation(frames, t + SUFFIX)], [meta], self.plan['max_length'],
                                    template_kwargs=self.template)
            last = self.model(**{k: v.cuda() for k, v in batch.items()}, logits_to_keep=1).logits[:, -1, :].float()
            values.append(torch.logsumexp(last[:, self.yes], 1) - torch.logsumexp(last[:, self.no], 1))
        return torch.cat(values)

    def answer(self, vid, texts, mirror=False):
        self.frames.mirror = mirror
        try:
            return self.logits(vid, texts)
        finally:
            self.frames.mirror = False


def setup(plan):
    from scripts.tracks_eval import common
    parent, queries, qrels, fm, vids, ix, texts, sets = common(plan)
    vocab, truth = labels(plan, vids); atoms = plan_atoms(plan, vocab)
    fit_q = list(ix['fit_query_ids']); qtext = {q: queries[q]['text'] for q in fit_q}
    extra = plan.get('extra_val_queries')  # entry 5 (2026-10-23): validation queries join the verification pool
    if extra:
        from baseline.io import unique_rows
        from scripts.val_eval import scenarios
        vq = unique_rows(ROOT / 'data/retrieval/cascade-v1.0/val/queries.jsonl', 'query_id'); vr = unique_rows(ROOT / 'data/retrieval/cascade-v1.0/val/qrels.jsonl', 'query_id')
        half, _ = scenarios(); keep = [q for q in vq if extra == 'all' or half[q] == extra['half']]
        queries = {**queries, **{q: vq[q] for q in keep}}; qrels = {**qrels, **{q: vr[q] for q in keep}}
        fit_q = fit_q + keep; qtext.update({q: vq[q]['text'] for q in keep})
    parsed = parse(qtext, translate_v2)
    needed = sorted({a for v in parsed.values() if v for a in v})
    qtruth = {a: np.asarray([atom_present(a, sets[v]) for v in vids]) for a in needed}
    return dict(queries=queries, qrels=qrels, vids=vids, ix=ix, sets=sets, atoms=atoms, truth=truth, fit_q=fit_q, qtext=qtext,
                parsed=parsed, qtruth=qtruth)


def train(out, plan):
    import torch
    from safetensors.torch import save_file
    from baseline.lora import adapter_state
    d = setup(plan); vids, ix, atoms, truth, sets = d['vids'], d['ix'], d['atoms'], d['truth'], d['sets']
    fit = np.asarray(ix['fit_v']); rng = np.random.default_rng(plan['seed'])
    pools = clip_query_pools(d['fit_q'], d['qrels'], d['parsed'], d['qtruth'], vids, fit)
    index = {v: i for i, v in enumerate(vids)}
    relevant_of = {q: {index[v] for v in d['qrels'][q]['relevant_video_ids'] if v in index} for q in d['fit_q']}
    model = MultiTaskAnswerer(plan); params = [p for p in model.model.parameters() if p.requires_grad]
    frozen = {n: p._version for n, p in model.model.named_parameters() if not p.requires_grad}
    initial = adapter_state(model.model); model.mode(True)
    opt = torch.optim.AdamW(params, lr=plan['learning_rate'], weight_decay=.01)
    losses, atom_losses, query_losses, mirrored = [], [], [], 0; started = time.monotonic()
    for step in range(plan['optimizer_steps']):
        if step % len(fit) == 0:
            examples = atom_question_batches(rng.permutation(fit), truth, atoms, rng, plan['atom_questions'])
        i, items = examples[step % len(fit)]
        qs = sample_queries(pools, i, d['fit_q'], relevant_of, rng, plan['query_questions'])
        mirror = bool(rng.random() < plan['mirror_probability']); mirrored += mirror
        # A mirrored clip keeps its labels; each question is mirrored so that it asks about the original content.
        atom_texts = [question(mirror_atom(a) if mirror else a) for a, _ in items]
        query_texts = [verifier_question(mirror_text(d['qtext'][q]) if mirror else d['qtext'][q]) for q, _ in qs]
        y = torch.tensor([float(l) for _, l in items] + [float(l) for _, l in qs], device='cuda')
        z = model.answer(vids[i], atom_texts + query_texts, mirror)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite loss')
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1); opt.step()
        losses.append(float(loss.detach())); k = len(items)
        atom_losses.append(float(torch.nn.functional.binary_cross_entropy_with_logits(z[:k].detach(), y[:k])))
        if len(qs):
            query_losses.append(float(torch.nn.functional.binary_cross_entropy_with_logits(z[k:].detach(), y[k:])))
        if torch.cuda.max_memory_reserved() >= plan['max_reserved_gib'] * 1024 ** 3:
            raise ValueError('Fine-tune exceeded its memory budget')
        if (step + 1) % 25 == 0:
            atomic_json(out / 'progress.json', {'optimizer_steps': step + 1, 'total': plan['optimizer_steps'],
                        'recent_loss': float(np.mean(losses[-25:])), 'recent_atom_loss': float(np.mean(atom_losses[-25:])),
                        'recent_query_loss': float(np.mean(query_losses[-25:])) if query_losses else None,
                        'elapsed_seconds': time.monotonic() - started})
    final = adapter_state(model.model)
    if not any(not torch.equal(v, initial[n]) for n, v in final.items()) or \
       not all(p._version == frozen[n] for n, p in model.model.named_parameters() if n in frozen):
        raise ValueError('Adapter/base-weight correctness gate failed')
    save_file(final, str(out / 'results/vlm-adapter.safetensors'))
    pool_sizes = {'positives_median': float(np.median([len(p) for p, _ in pools.values()])),
                  'near_misses_median': float(np.median([len(n) for _, n in pools.values()]))}
    finish(out, plan, {'kind': 'multi-task (atoms + query verification + mirror) LoRA fine-tune', 'gate_passed': True, 'accuracy_score': False,
           'validation_used': bool(plan.get('extra_val_queries')), 'verification_queries': len(d['fit_q']),
           'losses': losses, 'atom_losses': atom_losses, 'query_losses': query_losses, 'mirrored_steps': int(mirrored),
           'query_pools': pool_sizes, 'held_videos_in_training': 0, 'seconds': time.monotonic() - started,
           'peak_reserved_bytes': torch.cuda.max_memory_reserved()})


def score(out, plan):
    import torch
    from baseline.probe import auc
    d = setup(plan); vids, ix, atoms, truth = d['vids'], d['ix'], d['atoms'], d['truth']
    model = MultiTaskAnswerer(plan, verified_result(out.parent, plan['job_names']['train'], 'vlm-adapter.safetensors')); model.mode(False)
    held = list(ix['held_v']); dev_q = list(ix['dev_query_ids']); dev_text = [verifier_question(d['queries'][q]['text']) for q in dev_q]
    orig = np.zeros((len(held), len(atoms)), np.float32); mir = np.zeros_like(orig); ver = np.zeros((len(held), len(dev_q)), np.float16)
    sig = lambda z: 1 / (1 + np.exp(-z))
    with torch.inference_mode():
        probe = [question(atoms[0]), dev_text[0]]
        drift = float(np.abs(model.answer(vids[held[0]], probe).cpu().numpy() - model.full_logits(vids[held[0]], probe).cpu().numpy()).max())
        if drift > plan['max_prefix_logit_drift']:
            raise ValueError(f'Shared-prefix logits differ from full prompts by {drift:.3f}')
        started = time.monotonic()
        for n, i in enumerate(held):
            z = model.answer(vids[i], [question(a) for a in atoms] + dev_text).cpu().numpy()
            orig[n] = sig(z[:len(atoms)]); ver[n] = sig(z[len(atoms):])
            mir[n] = sig(model.answer(vids[i], [question(mirror_atom(a)) for a in atoms], True).cpu().numpy())
            if n % 16 == 0:
                atomic_json(out / 'progress.json', {'videos': n + 1, 'total': len(held), 'elapsed_seconds': time.monotonic() - started})
    seconds = (time.monotonic() - started) / len(held)
    logit = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
    tta = sig((logit(orig) + logit(mir)) / 2).astype(np.float32)
    np.savez_compressed(out / 'results/vlm-atom-probs.npz', video_ids=np.asarray([vids[i] for i in held]), atoms=np.asarray(atoms), probs=tta,
                        probs_original=orig, probs_mirrored=mir, query_ids=np.asarray(dev_q), verifier=ver)
    rows = [{'atom': a, 'positives': int(truth[a][held].sum()), 'auc_original': auc(orig[:, j], truth[a][held]),
             'auc_mirrored': auc(mir[:, j], truth[a][held]), 'auc_tta': auc(tta[:, j], truth[a][held])} for j, a in enumerate(atoms)]
    valid = [r for r in rows if r['positives'] >= 5]
    held_ids = [vids[i] for i in held]
    rel = np.asarray([[v in set(d['qrels'][q]['relevant_video_ids']) for q in dev_q] for v in held_ids])
    ver_auc = [auc(ver[:, k].astype(float), rel[:, k]) for k in range(len(dev_q)) if 0 < rel[:, k].sum() < len(held)]
    finish(out, plan, {'kind': 'multi-task 30B-class atom + verifier scores on held videos', 'gate_passed': True, 'accuracy_score': True,
           'held_fold': plan['held_fold'], 'audit_used': plan['held_fold'] == 'audit', 'atoms': rows,
           'mean_auc_original': float(np.nanmean([r['auc_original'] for r in valid])), 'mean_auc_mirrored': float(np.nanmean([r['auc_mirrored'] for r in valid])),
           'mean_auc_tta': float(np.nanmean([r['auc_tta'] for r in valid])), 'verifier_mean_query_auc': float(np.nanmean(ver_auc)),
           'prefix_logit_drift': drift, 'questions_per_clip': 2 * len(atoms) + len(dev_q), 'seconds_per_video_h200': seconds})


JOBS = {'train': train, 'score': score}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', choices=JOBS, required=True)
    parser.add_argument('--plan', type=Path, required=True); parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); os.chdir(ROOT)
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    import torch
    plan = json.loads(args.plan.read_text()); plan['plan_path'] = str(args.plan)
    if not plan['development_only'] or plan['official_submission_allowed']:
        raise ValueError('Only private local experiments are supported')
    torch.set_num_threads(plan['cpu_threads']); stage_directory(args.out)
    JOBS[args.job](args.out, plan)


if __name__ == '__main__':
    main()
