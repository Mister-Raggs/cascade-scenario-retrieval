"""Track 2: LoRA fine-tuned Qwen3-VL-Reranker on fitting-fold pairs (near-miss hard negatives), then top-K rescoring.

Jobs: 'train' (GPU) and 'score' (GPU: rescore the evaluator's top-K candidates for held videos, with throughput).
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
from baseline.annotation_events import load_events
from baseline.io import sha256_file, unique_rows
from baseline.post_submission import fold_indices
from baseline.query_struct import atom_present
from baseline.track_data import reranker_pairs
from scripts.feature_retrieval import atomic_json
from scripts.post_submission import stage_directory, verified_result
from scripts.query_side import finish
from scripts import lora_pilot


def fold_data(plan):
    parent, queries, qrels = lora_pilot.inputs(plan)
    fm = json.loads((ROOT / plan['feature_directory'] / 'feature_manifest.json').read_text()); vids = fm['video_ids']
    ix = fold_indices(parent, qrels, fm['query_ids'], vids, plan['held_fold'])
    return parent, queries, qrels, vids, ix


def training_queries(plan, queries, qrels, ix):
    rows = [(q, queries[q]['text']) for q in ix['fit_query_ids']]
    labels = {q: qrels[q]['relevant_video_ids'] for q, _ in rows}
    if plan.get('synthetic'):
        synth = unique_rows(ROOT / plan['synthetic']['queries'], 'query_id'); srels = unique_rows(ROOT / plan['synthetic']['qrels'], 'query_id')
        if sha256_file(ROOT / plan['synthetic']['queries']) != plan['synthetic']['queries_sha256']:
            raise ValueError('Synthetic queries changed')
        for q, r in synth.items():
            rows.append((q, r['text'])); labels[q] = srels[q]['relevant_video_ids']
    return rows, labels


class Reranker:
    def __init__(self, plan, adapter=None):
        import torch
        from baseline.h100_models import VideoReranker, FrameStore
        from baseline.lora import attach_language_lora, load_adapter_state
        path = ROOT / plan['reranker_directory']
        if json.loads((path / 'model_provenance.json').read_text())['revision'] != plan['reranker_revision']:
            raise ValueError('Pinned reranker revision differs')
        self.base = VideoReranker(str(path)); self.frames = FrameStore(ROOT / plan['frame_directory'], 4)
        attach_language_lora(self.base.model, plan['rank'], plan['alpha'])
        if adapter is not None:
            from safetensors.torch import load_file
            load_adapter_state(self.base.model, load_file(str(adapter)))
        self.torch = torch

    def logits(self, pairs):
        """pairs: [(query text, video id)] -> yes-minus-no logits (differentiable when grad is enabled)."""
        from baseline.h100_models import visual_batch
        b = self.base; conversations, metadata = [], []
        for text, vid in pairs:
            images, meta = self.frames.get(vid)
            c = b.formatter.format_mm_instruction(query_text=text, doc_video=images, instruction=b.instruction)
            for item in c[1]['content']:
                if item['type'] == 'video':
                    item.update(resized_height=256, resized_width=448, min_pixels=4 * 32 * 32)
            conversations.append(c); metadata.append(meta)
        batch, _ = visual_batch(b.processor, conversations, metadata)
        batch = {k: v.cuda() for k, v in batch.items()}
        hidden = b.model(**batch, use_cache=False).last_hidden_state[:, -1]
        return (hidden.float() @ b.weight.float())


def train(out, plan):
    import torch
    from safetensors.torch import save_file
    from baseline.lora import adapter_state
    parent, queries, qrels, vids, ix = fold_data(plan)
    events = load_events(ROOT / plan['annotation_directory']); labels_of = {v: {e[0] for e in events[v]['events']} for v in vids}
    rows, labels = training_queries(plan, queries, qrels, ix)
    from baseline.structured_recipe import query_atoms
    atom_set = sorted({a for _, t in rows for a in (query_atoms(t) or [])})
    truth = {a: np.asarray([atom_present(a, labels_of[v]) for v in vids]) for a in atom_set}
    rng = np.random.default_rng(plan['seed']); fit = np.asarray(ix['fit_v'])
    pairs = reranker_pairs(rows, labels, vids, fit, truth, rng, plan['negatives'], plan['positives'])
    rng.shuffle(pairs)
    model = Reranker(plan); params = [p for p in model.base.model.parameters() if p.requires_grad]
    frozen = {n: p._version for n, p in model.base.model.named_parameters() if not p.requires_grad}
    initial = adapter_state(model.base.model)
    model.base.model.train(); opt = torch.optim.AdamW(params, lr=plan['learning_rate'], weight_decay=.01)
    losses = []; started = time.monotonic(); size = plan['pair_batch']
    for step in range(plan['optimizer_steps']):
        chunk = [pairs[(step * size + j) % len(pairs)] for j in range(size)]
        y = torch.tensor([float(l) for _, _, l in chunk], device='cuda')
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model.logits([(t, vids[i]) for t, i, _ in chunk]), y)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite reranker loss')
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1); opt.step()
        losses.append(float(loss.detach()))
        if torch.cuda.max_memory_reserved() >= plan['max_reserved_gib'] * 1024 ** 3:
            raise ValueError('Reranker fine-tune exceeded its memory budget')
        if (step + 1) % 25 == 0:
            atomic_json(out / 'progress.json', {'optimizer_steps': step + 1, 'total': plan['optimizer_steps'],
                        'recent_loss': float(np.mean(losses[-25:])), 'elapsed_seconds': time.monotonic() - started})
    final = adapter_state(model.base.model)
    changed = any(not torch.equal(v, initial[n]) for n, v in final.items())
    untouched = all(p._version == frozen[n] for n, p in model.base.model.named_parameters() if n in frozen)
    if not changed or not untouched:
        raise ValueError('Adapter/base-weight correctness gate failed')
    save_file(final, str(out / 'results/reranker-adapter.safetensors'))
    finish(out, plan, {'kind': 'LoRA fine-tuned reranker on fitting-fold pairs', 'gate_passed': True, 'accuracy_score': False,
           'pairs_available': len(pairs), 'pairs_seen': plan['optimizer_steps'] * size, 'losses': losses,
           'held_videos_in_training': 0, 'synthetic_queries': bool(plan.get('synthetic')), 'seconds': time.monotonic() - started,
           'peak_reserved_bytes': torch.cuda.max_memory_reserved()})


def score(out, plan):
    import torch
    names = plan['job_names']
    candidates = json.loads(verified_result(out.parent, names['candidates'], 'candidates.json').read_text())
    model = Reranker(plan, verified_result(out.parent, names['train'], 'reranker-adapter.safetensors'))
    model.base.model.eval(); scores = {}; started = time.monotonic(); pairs_done = 0
    with torch.inference_mode():
        for n, (qid, row) in enumerate(candidates.items()):
            values = []
            for i in range(0, len(row['videos']), plan['score_batch']):
                chunk = row['videos'][i:i + plan['score_batch']]
                values += model.logits([(row['text'], v) for v in chunk]).cpu().numpy().tolist()
            scores[qid] = dict(zip(row['videos'], values)); pairs_done += len(row['videos'])
            if n % 50 == 0:
                atomic_json(out / 'progress.json', {'queries': n + 1, 'total': len(candidates), 'pairs_per_second': pairs_done / (time.monotonic() - started)})
    seconds = time.monotonic() - started
    atomic_json(out / 'results/reranker-scores.json', scores)
    rate = pairs_done / seconds
    finish(out, plan, {'kind': 'fine-tuned reranker scores for held-video candidates', 'gate_passed': True, 'accuracy_score': False,
           'pairs': pairs_done, 'seconds': seconds, 'pairs_per_second': rate,
           'projected_test_hours_h100': plan['test_queries'] * plan['top_k'] / rate / 3600})


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
