"""Experiment B: zero-shot yes/no attribute scores from a generative VLM (Qwen3-VL-8B-Instruct), GPU.

Score = logsumexp(yes-token logits) - logsumexp(no-token logits) at the answer position. A pilot on a
few clips measures throughput and gates the full development run.
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
from baseline.atom_labels import query_atoms, clip_atom_matrix
from baseline.atom_questions import question
from baseline.io import sha256_file, unique_rows
from baseline.probe import auc, cross_validated
from scripts.egomotion_run import development_videos
from scripts.feature_retrieval import atomic_json, load_vectors
from scripts.post_submission import stage_directory, verified_result
from scripts.query_side import finish


def vocabulary():
    texts = [r['text'] for r in unique_rows(ROOT / 'data/retrieval/cascade-v1.0/train/queries.jsonl', 'query_id').values()]
    return query_atoms(texts)


def yes_no_ids(tokenizer):
    def ids(words):
        out = sorted({tokenizer.encode(w, add_special_tokens=False)[0] for w in words})
        if not out:
            raise ValueError('Missing answer tokens')
        return out
    return ids(['yes', 'Yes', ' yes', ' Yes']), ids(['no', 'No', ' no', ' No'])


def yes_no_score(logits, yes, no):
    import torch
    return (torch.logsumexp(logits[:, yes].float(), 1) - torch.logsumexp(logits[:, no].float(), 1)).cpu().numpy()


class Scorer:
    def __init__(self, plan):
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        from baseline.h100_models import FrameStore
        path = ROOT / plan['vlm_directory']
        provenance = json.loads((path / 'model_provenance.json').read_text())
        if provenance['revision'] != plan['vlm_revision']:
            raise ValueError('Pinned VLM revision differs')
        self.processor = AutoProcessor.from_pretrained(str(path), local_files_only=True)
        self.processor.tokenizer.padding_side = 'left'
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(str(path), dtype=torch.bfloat16, attn_implementation='sdpa',
                                                                     local_files_only=True).cuda().eval()
        self.frames = FrameStore(ROOT / plan['frame_directory'], plan['frames'])
        self.yes, self.no = yes_no_ids(self.processor.tokenizer); self.plan = plan

    def clip(self, vid, questions):
        import torch
        from baseline.h100_models import visual_batch
        frames, meta = self.frames.get(vid); scores = []
        for i in range(0, len(questions), self.plan['question_batch']):
            chunk = questions[i:i + self.plan['question_batch']]
            conversations = [[{'role': 'user', 'content': [
                {'type': 'video', 'video': frames, 'resized_height': 256, 'resized_width': 448, 'min_pixels': 4 * 32 * 32},
                {'type': 'text', 'text': q + ' Answer with yes or no.'}]}] for q in chunk]
            batch, _ = visual_batch(self.processor, conversations, [meta] * len(chunk), max_length=self.plan['max_length'])
            with torch.no_grad():
                logits = self.model(**{k: v.cuda() for k, v in batch.items()}).logits[:, -1, :]
            scores.append(yes_no_score(logits, self.yes, self.no))
        return np.concatenate(scores)


def score_clips(out, plan, vids):
    atoms = vocabulary(); questions = [question(a) for a in atoms]; scorer = Scorer(plan)
    features = np.zeros((len(vids), len(atoms)), np.float32); started = time.monotonic()
    for i, vid in enumerate(vids):
        features[i] = scorer.clip(vid, questions)
        if not np.isfinite(features[i]).all():
            raise ValueError('Nonfinite VLM score: ' + vid)
        atomic_json(out / 'progress.json', {'completed': i + 1, 'total': len(vids), 'elapsed_seconds': time.monotonic() - started})
    np.savez_compressed(out / 'results/vqa-features.npz', video_ids=np.asarray(vids), atoms=np.asarray(atoms), features=features)
    return atoms, features, time.monotonic() - started


def pilot(out, plan):
    vids = development_videos(plan)[:plan['pilot_clips']]
    atoms, features, seconds = score_clips(out, plan, vids)
    per_clip = seconds / len(vids); projected = per_clip * len(development_videos(plan)) / 3600
    varied = bool(np.median(features.std(0)) > 1e-3)
    finish(out, plan, {'kind': 'VLM yes/no attribute pilot', 'accuracy_score': False,
           'gate_passed': bool(varied and projected <= plan['max_full_hours']), 'seconds_per_clip': per_clip,
           'projected_full_hours': projected, 'questions_per_clip': len(atoms), 'scores_vary': varied})


def full(out, plan):
    vids = development_videos(plan)
    atoms, features, seconds = score_clips(out, plan, vids)
    truth = clip_atom_matrix(load_events(ROOT / plan['annotation_directory']), vids, atoms)
    fm = json.loads((ROOT / plan['feature_directory'] / 'feature_manifest.json').read_text())
    index = {v: i for i, v in enumerate(fm['video_ids'])}
    pooled = load_vectors(ROOT / plan['feature_directory'] / 'aggregate.npz')[[index[v] for v in vids]]
    folds = np.random.default_rng(plan['seed']).permutation(len(vids)) % plan['folds']
    rows = []
    for j, atom in enumerate(atoms):
        y = truth[:, j]
        if y.sum() < plan['minimum_positive_clips']:
            continue
        rows.append({'atom': atom, 'positives': int(y.sum()), 'vqa_zero_shot_auc': auc(features[:, j], y),
                     'pooled_probe_auc': auc(cross_validated(pooled, y, folds), y)})
    gain = float(np.mean([r['vqa_zero_shot_auc'] - r['pooled_probe_auc'] for r in rows]))
    finish(out, plan, {'kind': 'VLM zero-shot atom scores versus pooled-embedding probes', 'accuracy_score': True,
           'gate_passed': gain >= plan['minimum_auc_gain'], 'mean_auc_gain': gain, 'atoms': rows, 'seconds': seconds})


JOBS = {'pilot': pilot, 'full': full}


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
    torch.set_num_threads(plan['cpu_threads'])
    stage_directory(args.out)
    JOBS[args.job](args.out, plan)


if __name__ == '__main__':
    main()
