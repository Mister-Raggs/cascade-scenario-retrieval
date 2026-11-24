"""Track 3: LoRA fine-tuned Qwen3-VL-8B-Instruct answering per-atom yes/no questions (fitting-fold labels only).

Jobs: 'train' (GPU) and 'score' (GPU: probabilities for every atom on the held fold's videos).
Held-video probabilities feed the structured scorer directly, so no probe is ever fitted on in-sample outputs.
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
from baseline.atom_questions import question
from baseline.query_struct import atom_present
from baseline.track_data import atom_question_batches
from scripts.feature_retrieval import atomic_json
from scripts.post_submission import stage_directory, verified_result
from scripts.query_side import finish
from scripts.reranker_finetune import fold_data
from scripts.vqa_attributes import vocabulary, yes_no_ids


class Answerer:
    def __init__(self, plan, adapter=None):
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        from baseline.h100_models import FrameStore
        from baseline.lora import attach_language_lora, load_adapter_state
        path = ROOT / plan['vlm_directory']
        if json.loads((path / 'model_provenance.json').read_text())['revision'] != plan['vlm_revision']:
            raise ValueError('Pinned VLM revision differs')
        self.processor = AutoProcessor.from_pretrained(str(path), local_files_only=True); self.processor.tokenizer.padding_side = 'left'
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(str(path), dtype=torch.bfloat16, attn_implementation='sdpa',
                                                                     local_files_only=True).cuda()
        attach_language_lora(self.model, plan['rank'], plan['alpha'])
        if adapter is not None:
            from safetensors.torch import load_file
            load_adapter_state(self.model, load_file(str(adapter)))
        self.frames = FrameStore(ROOT / plan['frame_directory'], plan['frames'])
        self.yes, self.no = yes_no_ids(self.processor.tokenizer); self.plan = plan

    def logits(self, vid, atoms):
        import torch
        from baseline.h100_models import visual_batch
        frames, meta = self.frames.get(vid)
        conversations = [[{'role': 'user', 'content': [
            {'type': 'video', 'video': frames, 'resized_height': 256, 'resized_width': 448, 'min_pixels': 4 * 32 * 32},
            {'type': 'text', 'text': question(a) + ' Answer with yes or no.'}]}] for a in atoms]
        batch, _ = visual_batch(self.processor, conversations, [meta] * len(atoms), max_length=self.plan['max_length'])
        last = self.model(**{k: v.cuda() for k, v in batch.items()}).logits[:, -1, :].float()
        return torch.logsumexp(last[:, self.yes], 1) - torch.logsumexp(last[:, self.no], 1)


def labels(plan, vids):
    atoms = vocabulary(); events = load_events(ROOT / plan['annotation_directory'])
    sets = {v: {e[0] for e in events[v]['events']} for v in vids}
    return atoms, {a: np.asarray([atom_present(a, sets[v]) for v in vids]) for a in atoms}


def train(out, plan):
    import torch
    from safetensors.torch import save_file
    from baseline.lora import adapter_state
    _, _, _, vids, ix = fold_data(plan); atoms, truth = labels(plan, vids)
    rng = np.random.default_rng(plan['seed']); fit = np.asarray(ix['fit_v'])
    model = Answerer(plan); params = [p for p in model.model.parameters() if p.requires_grad]
    frozen = {n: p._version for n, p in model.model.named_parameters() if not p.requires_grad}
    initial = adapter_state(model.model); model.model.train()
    opt = torch.optim.AdamW(params, lr=plan['learning_rate'], weight_decay=.01); losses = []; started = time.monotonic()
    for step in range(plan['optimizer_steps']):
        if step % len(fit) == 0:
            examples = atom_question_batches(rng.permutation(fit), truth, atoms, rng, plan['questions_per_video'])
        vid_index, items = examples[step % len(fit)]
        y = torch.tensor([float(l) for _, l in items], device='cuda')
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model.logits(vids[vid_index], [a for a, _ in items]), y)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite VLM atom loss')
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1); opt.step()
        losses.append(float(loss.detach()))
        if torch.cuda.max_memory_reserved() >= plan['max_reserved_gib'] * 1024 ** 3:
            raise ValueError('VLM fine-tune exceeded its memory budget')
        if (step + 1) % 25 == 0:
            atomic_json(out / 'progress.json', {'optimizer_steps': step + 1, 'total': plan['optimizer_steps'],
                        'recent_loss': float(np.mean(losses[-25:])), 'elapsed_seconds': time.monotonic() - started})
    final = adapter_state(model.model)
    if not any(not torch.equal(v, initial[n]) for n, v in final.items()) or \
       not all(p._version == frozen[n] for n, p in model.model.named_parameters() if n in frozen):
        raise ValueError('Adapter/base-weight correctness gate failed')
    save_file(final, str(out / 'results/vlm-adapter.safetensors'))
    finish(out, plan, {'kind': 'LoRA fine-tuned VLM atom answerer on fitting-fold videos', 'gate_passed': True, 'accuracy_score': False,
           'losses': losses, 'held_videos_in_training': 0, 'seconds': time.monotonic() - started,
           'peak_reserved_bytes': torch.cuda.max_memory_reserved()})


def score(out, plan):
    import torch
    _, _, _, vids, ix = fold_data(plan); atoms, truth = labels(plan, vids)
    model = Answerer(plan, verified_result(out.parent, plan['job_names']['train'], 'vlm-adapter.safetensors')); model.model.eval()
    held = list(ix['held_v']); probs = np.zeros((len(held), len(atoms)), np.float32); started = time.monotonic()
    with torch.inference_mode():
        for n, i in enumerate(held):
            parts = [model.logits(vids[i], atoms[j:j + plan['score_batch']]).cpu().numpy() for j in range(0, len(atoms), plan['score_batch'])]
            probs[n] = 1 / (1 + np.exp(-np.concatenate(parts)))
            if n % 16 == 0:
                atomic_json(out / 'progress.json', {'videos': n + 1, 'total': len(held), 'elapsed_seconds': time.monotonic() - started})
    np.savez_compressed(out / 'results/vlm-atom-probs.npz', video_ids=np.asarray([vids[i] for i in held]), atoms=np.asarray(atoms), probs=probs)
    from baseline.probe import auc
    rows = [{'atom': a, 'held_auc': auc(probs[:, j], truth[a][held]), 'positives': int(truth[a][held].sum())} for j, a in enumerate(atoms)]
    valid = [r['held_auc'] for r in rows if r['positives'] >= 5]
    finish(out, plan, {'kind': 'fine-tuned VLM atom probabilities on held videos', 'gate_passed': True, 'accuracy_score': True,
           'held_fold': plan['held_fold'], 'mean_held_auc': float(np.nanmean(valid)), 'atoms': rows,
           'seconds_per_video': (time.monotonic() - started) / len(held)})


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
