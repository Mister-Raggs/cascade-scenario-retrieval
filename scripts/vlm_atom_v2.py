"""Bigger-VLM atom answerer (2026-10-19): LoRA fine-tuned Qwen3-VL (dense or MoE) on per-atom yes/no questions.

Each clip's video prefix is encoded once and its key/value cache is shared by every question about that clip, so the
cost per question is a few text tokens. The first clip of every job proves the split is exact: prefix + suffix token ids
must equal the ids of the unsplit prompt, and (when scoring) shared-prefix logits must match full-prompt logits.

Jobs: 'train' and 'score' (GPU), restricted to the plan's atom families (agents, lights, objects by default; ego-motion
atoms are left to the motion experiments). Fitting-fold labels only; held-video probabilities feed the evaluator.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from baseline.atom_questions import question
from baseline.track_data import atom_question_batches
from scripts.feature_retrieval import atomic_json
from scripts.post_submission import stage_directory, verified_result
from scripts.query_side import finish
from scripts.reranker_finetune import fold_data
from scripts.vlm_atom_finetune import labels
from scripts.vqa_attributes import yes_no_ids

MARK = '<<QUESTION>>'
SUFFIX = ' Answer with yes or no.'


def plan_atoms(plan, atoms):
    return [a for a in atoms if a.split('|')[0] in plan['atom_families']]


def routed_experts_forward(self, hidden_states, routing_weights, router_indices):
    """Exact MoE expert mixture over routed experts only, indexing expert weights by integer (views, never copies).

    The library's training loop indexes weights with tensors, which copies and saves each expert's weights for backward;
    its inference path multiplies every token by every expert. Both give the same values as this loop.
    """
    import torch
    batch = hidden_states.shape[0]; h = hidden_states.reshape(-1, self.hidden_size); out = torch.zeros_like(h)
    flat = router_indices.reshape(-1); order = torch.argsort(flat, stable=True); top_k = router_indices.shape[-1]
    counts = torch.bincount(flat, minlength=self.num_experts).tolist(); tokens = order // top_k; start = 0
    for e, c in enumerate(counts):
        if not c:
            continue
        idx = tokens[start:start + c]; start += c
        gate, up = (h[idx] @ self.gate_up_proj[e]).chunk(2, dim=-1)
        out.index_add_(0, idx, ((up * self.act_fn(gate)) @ self.down_proj[e]) * routing_weights[idx, e, None])
    return out.view(batch, -1, self.hidden_size)


def patch_experts(model):
    import types
    found = [m for m in model.modules() if type(m).__name__.endswith('TextExperts')]
    for m in found:
        m.forward = types.MethodType(routed_experts_forward, m)
    return len(found)


def repeat_cache(cache, n):
    """Batch-repeat a one-row shared prefix cache n times. Attention-only caches use batch_repeat_interleave; caches with
    linear-attention layers (no such method) are expanded by reorder_cache with an all-zero index, an out-of-place
    index_select over keys/values and conv/recurrent states alike."""
    import torch
    layers = getattr(cache, 'layers', [])
    if all(hasattr(layer, 'batch_repeat_interleave') for layer in layers):
        cache.batch_repeat_interleave(n)
    else:
        cache.reorder_cache(torch.zeros(n, dtype=torch.long))
    return cache


def patch_linear_cache():
    """Linear-attention cache layers update their states in place (fixed addresses for CUDA graphs, which we do not use).
    Training through a shared prefix needs out-of-place updates so autograd keeps the prefix states it saved; the logic is
    otherwise identical to the library's. Returns False where the library has no linear-attention cache."""
    try:
        import torch
        from transformers.cache_utils import LinearAttentionLayer
    except ImportError:
        return False

    def update_conv_state(self, conv_states, state_idx=0, conv_kernel_size=None, **kwargs):
        if not self.is_conv_states_initialized[state_idx]:
            self.lazy_initialization(conv_states=conv_states, state_idx=state_idx, conv_kernel_size=conv_kernel_size)
        if not self.has_previous_state[state_idx]:
            full = conv_states; self.has_previous_state[state_idx] = True
            if not self.record_past and full.shape[-1] < self.conv_kernel_size[state_idx]:
                full = torch.nn.functional.pad(full, (self.conv_kernel_size[state_idx] - full.shape[-1], 0), value=0)
        else:
            full = torch.cat([self.conv_states[state_idx], conv_states], dim=-1)
        self.conv_states[state_idx] = full if self.record_past else full[..., -self.conv_kernel_size[state_idx]:]
        return full

    def update_recurrent_state(self, recurrent_states, state_idx=0, **kwargs):
        if not self.is_recurrent_states_initialized[state_idx]:
            self.lazy_initialization(recurrent_states=recurrent_states, state_idx=state_idx)
        self.recurrent_states[state_idx] = recurrent_states
        return recurrent_states

    LinearAttentionLayer.update_conv_state = update_conv_state
    LinearAttentionLayer.update_recurrent_state = update_recurrent_state
    return True


class SharedPrefixAnswerer:
    def __init__(self, plan, adapter=None):
        import torch
        import transformers
        from transformers import AutoProcessor
        from baseline.h100_models import FrameStore
        from baseline.lora import attach_language_lora, load_adapter_state
        path = ROOT / plan['vlm_directory']
        if json.loads((path / 'model_provenance.json').read_text())['revision'] != plan['vlm_revision']:
            raise ValueError('Pinned VLM revision differs')
        self.processor = AutoProcessor.from_pretrained(str(path), local_files_only=True)
        self.model = getattr(transformers, plan['vlm_class']).from_pretrained(
            str(path), dtype=torch.bfloat16, attn_implementation='sdpa', local_files_only=True).cuda()
        if plan.get('lora_target'):  # architectures whose attention projections differ (for example hybrid linear attention)
            import re
            from baseline.lora import attach_lora
            attach_lora(self.model, re.compile(plan['lora_target']), plan['rank'], plan['alpha'])
        else:
            attach_language_lora(self.model, plan['rank'], plan['alpha'])
        if adapter is not None:
            from safetensors.torch import load_file
            load_adapter_state(self.model, load_file(str(adapter)))
        self.frames = FrameStore(ROOT / plan['frame_directory'], plan['frames'], plan.get('frame_width', 640))
        self.yes, self.no = yes_no_ids(self.processor.tokenizer); self.plan = plan
        self.pad = self.processor.tokenizer.pad_token_id
        self.post = None; self.checked = False; self.template = plan.get('template_kwargs') or {}
        self.routed = patch_experts(self.model); self.linear_cache = patch_linear_cache()

    def mode(self, training):
        self.model.train(training)

    def suffix_inputs(self, ids):
        """Extra model inputs for the question pass (none for Qwen models)."""
        return {}

    def conversation(self, frames, text):
        return [{'role': 'user', 'content': [
            {'type': 'video', 'video': frames, 'resized_height': self.plan['height'], 'resized_width': self.plan['width'],
             'min_pixels': 4 * 32 * 32},
            {'type': 'text', 'text': text}]}]

    def prefix(self, vid):
        from baseline.h100_models import visual_batch
        frames, meta = self.frames.get(vid)
        batch, _ = visual_batch(self.processor, [self.conversation(frames, MARK)], [meta], self.plan['max_length'], prefix_marker=MARK,
                                template_kwargs=self.template)
        if self.post is None:
            text = self.processor.apply_chat_template([self.conversation(frames, MARK)], tokenize=False, add_generation_prompt=True, **self.template)[0]
            self.post = text.split(MARK)[1]
        return batch, frames, meta

    def suffix_ids(self, atom):
        return self.processor.tokenizer.encode(question(atom) + SUFFIX + self.post, add_special_tokens=False)

    def check_split(self, vid, batch, frames, meta, atom):
        """Token-level proof that prefix + suffix equals the unsplit prompt."""
        from baseline.h100_models import visual_batch
        full, _ = visual_batch(self.processor, [self.conversation(frames, question(atom) + SUFFIX)], [meta], self.plan['max_length'],
                               template_kwargs=self.template)
        joined = batch['input_ids'][0].tolist() + self.suffix_ids(atom)
        if full['input_ids'][0].tolist() != joined:
            raise ValueError('Shared-prefix tokenization differs from the full prompt: ' + vid)
        return full

    def logits(self, vid, atoms):
        import torch
        batch, frames, meta = self.prefix(vid)
        if not self.checked:
            self.check_split(vid, batch, frames, meta, atoms[0]); self.checked = True
        out = self.model(**{k: v.cuda() for k, v in batch.items()}, use_cache=True, logits_to_keep=1)
        cache, start = out.past_key_values, batch['input_ids'].shape[1]
        size = self.plan['question_batch']; parts = []
        for j in range(0, len(atoms), size):
            chunk = [self.suffix_ids(a) for a in atoms[j:j + size]]
            shared = cache if len(atoms) <= size else copy.deepcopy(cache)
            repeat_cache(shared, len(chunk))
            width = max(map(len, chunk)); lengths = torch.tensor([len(c) for c in chunk], device='cuda')
            ids = torch.tensor([c + [self.pad] * (width - len(c)) for c in chunk], device='cuda')
            # Right padding: real tokens attend only to the prefix and earlier real tokens under the causal mask.
            o = self.model(input_ids=ids, past_key_values=shared, use_cache=True,
                           cache_position=torch.arange(start, start + width, device='cuda'), **self.suffix_inputs(ids))
            last = o.logits[torch.arange(len(chunk), device='cuda'), lengths - 1].float()
            parts.append(torch.logsumexp(last[:, self.yes], 1) - torch.logsumexp(last[:, self.no], 1))
        return torch.cat(parts)

    def full_logits(self, vid, atoms):
        """Unshared, unpadded reference path (one full prompt per question) for the equivalence check."""
        import torch
        from baseline.h100_models import visual_batch
        frames, meta = self.frames.get(vid); values = []
        for a in atoms:
            batch, _ = visual_batch(self.processor, [self.conversation(frames, question(a) + SUFFIX)], [meta], self.plan['max_length'],
                                    template_kwargs=self.template)
            last = self.model(**{k: v.cuda() for k, v in batch.items()}, logits_to_keep=1).logits[:, -1, :].float()
            values.append(torch.logsumexp(last[:, self.yes], 1) - torch.logsumexp(last[:, self.no], 1))
        return torch.cat(values)

class GemmaAnswerer(SharedPrefixAnswerer):
    """Shared-prefix scorer for Gemma 4 (2026-10-21). Frames go through Gemma4VideoProcessor without resampling, with the
    true source frame indices and fps as metadata (Gemma writes an mm:ss timestamp before each frame) and a fixed per-frame
    token budget (`video_soft_tokens`, one of 70/140/280/560/1120). Vision tokens attend bidirectionally inside their
    frame block, controlled by mm_token_type_ids; question tokens are text (type 0)."""

    def chat(self, text):
        return [{'role': 'user', 'content': [{'type': 'video'}, {'type': 'text', 'text': text}]}]

    def encode(self, text, frames, meta):
        from transformers.video_utils import VideoMetadata
        decoder = meta['decoder']
        md = VideoMetadata(total_num_frames=len(frames), fps=float(decoder['fps']), frames_indices=list(decoder['requested_source_indices']),
                           duration=float(meta['duration']))
        batch = self.processor(text=[text], videos=[frames], video_metadata=[md], return_tensors='pt',
                               videos_kwargs={'do_sample_frames': False, 'max_soft_tokens': self.plan['video_soft_tokens']})
        if batch['input_ids'].shape[1] > self.plan['max_length']:
            raise ValueError('Context cap exceeded; never truncate visual tokens')
        return batch

    def template_text(self, text):
        return self.processor.apply_chat_template(self.chat(text), tokenize=False, add_generation_prompt=True, **self.template)

    def prefix(self, vid):
        frames, meta = self.frames.get(vid)
        pre, post = self.template_text(MARK).split(MARK)
        if self.post is None:
            self.post = post
        return self.encode(pre, frames, meta), frames, meta

    def check_split(self, vid, batch, frames, meta, atom):
        full = self.encode(self.template_text(question(atom) + SUFFIX), frames, meta)
        if full['input_ids'][0].tolist() != batch['input_ids'][0].tolist() + self.suffix_ids(atom):
            raise ValueError('Shared-prefix tokenization differs from the full prompt: ' + vid)
        return full

    def full_logits(self, vid, atoms):
        import torch
        frames, meta = self.frames.get(vid); values = []
        for a in atoms:
            batch = self.encode(self.template_text(question(a) + SUFFIX), frames, meta)
            last = self.model(**{k: v.cuda() for k, v in batch.items()}, logits_to_keep=1).logits[:, -1, :].float()
            values.append(torch.logsumexp(last[:, self.yes], 1) - torch.logsumexp(last[:, self.no], 1))
        return torch.cat(values)

    def suffix_inputs(self, ids):
        import torch
        return {'mm_token_type_ids': torch.zeros_like(ids)}


def answerer_for(plan, adapter=None):
    """Scorer class for the plan's model family."""
    return (GemmaAnswerer if plan['vlm_class'].startswith('Gemma') else SharedPrefixAnswerer)(plan, adapter)


def train(out, plan):
    import torch
    from safetensors.torch import save_file
    from baseline.lora import adapter_state
    parent, _, _, vids, ix = fold_data(plan); vocab, truth = labels(plan, vids); atoms = plan_atoms(plan, vocab)
    rng = np.random.default_rng(plan['seed']); fit = np.asarray(ix['fit_v'])
    if plan.get('training_folds') is not None:  # learning-curve plans train on a subset of the fitting folds
        keep = {v for f in plan['training_folds'] for v in parent['development_video_groups'][f]}
        fit = np.asarray([i for i in fit if vids[i] in keep])
    model = answerer_for(plan); params = [p for p in model.model.parameters() if p.requires_grad]
    frozen = {n: p._version for n, p in model.model.named_parameters() if not p.requires_grad}
    initial = adapter_state(model.model); model.mode(True)
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
    finish(out, plan, {'kind': f"LoRA fine-tuned {plan['vlm_repo']} atom answerer (shared video prefix)", 'gate_passed': True,
           'accuracy_score': False, 'losses': losses, 'atoms': len(atoms), 'held_videos_in_training': 0,
           'seconds': time.monotonic() - started, 'peak_reserved_bytes': torch.cuda.max_memory_reserved()})


def score(out, plan):
    import torch
    from baseline.probe import auc
    _, _, _, vids, ix = fold_data(plan); vocab, truth = labels(plan, vids); atoms = plan_atoms(plan, vocab)
    model = answerer_for(plan, verified_result(out.parent, plan['job_names']['train'], 'vlm-adapter.safetensors')); model.mode(False)
    held = list(ix['held_v']); probs = np.zeros((len(held), len(atoms)), np.float32)
    with torch.inference_mode():
        probe = atoms[:4]; shared = model.logits(vids[held[0]], probe).cpu().numpy(); full = model.full_logits(vids[held[0]], probe).cpu().numpy()
        drift = float(np.abs(shared - full).max())
        if drift > plan['max_prefix_logit_drift']:
            raise ValueError(f'Shared-prefix logits differ from full prompts by {drift:.3f}')
        started = time.monotonic()
        for n, i in enumerate(held):
            probs[n] = 1 / (1 + np.exp(-model.logits(vids[i], atoms).cpu().numpy()))
            if n % 16 == 0:
                atomic_json(out / 'progress.json', {'videos': n + 1, 'total': len(held), 'elapsed_seconds': time.monotonic() - started})
    seconds = (time.monotonic() - started) / len(held)
    np.savez_compressed(out / 'results/vlm-atom-probs.npz', video_ids=np.asarray([vids[i] for i in held]), atoms=np.asarray(atoms), probs=probs)
    rows = [{'atom': a, 'held_auc': auc(probs[:, j], truth[a][held]), 'positives': int(truth[a][held].sum())} for j, a in enumerate(atoms)]
    valid = [r['held_auc'] for r in rows if r['positives'] >= 5]
    finish(out, plan, {'kind': f"fine-tuned {plan['vlm_repo']} atom probabilities on held videos", 'gate_passed': True,
           'accuracy_score': True, 'held_fold': plan['held_fold'], 'audit_used': plan['held_fold'] == 'audit', 'mean_held_auc': float(np.nanmean(valid)), 'atoms': rows,
           'prefix_logit_drift': drift, 'seconds_per_video_h200': seconds,
           'projected_test_hours_a100': seconds * plan['test_videos'] * plan['a100_slowdown'] / 3600,
           'projection_assumptions': f"{plan['test_videos']} test videos, A100 {plan['a100_slowdown']}x slower than H200 (assumed)"})


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
