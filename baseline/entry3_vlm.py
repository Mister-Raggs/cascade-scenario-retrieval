"""Entry-3 stage B: fine-tuned Qwen3.8-27B yes/no atom answers for every input video (runs in the Qwen3.8 environment).

Same prompts, frames (16 source-timed frames decoded 640 wide, resized to 352x640), shared video prefix and yes-minus-no
log-odds as the audited research scorer (scripts/vlm_atom_v2.py). The first clip proves the prefix split token by token;
the shared-prefix versus full-prompt logit drift on that clip is recorded, not gated.
Entry 5: the atom pass may use more frames than the verifier (recipe 'frames' versus 'verifier_frames'), appends the
zero-shot property questions (recipe 'properties'), and obeys a time guard (recipe 'guard'): if the atom pass is projected
to end after guard['atoms_switch_seconds'] of the run (measured from ENTRY5_START), the remaining clips are decoded with
guard['fallback_frames']; the verifier stops at guard['verifier_stop_seconds'] and the remaining pairs stay unanswered.
ENTRY5_GUARD_SCALE (verification only) scales both deadlines so the fallbacks can be forced in tests.
Usage: python -m baseline.entry3_vlm --recipe R --videos LIST.json --out PROBS.npz
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import json
import os
from pathlib import Path
import re
import time

import numpy as np

from .atom_questions import question
from .h100_models import visual_batch
from .io import sha256_file, write_json
from .native_video import decode_source_times

MARK = '<<QUESTION>>'
SUFFIX = ' Answer with yes or no.'


def yes_no_ids(tokenizer):
    def ids(words):
        out = sorted({tokenizer.encode(w, add_special_tokens=False)[0] for w in words})
        if not out:
            raise ValueError('Missing answer tokens')
        return out
    return ids(['yes', 'Yes', ' yes', ' Yes']), ids(['no', 'No', ' no', ' No'])


def repeat_cache(cache, n):
    """Batch-repeat a one-row prefix cache; linear-attention layers are expanded by reorder_cache with an all-zero index."""
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


class AtomAnswerer:
    def __init__(self, root, recipe):
        import torch
        import transformers
        from safetensors.torch import load_file
        from transformers import AutoProcessor
        from .lora import attach_lora, load_adapter_state
        self.recipe = recipe; model_path = root / recipe['model_directory']
        self.processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True)
        self.model = getattr(transformers, recipe['model_class']).from_pretrained(
            str(model_path), dtype=torch.bfloat16, attn_implementation='sdpa', local_files_only=True).cuda()
        self.yes, self.no = yes_no_ids(self.processor.tokenizer)
        if recipe.get('int8_scales'):
            self.dequantize(load_file(str(model_path / recipe['int8_scales'])))
        if recipe.get('lm_head_rows'):
            self.answer_head(load_file(str(model_path / recipe['lm_head_rows'])))
        attach_lora(self.model, re.compile(recipe['lora_target']), recipe['rank'], recipe['alpha'])
        load_adapter_state(self.model, load_file(str(root / recipe['adapter'])))
        self.model.eval(); patch_linear_cache()
        self.pad = self.processor.tokenizer.pad_token_id; self.template = recipe['template_kwargs']; self.post = None

    def use_adapter(self, path):
        """Swap in another LoRA state of the same shape (entry 5: the fine-tuned verifier's adapter for the verifier stage)."""
        from safetensors.torch import load_file
        from .lora import load_adapter_state
        load_adapter_state(self.model, load_file(str(path))); self.model.eval()

    def dequantize(self, scales):
        """int8 rows (loaded into bf16 parameters, where every int8 value is exact) times their per-row scale, in bf16."""
        import torch
        params = dict(self.model.named_parameters())
        if not set(scales) <= set(params):
            raise ValueError('int8 scales name parameters the model does not have')
        with torch.no_grad():
            for key, s in scales.items():
                p = params[key]
                if p.dtype != torch.bfloat16 or p.abs().max() > 127 or not torch.equal(p, p.round()):
                    raise ValueError('Expected int8-valued weights before dequantization: ' + key)
                p.copy_((p.float() * s.to(p.device)[:, None]).to(p.dtype))

    def answer_head(self, rows):
        """Keep only the yes/no answer rows of lm_head: their logits, and so the yes-minus-no log-odds, are unchanged."""
        import torch
        ids = rows['token_ids'].tolist()
        if not set(self.yes + self.no) <= set(ids):
            raise ValueError('Answer-token rows do not cover the yes/no tokens')
        head = torch.nn.Linear(rows['weight'].shape[1], len(ids), bias=False, dtype=torch.bfloat16, device='cuda')
        with torch.no_grad():
            head.weight.copy_(rows['weight'])
        self.model.lm_head = head
        self.yes, self.no = [ids.index(t) for t in self.yes], [ids.index(t) for t in self.no]

    def conversation(self, frames, text):
        return [{'role': 'user', 'content': [
            {'type': 'video', 'video': frames, 'resized_height': self.recipe['height'], 'resized_width': self.recipe['width'],
             'min_pixels': 4 * 32 * 32},
            {'type': 'text', 'text': text}]}]

    def prefix(self, frames, meta):
        batch, _ = visual_batch(self.processor, [self.conversation(frames, MARK)], [meta], self.recipe['max_length'], prefix_marker=MARK,
                                template_kwargs=self.template)
        if self.post is None:
            text = self.processor.apply_chat_template([self.conversation(frames, MARK)], tokenize=False, add_generation_prompt=True, **self.template)[0]
            self.post = text.split(MARK)[1]
        return batch

    def suffix_ids(self, atom):
        """An atom key, or ('text', question) for a free-text question such as the whole-query verifier."""
        text = atom[1] if isinstance(atom, tuple) else question(atom)
        return self.processor.tokenizer.encode(text + SUFFIX + self.post, add_special_tokens=False)

    def check_split(self, frames, meta, atom):
        batch = self.prefix(frames, meta)
        full, _ = visual_batch(self.processor, [self.conversation(frames, question(atom) + SUFFIX)], [meta], self.recipe['max_length'],
                               template_kwargs=self.template)
        if full['input_ids'][0].tolist() != batch['input_ids'][0].tolist() + self.suffix_ids(atom):
            raise ValueError('Shared-prefix tokenization differs from the full prompt')

    def logits(self, frames, meta, atoms):
        import torch
        batch = self.prefix(frames, meta)
        out = self.model(**{k: v.cuda() for k, v in batch.items()}, use_cache=True, logits_to_keep=1)
        cache, start = out.past_key_values, batch['input_ids'].shape[1]
        size = self.recipe['question_batch']; parts = []
        for j in range(0, len(atoms), size):
            chunk = [self.suffix_ids(a) for a in atoms[j:j + size]]
            shared = cache if len(atoms) <= size else copy.deepcopy(cache)
            repeat_cache(shared, len(chunk))
            width = max(map(len, chunk)); lengths = torch.tensor([len(c) for c in chunk], device='cuda')
            ids = torch.tensor([c + [self.pad] * (width - len(c)) for c in chunk], device='cuda')
            # Right padding: real tokens attend only to the prefix and earlier real tokens under the causal mask.
            o = self.model(input_ids=ids, past_key_values=shared, use_cache=True, cache_position=torch.arange(start, start + width, device='cuda'))
            last = o.logits[torch.arange(len(chunk), device='cuda'), lengths - 1].float()
            parts.append(torch.logsumexp(last[:, self.yes], 1) - torch.logsumexp(last[:, self.no], 1))
        return torch.cat(parts)

    def full_logits(self, frames, meta, atoms):
        import torch
        values = []
        for a in atoms:
            batch, _ = visual_batch(self.processor, [self.conversation(frames, question(a) + SUFFIX)], [meta], self.recipe['max_length'],
                                    template_kwargs=self.template)
            last = self.model(**{k: v.cuda() for k, v in batch.items()}, logits_to_keep=1).logits[:, -1, :].float()
            values.append(torch.logsumexp(last[:, self.yes], 1) - torch.logsumexp(last[:, self.no], 1))
        return torch.cat(values)


def decode(path, recipe, frames=None):
    stamps, frames, duration, _ = decode_source_times(path, frames or recipe['frames'], recipe['frame_width'])
    return frames, {'timestamps': stamps, 'duration': duration}


def verifier_pairs(native, vlm_atoms, vlm_probs, k):
    """Each query's top-k clips under the entry-3 recipe (parser v2, rank fusion at its weight), as clip -> [query index]."""
    from .entry3_recipe import fuse, structured_scores, with_vlm
    t4 = {str(a): native['t4_probs'][:, j] for j, a in enumerate(native['t4_atoms'])}
    vlm = {str(a): np.clip(vlm_probs[:, j], 1e-4, 1 - 1e-4) for j, a in enumerate(vlm_atoms)}
    texts = [str(t) for t in native['texts']]; scores, mask = structured_scores(texts, with_vlm(t4, vlm))
    ranking = fuse(native['raw'], native['submitted'], scores, mask, float(native['fusion_weight']))
    per = {}
    for i in range(len(texts)):
        for j in np.argsort(-ranking[i], kind='stable')[:k]:
            per.setdefault(int(j), []).append(i)
    return per, texts


def run(root, recipe_path, videos, out, progress=None, native=None):
    import torch
    recipe = json.loads(recipe_path.read_text())['vlm']
    if sha256_file(root / recipe['adapter']) != recipe['adapter_sha256']:
        raise ValueError('Qwen3.8 adapter changed')
    cap = os.environ.get('ENTRY3_GPU_MEMORY_GIB')  # verification only: emulate a smaller GPU
    if cap:
        torch.cuda.set_per_process_memory_fraction(min(1.0, float(cap) * 2 ** 30 / torch.cuda.get_device_properties(0).total_memory))
    began = time.perf_counter(); model = AtomAnswerer(root, recipe); loaded = time.perf_counter() - began
    atoms = recipe['atoms']; probs = np.zeros((len(videos), len(atoms)), np.float32); drift = None
    props = recipe.get('properties') or []; prop_probs = np.zeros((len(videos), len(props)), np.float32)
    if props:
        from .property_atoms import question as prop_question
        asked = atoms + [('text', prop_question(k)) for k in props]
    else:
        asked = atoms
    guard = recipe.get('guard') or {}; scale = float(os.environ.get('ENTRY5_GUARD_SCALE', '1'))
    start = float(os.environ.get('ENTRY5_START', time.time()))
    elapsed = lambda: time.time() - start
    frames_now = recipe['frames']; switched_at = None; per_clip = []
    with ThreadPoolExecutor(max_workers=recipe['decode_workers']) as pool, torch.inference_mode():
        pending = [pool.submit(decode, Path(p), recipe, frames_now) for p in videos[:recipe['decode_workers'] * 2]]; nxt = len(pending)
        for n in range(len(videos)):
            t0 = time.perf_counter(); frames, meta = pending.pop(0).result()
            if nxt < len(videos):
                pending.append(pool.submit(decode, Path(videos[nxt]), recipe, frames_now)); nxt += 1
            if n == 0:
                model.check_split(frames, meta, atoms[0])
                shared = model.logits(frames, meta, atoms[:4]).cpu().numpy(); full = model.full_logits(frames, meta, atoms[:4]).cpu().numpy()
                drift = float(np.abs(shared - full).max())
            z = model.logits(frames, meta, asked).cpu().numpy()
            probs[n] = 1 / (1 + np.exp(-z[:len(atoms)])); prop_probs[n] = 1 / (1 + np.exp(-z[len(atoms):]))
            per_clip.append(time.perf_counter() - t0)
            if guard and switched_at is None and n >= 8 and frames_now != guard['fallback_frames']:
                projected = elapsed() + np.mean(per_clip[-8:]) * (len(videos) - n - 1)
                if projected > guard['atoms_switch_seconds'] * scale:
                    frames_now = guard['fallback_frames']; switched_at = n + 1  # clips decoded from here on use the fallback
            if progress and (n % 16 == 0 or n == len(videos) - 1):
                progress(n + 1, len(videos))
    if not np.isfinite(probs).all() or not np.isfinite(prop_probs).all():
        raise ValueError('Nonfinite atom probabilities')
    np.savez(out, atoms=np.asarray(atoms), probs=probs, prop_atoms=np.asarray(props), prop_probs=prop_probs)
    report = {'videos': len(videos), 'atoms': len(atoms), 'properties': len(props), 'frames': recipe['frames'], 'load_seconds': loaded,
              'atom_seconds': time.perf_counter() - began, 'prefix_logit_drift': drift,
              'guard_atoms_switched_after_clips': switched_at, 'guard_scale': scale}
    verifier = json.loads(recipe_path.read_text()).get('entry4', {}).get('verifier')
    if verifier and native is not None:  # whole-query verifier on each query's top-k clips under the entry-3 recipe
        from .query_pairs import verifier_question
        if recipe.get('verifier_adapter'):  # entry 5: the fine-tuned verifier answers the whole-query questions
            if sha256_file(root / recipe['verifier_adapter']) != recipe['verifier_adapter_sha256']:
                raise ValueError('Verifier adapter changed')
            model.use_adapter(root / recipe['verifier_adapter'])
        started = time.perf_counter(); per, texts = verifier_pairs(np.load(native), atoms, probs, verifier['k'])
        qi, vi, lo = [], [], []; stopped_at = None; vframes = recipe.get('verifier_frames', recipe['frames'])
        with ThreadPoolExecutor(max_workers=recipe['decode_workers']) as pool, torch.inference_mode():
            clips = sorted(per); frames_of = pool.map(lambda j: decode(Path(videos[j]), recipe, vframes), clips)
            for n, (j, (frames, meta)) in enumerate(zip(clips, frames_of)):
                if guard and elapsed() > guard['verifier_stop_seconds'] * scale:
                    stopped_at = n; break
                z = model.logits(frames, meta, [('text', verifier_question(texts[i])) for i in per[j]]).cpu().numpy()
                qi += per[j]; vi += [j] * len(per[j]); lo += z.tolist()
                if progress and (n % 16 == 0 or n == len(clips) - 1):
                    progress(n + 1, len(clips), 'verifier')
        if not np.isfinite(lo).all():
            raise ValueError('Nonfinite verifier answers')
        np.savez(Path(out).with_name('verify.npz'), query_index=np.asarray(qi, int), clip_index=np.asarray(vi, int), logits=np.asarray(lo, np.float32))
        report.update(verifier_pairs=len(lo), verifier_clips=len(clips), verifier_frames=vframes, verifier_seconds=time.perf_counter() - started,
                      verifier_adapter=recipe.get('verifier_adapter'),
                      guard_verifier_stopped_at_clip=stopped_at)
    report.update(seconds=time.perf_counter() - began, peak_reserved_gib=torch.cuda.max_memory_reserved() / 2 ** 30,
                  device=torch.cuda.get_device_name(0))
    return report


def main():
    os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    import torch
    torch.set_num_threads(2); torch.manual_seed(0)
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recipe', type=Path, required=True); p.add_argument('--videos', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True); p.add_argument('--progress', type=Path)
    p.add_argument('--native', type=Path, help='stage-A outputs (raw/submitted similarities, t4 probabilities, query texts) for the verifier')
    a = p.parse_args()
    report = run(root, a.recipe, json.loads(a.videos.read_text()), a.out,
                 (lambda d, t, stage='vlm': write_json(a.progress, {'stage': stage, 'completed': d, 'total': t})) if a.progress else None,
                 a.native)
    write_json(a.out.with_suffix('.json'), report)
    print(f"Scored {report['videos']} videos x {report['atoms']} atoms in {report['seconds']:.1f}s", flush=True)


if __name__ == '__main__':
    main()
