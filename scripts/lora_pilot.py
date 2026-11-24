"""Prepare and profile training-only language LoRA; no accuracy run or promotion."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from baseline.io import sha256_file, unique_rows
from baseline.lora_data import check_partitions, select_queries, sample_batch, FOCUS
from scripts.feature_retrieval import atomic_json


def inputs(plan):
    for name, digest in plan['input_hashes'].items():
        if sha256_file(Path(name)) != digest:
            raise ValueError('Frozen pilot input changed: ' + name)
    parent = json.loads(Path(plan['partition_parent']).read_text())
    queries = unique_rows(Path('data/retrieval/cascade-v1.0/train/queries.jsonl'), 'query_id')
    qrels = unique_rows(Path('data/retrieval/cascade-v1.0/train/qrels.jsonl'), 'query_id')
    check_partitions(parent, qrels)
    if parent['training_qrels_sha256'] != sha256_file(Path('data/retrieval/cascade-v1.0/train/qrels.jsonl')):
        raise ValueError('Partition labels differ')
    return parent, queries, qrels


def finish(out, plan, report):
    report.update(validation_used=False, audit_used=False, accuracy_score=False, official_evaluation=False,
                  code_sha256=sha256_file(Path(__file__)))
    atomic_json(out / 'results/report.json', report)
    subprocess.run([sys.executable, 'scripts/spaces.py', 'upload', '--source', str(out / 'results'),
                    '--prefix', plan['backup_prefix'] + '/' + out.name + '/results', '--receipt', str(out / 'receipt.json')], check=True)
    atomic_json(out / 'status.json', {'state': 'complete', 'backup_verified': True,
                'gate_passed': report['gate_passed'], 'report_sha256': sha256_file(out / 'results/report.json')})
    print(json.dumps(report), flush=True)


def prepare(out, plan):
    parent, queries, qrels = inputs(plan)
    held = set(parent['development_video_groups'][plan['held_fold']])
    fitting = sorted(vid for i, group in enumerate(parent['development_video_groups'])
                     if i != plan['held_fold'] for vid in group)
    audit = {vid for group in parent['audit_video_groups'] for vid in group}
    if set(fitting) & (held | audit):
        raise ValueError('Held/audit videos cannot enter fitting')
    selected = select_queries(queries, qrels, parent['query_groups']['fit'], fitting,
                              plan['fitting_query_cap'], plan['seed'])
    review = json.loads(Path('configs/h100_actor_event_review_v1.json').read_text())
    cases = {c['id']: c for c in json.loads(Path(review['cases_path']).read_text())['cases']}
    excluded = {}
    for decision in review['decisions']:
        if not decision['usable_for_actor_event_diagnostic']:
            case = cases[decision['case_id']]
            excluded.setdefault(case['query_id'], set()).update(v['video_id'] for v in case['videos'])
    excluded = {qid: sorted(vids) for qid, vids in excluded.items()}
    groups = {}
    for size in plan['query_batches']:
        rng = np.random.default_rng(plan['seed'] + size)
        groups[str(size)] = [sample_batch(selected, fitting, qrels, query_batch=size, excluded=excluded, rng=rng)
                             for _ in range(plan['profile_microsteps'])]
    manifest = {'kind': 'training-only LoRA profiling inputs', 'fitting_query_ids': selected,
                'fitting_video_ids': fitting, 'held_video_ids': sorted(held), 'audit_video_ids': sorted(audit),
                'profile_batches': groups, 'ambiguous_negative_exclusions': excluded,
                'input_hashes': plan['input_hashes'], 'plan_sha256': sha256_file(Path(plan['plan_path'])),
                'training_query_focus': 'Heuristic actor/order wording; published qrels supply all positives. No synthetic negatives or new annotation targets.',
                'evaluation_protocol': 'Future accuracy comparison must re-encode adapted fitting and held pixels, then fit the same E4 head separately for adapted and frozen controls; never reuse an all-training head.'}
    atomic_json(out / 'results/input-manifest.json', manifest)
    finish(out, plan, {'kind': 'LoRA profiling preparation', 'gate_passed': True,
           'fitting_queries': len(selected), 'fitting_query_positive_set_groups': len(selected),
           'focused_wording_queries': sum(bool(FOCUS.search(queries[q]['text'])) for q in selected),
           'fitting_videos': len(fitting), 'held_videos': len(held), 'audit_videos_excluded': len(audit),
           'synthetic_negatives_created': 0, 'input_manifest_sha256': sha256_file(out / 'results/input-manifest.json')})


class TrainingEncoder:
    def __init__(self, plan):
        from baseline.native_video import NativeEncoder
        from baseline.h100_models import FrameStore
        recipe = json.loads(Path(plan['native_recipe']).read_text())
        if recipe['attention_backend'] != 'math':
            raise ValueError('Pilot requires the reproducible math attention recipe')
        if plan.get('model_provenance_sha256') and sha256_file(Path(plan['model_directory']) / 'model_provenance.json') != plan['model_provenance_sha256']:
            raise ValueError('Pinned encoder provenance changed')
        self.native = NativeEncoder(Path(plan.get('model_directory', 'artifacts/models/embed-8b')), Path('vendor/qwen3-vl-embedding/qwen3_vl_embedding.py'), recipe, 'cuda')
        self.e = self.native.embedder
        self.model = self.e.model
        self.frames = FrameStore(Path(plan.get('frame_directory', 'runs/h100-2026-10-11/frames')), plan.get('frames', 4))
        # The helper subsamples frame lists to num_frames/max_frames; keep every decoded frame and its pixel budget.
        self.e.num_frames = self.e.max_frames = self.frames.count
        self.e.total_pixels = self.frames.count * 256 * 448

    def batch_inputs(self, batch, queries):
        video, info = self.video_inputs(batch['video_ids'])
        query = self.query_inputs([queries[qid]['text'] for qid in batch['query_ids']])
        return query, video, info

    def video_inputs(self, video_ids):
        from baseline.native_video import DOC_INSTRUCTION, QUERY_INSTRUCTION
        from baseline.h100_models import visual_batch
        conversations, metadata = [], []
        for vid in video_ids:
            # Profiling must reuse verified frames; an absent cache does not trigger new decoding.
            if not (self.frames.directory / vid / 'metadata.json').is_file():
                raise ValueError('Verified source frame cache missing: ' + vid)
            images, meta = self.frames.get(vid)
            c = self.e.format_model_input(video=images, instruction=DOC_INSTRUCTION)
            c[1]['content'][0].update(resized_height=256, resized_width=448, min_pixels=4*32*32)
            conversations.append(c); metadata.append(meta)
        video, info = visual_batch(self.e.processor, conversations, metadata)
        if any(grid[0] != self.frames.count // 2 for grid in info['video_grids']):
            raise ValueError('Processed temporal grid does not match the decoded frame count')
        return video, info

    def query_inputs(self, texts):
        from baseline.native_video import QUERY_INSTRUCTION
        qconversations = [self.e.format_model_input(text=text, instruction=QUERY_INSTRUCTION) for text in texts]
        text = self.e.processor.apply_chat_template(qconversations, tokenize=False, add_generation_prompt=True)
        query = self.e.processor(text=text, images=None, videos=None, padding=True,
                                 truncation=False, do_resize=False, return_tensors='pt')
        if query['attention_mask'].sum(1).max() > 2048:
            raise ValueError('Query token budget exceeded')
        return query

    def encode(self, batch):
        import torch
        from torch.nn import functional as F
        tensors = {k: v.to(self.model.device) for k, v in batch.items()}
        with self.native.attention_context():
            hidden = self.model(**tensors, use_cache=False).last_hidden_state
            return F.normalize(self.e._pooling_last(hidden, tensors['attention_mask']).float(), dim=-1)


def profile(out, plan):
    import torch
    from safetensors.torch import save_file
    from baseline.lora import attach_language_lora, adapter_state, load_adapter_state, multi_positive_loss
    parent, queries, qrels = inputs(plan)
    previous = out.parent / 'prepare-lora'
    receipt = json.loads((previous / 'receipt.json').read_text())
    manifest_path = previous / 'results/input-manifest.json'
    if sha256_file(manifest_path) != receipt['files']['input-manifest.json']['sha256']:
        raise ValueError('Prepared profiling inputs changed')
    manifest = json.loads(manifest_path.read_text())
    allowed = [tuple(c) for c in plan.get('allowed_capabilities', [[9, 0]])]
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in allowed:
        raise ValueError('GPU capability is outside the profile plan')
    if not torch.cuda.is_bf16_supported() or 'sm_80' not in torch.cuda.get_arch_list():
        raise ValueError('BF16 and A100-targeted framework binaries are required')
    if torch.cuda.get_device_properties(0).total_memory < plan.get('minimum_gpu_total_gib', 0) * 1024 ** 3:
        raise ValueError('GPU is below the plan memory floor')
    torch.set_num_threads(2);torch.manual_seed(plan['seed'])
    started = time.monotonic();encoder = TrainingEncoder(plan)
    first = manifest['profile_batches'][str(plan['query_batches'][0])][0]
    qinput, vinput, _ = encoder.batch_inputs(first, queries)
    with torch.no_grad():
        original_q = encoder.encode(qinput).detach().clone()
        original_v = encoder.encode(vinput).detach().clone()
    targets = attach_language_lora(encoder.model, plan['rank'], plan['alpha'])
    with torch.no_grad():
        no_op_drift = max(float((encoder.encode(qinput) - original_q).abs().max()),
                          float((encoder.encode(vinput) - original_v).abs().max()))
    if no_op_drift > 1e-6:
        raise ValueError('Zero-initialized LoRA changed original embeddings')
    initial = adapter_state(encoder.model)
    frozen = {name: p._version for name, p in encoder.model.named_parameters() if not p.requires_grad}
    parameters = [p for p in encoder.model.parameters() if p.requires_grad]
    if not parameters or any(not name.endswith(('.lora_A', '.lora_B')) for name, p in encoder.model.named_parameters() if p.requires_grad):
        raise ValueError('Unexpected trainable backbone parameter')
    encoder.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    rows = []
    for size in plan['query_batches']:
        load_adapter_state(encoder.model, initial)
        encoder.model.eval();references = [];prepared = [];prep_seconds = 0.0
        for batch in manifest['profile_batches'][str(size)]:
            prep_start = time.monotonic();qi, vi, info = encoder.batch_inputs(batch, queries);prep_seconds += time.monotonic()-prep_start
            with torch.no_grad():
                references.append((encoder.encode(qi).detach().cpu(), encoder.encode(vi).detach().cpu()))
            prepared.append((batch, qi, vi, info))
        encoder.model.train();encoder.model.visual.eval()
        torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        optimizer = torch.optim.AdamW(parameters, lr=plan['learning_rate'], weight_decay=.01)
        optimizer.zero_grad();seconds, losses, gradient_norms = [], [], []
        for index, ((batch, qi, vi, info), (ref_q, ref_v)) in enumerate(zip(prepared, references, strict=True)):
            torch.cuda.synchronize();begin = time.monotonic()
            with encoder.native.attention_context():
                q = encoder.encode(qi);v = encoder.encode(vi)
                positive = torch.tensor(batch['positive_mask'], dtype=torch.bool, device='cuda')
                negative = torch.tensor(batch['negative_mask'], dtype=torch.bool, device='cuda')
                contrast = multi_positive_loss(q @ v.T / plan['temperature'], positive, negative)
                anchor = ((q-ref_q.to('cuda')).square().sum(1).mean()+(v-ref_v.to('cuda')).square().sum(1).mean())/2
                loss = contrast + plan['anchor_weight'] * anchor
                if not torch.isfinite(loss):raise ValueError('Nonfinite training profile loss')
                (loss/plan['gradient_accumulation']).backward()
            if (index+1) % plan['gradient_accumulation'] == 0:
                norm = torch.nn.utils.clip_grad_norm_(parameters, 1)
                if not torch.isfinite(norm) or norm <= 0:raise ValueError('Adapter gradient missing/nonfinite')
                gradient_norms.append(float(norm));optimizer.step();optimizer.zero_grad()
            torch.cuda.synchronize();seconds.append(time.monotonic()-begin);losses.append(float(loss.detach()))
            atomic_json(out/'progress.json', {'query_batch':size, 'video_batch':size*2, 'completed_microsteps':index+1,
                        'total_microsteps':plan['profile_microsteps'], 'last_seconds':seconds[-1]})
            del q,v,loss,contrast,anchor
        changed = any(not torch.equal(value, initial[name]) for name,value in adapter_state(encoder.model).items())
        untouched = all(p._version == frozen[name] and p.grad is None for name,p in encoder.model.named_parameters() if name in frozen)
        peak = torch.cuda.max_memory_reserved();steady = float(np.mean(seconds[plan['warmup_microsteps']:]))
        row = {'query_batch':size, 'video_batch':size*2, 'microstep_seconds':seconds,
               'steady_microstep_seconds':steady, 'input_preparation_seconds':prep_seconds,
               'query_examples_per_compute_second':size/steady, 'peak_reserved_bytes':peak,
               'losses':losses, 'gradient_norms':gradient_norms, 'adapter_changed':changed, 'frozen_weights_unchanged':untouched,
               'projected_200_optimizer_steps_seconds':200*plan['gradient_accumulation']*steady,
               'gate_passed':changed and untouched and peak < plan['max_reserved_gib']*1024**3}
        rows.append(row)
        save_file(adapter_state(encoder.model),str(out/f'results/profile-only-query-batch-{size}.safetensors'))
        if not row['gate_passed']:raise ValueError('Training profile correctness/memory gate failed')
        del optimizer;gc.collect();torch.cuda.empty_cache()
    selected = max(rows,key=lambda r:r['query_examples_per_compute_second'])
    finish(out,plan,{'kind':'language LoRA training profile only','device':torch.cuda.get_device_name(0),
        'capability':list(torch.cuda.get_device_capability()),'gate_passed':True,'rank':plan['rank'],'alpha':plan['alpha'],
        'target_modules':targets,'trainable_parameters':sum(p.numel() for p in parameters),
        'zero_adapter_max_absolute_drift':no_op_drift,'profiles':rows,'selected_query_batch':selected['query_batch'],
        'seconds_including_load_input_preparation_and_reference_encoding':time.monotonic()-started,
        'input_manifest_sha256':sha256_file(manifest_path),'model_provenance_sha256':sha256_file(Path('artifacts/models/embed-8b/model_provenance.json')),
        'limits':'Six microsteps per size, fitting-only pixels/qrels. Checkpoints are profiling fixtures, not accuracy candidates. Projection-head fitting and excluded-video accuracy are not run or promoted.'})


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--job',required=True,choices=['prepare-lora','profile-lora']);parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();os.chdir(ROOT);os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    plan=json.loads(args.plan.read_text());args.out.mkdir(parents=True,exist_ok=True);(args.out/'results').mkdir(exist_ok=True)
    {'prepare-lora':prepare,'profile-lora':profile}[args.job](args.out,plan)


if __name__=='__main__':main()
