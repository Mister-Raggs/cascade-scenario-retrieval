"""Stage (and optionally build with --no-push) the audit-confirmed entry-3 package. Never publishes.

Recipe: the submitted native 8B raw/head ranking rank-fused at weight 0.8 with t4 atoms (block probes + ego motion CNN,
parser v2) late-fused with fine-tuned Qwen3.8-27B atom answers (adapter from the audit training run, all 1,549
development clips). Two Python environments share the image: the native one (unchanged entry-2 pins) and the Qwen3.8 one
(transformers 5.18, flash-linear-attention, causal-conv1d built with sm_80). The Dockerfile puts large stable layers first
and splits the Qwen3.8 shards over three layers (about 17 GB each).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from baseline.entry3_recipe import FUSION_WEIGHT
from baseline.io import sha256_file, write_json
from baseline.structured_recipe import MOTION, WINDOWS
from scripts.package_local import BASE_IMAGE
from scripts.package_native_local import stage as stage_native
from scripts.post_submission import verified_result

EXTRA_MODULES = ['egomotion.py', 'windows.py', 'structured_recipe.py', 'structured_run.py', 'query_parser.py', 'query_parser_v2.py',
                 'query_frames.py', 'query_struct.py', 'dsl_aliases.json', 'entry3_recipe.py', 'entry3_run.py', 'entry3_vlm.py',
                 'h100_models.py', 'atom_questions.py', 'lora.py', 'entry4_scorer.py',
                 'query_parser_v3.py', 'query_pairs.py', 'property_atoms.py', 'annotation_events.py']
VLM_PLAN = 'configs/audit-q38/vlm.json'
AUDIT = 'runs/audit-v2'
WHEEL = 'causal_conv1d-1.7.0-cp312-cp312-linux_x86_64.whl'
# Recorded causal-conv1d 1.7.0 builds (nvcc 13.1, torch 2.14.1+cu130), each checked to contain sm_75-sm_121 including sm_80.
WHEELS = {'f47e7d6d4574b2b543ed5b4e632c3be3b52437d79c784a1ad623459386eaa795': 'entry-3 build, 2026-10-21',
          'fe888ca659f7b20cefc1e13b9e7a4d88617620e6ec780f8a88109bc18a52ed38': 'rebuild on the 2026-10-22 H200'}
# Exact pins of the verified research Qwen3.8 environment (.venv-q38), minus storage SDKs (boto3 and its dependencies).
Q38_REQUIREMENTS = '''--extra-index-url https://download.pytorch.org/whl/cu130
annotated-doc==0.0.5
anyio==4.15.1
av==19.0.1
certifi==2026.10.19
charset-normalizer==3.5.2
click==8.5.0
cuda-bindings==13.3.1
cuda-pathfinder==1.6.0
cuda-toolkit==13.0.3.0
einops==0.8.2
filelock==3.32.3
fla-core==0.5.2
flash-linear-attention==0.5.2
fsspec==2026.7.0
h11==0.16.0
hf-xet==1.7.0
httpcore2==2.15.1
httpx2==2.15.1
huggingface-hub==2.2.0
idna==3.20
jinja2==3.1.6
markdown-it-py==4.2.0
markupsafe==3.0.3
mdurl==0.1.2
mpmath==1.3.0
networkx==3.6.1
ninja==1.13.2
numpy==2.5.3
nvidia-cublas==13.1.1.3
nvidia-cuda-cupti==13.0.85
nvidia-cuda-nvrtc==13.0.88
nvidia-cuda-runtime==13.0.96
nvidia-cudnn-cu13==9.24.0.43
nvidia-cufft==12.0.0.61
nvidia-cufile==1.15.1.6
nvidia-curand==10.4.0.35
nvidia-cusolver==12.0.4.66
nvidia-cusparse==12.6.3.3
nvidia-cusparselt-cu13==0.8.1
nvidia-nccl-cu13==2.30.7
nvidia-nvjitlink==13.3.33
nvidia-nvshmem-cu13==3.4.5
nvidia-nvtx==13.0.85
packaging==26.3
pillow==12.3.0
pygments==2.21.0
pyyaml==6.0.3
qwen-vl-utils==0.0.14
regex==2026.9.29
requests==2.34.2
rich==15.0.0
safetensors==0.8.0
setuptools==78.1.0
shellingham==1.5.4
sympy==1.14.0
tokenizers==0.23.3
torch==2.14.1+cu130
torchvision==0.29.1+cu130
tqdm==4.70.1
transformers==5.18.0
triton==3.8.0
truststore==0.10.5
typer==0.27.3
typing-extensions==4.16.0
urllib3==2.8.0
'''
SHARD_GROUPS = [range(1, 7), range(7, 13), range(13, 19)]


def link(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copyfile(source, target)


def stage_vlm_model(context, model_path, revision):
    provenance = json.loads((model_path / 'model_provenance.json').read_text())
    if provenance['revision'] != revision or provenance['declared_license'] != 'apache-2.0':
        raise ValueError('Qwen3.8 revision or licence differs')
    index = json.loads((model_path / 'model.safetensors.index.json').read_text())
    shards = sorted(set(index['weight_map'].values())); names = []
    for name, digest in provenance['files'].items():
        if name.startswith('model-') and name.endswith('.safetensors') and name not in shards:
            continue  # a shard emptied by scripts/quantize_q38.py (only dropped tensors); the index does not reference it
        relative = PurePosixPath(name)
        if relative.is_absolute() or '..' in relative.parts or any(p.startswith('.') for p in relative.parts) or name in {'README.md', 'crc32.txt'}:
            continue
        p = model_path / name
        if p.is_symlink() or not p.is_file() or (p.suffix.lower() not in {'.json', '.txt', '.jinja', '.safetensors'} and name != 'LICENSE'):
            raise ValueError('Unapproved Qwen3.8 asset: ' + name)
        if sha256_file(p) != digest:
            raise ValueError('Pinned Qwen3.8 asset changed: ' + name)
        link(p, context / 'vlm-model' / name); names.append(name)
    if not set(shards) <= set(names) or len(shards) not in (17, 18):
        raise ValueError('Missing Qwen3.8 shards')
    link(model_path / 'model_provenance.json', context / 'vlm-model/model_provenance.json')
    return shards


def dockerfile(shards, int8=False):
    groups = [' '.join(f'vlm-model/{s}' for s in shards if int(s.split('-')[1]) in g) for g in SHARD_GROUPS]
    copies = '\n'.join(f'COPY {g} /opt/app/vlm-model/' for g in groups)
    extras = 'COPY vlm-model/int8_scales.safetensors vlm-model/lm_head_rows.safetensors /opt/app/vlm-model/\n' if int8 else ''
    return f'''ARG BASE_IMAGE={BASE_IMAGE}
FROM ${{BASE_IMAGE}}
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libgomp1 && rm -rf /var/lib/apt/lists/*
RUN python -m venv /opt/app/.venv
COPY requirements.txt /opt/app/requirements.txt
RUN /opt/app/.venv/bin/pip install --no-cache-dir -r /opt/app/requirements.txt && /opt/app/.venv/bin/pip check
COPY requirements-q38.txt /opt/app/requirements-q38.txt
COPY wheels /opt/app/wheels
RUN python -m venv /opt/app/.venv-q38 && /opt/app/.venv-q38/bin/pip install --no-cache-dir --no-deps -r /opt/app/requirements-q38.txt /opt/app/wheels/{WHEEL} && /opt/app/.venv-q38/bin/pip check
COPY model /opt/app/model
{copies}
COPY vlm-model/*.json vlm-model/*.jinja vlm-model/*.txt vlm-model/LICENSE /opt/app/vlm-model/
{extras}COPY checkpoint /opt/app/checkpoint
COPY components /opt/app/components
COPY vlm /opt/app/vlm
COPY vendor /opt/app/vendor
COPY baseline /opt/app/baseline
COPY native_recipe.json run.py LICENSE NOTICE THIRD_PARTY.md /opt/app/
COPY third_party_licenses /opt/app/third_party_licenses
'''


def stage(context, model_path, checkpoint, training_manifest, features, components, vlm_model, wheel, int8=False, scorer=None, adapter=None, all_clips=False,
          verifier_adapter=None):
    manifest = stage_native(context, model_path, checkpoint, training_manifest, features, 'h1004')
    for name in EXTRA_MODULES:
        shutil.copyfile(PROJECT / 'baseline' / name, context / 'baseline' / name)
    comp = json.loads((components / 'entry3_manifest.json').read_text())
    if not comp['parity']['passed'] or (comp['audit_videos_used_for_fitting'] and not all_clips) or comp['atoms_sha256'] != sha256_file(components / 'entry3_atoms.npz'):
        raise ValueError('Entry-3 components did not pass parity, used audit clips without --all-clips, or changed')
    (context / 'components').mkdir()
    for name in ['entry3_atoms.npz', 'entry3_manifest.json']:
        shutil.copyfile(components / name, context / 'components' / name)
    plan = json.loads((PROJECT / VLM_PLAN).read_text())
    adapter = verified_result(*(adapter or (PROJECT / AUDIT, 'audit-vlm-train')), 'vlm-adapter.safetensors')
    probs = json.loads(verified_result(PROJECT / AUDIT, 'audit-vlm-score', 'report.json').read_text())
    import numpy as np
    atoms = [str(a) for a in np.load(verified_result(PROJECT / AUDIT, 'audit-vlm-score', 'vlm-atom-probs.npz'))['atoms']]
    if len(atoms) != len(probs['atoms']):
        raise ValueError('Audit atom list differs from its report')
    (context / 'vlm').mkdir(); shutil.copyfile(adapter, context / 'vlm/vlm-adapter.safetensors')
    shards = stage_vlm_model(context, vlm_model, plan['vlm_revision'])
    if sha256_file(wheel) not in WHEELS:
        raise ValueError('causal-conv1d wheel is not a recorded build')
    (context / 'wheels').mkdir(); shutil.copyfile(wheel, context / 'wheels' / WHEEL)
    (context / 'requirements-q38.txt').write_text(Q38_REQUIREMENTS)
    recipe = json.loads((context / 'native_recipe.json').read_text())
    recipe['entry3'] = {
        'atoms': 'components/entry3_atoms.npz', 'atoms_sha256': comp['atoms_sha256'], 'fusion_weight': FUSION_WEIGHT,
        'motion': MOTION, 'windows': WINDOWS, 'motion_workers': 4, 'vlm_python': '.venv-q38/bin/python',
        'parser': 'baseline/query_parser_v2.py', 'ego_atoms': comp['ego_atoms'],
        'confirmation': 'audit confirmation 2026-10-21 (second look at the audit set): RP 0.2026 versus 0.1431 for entry 2 and '
                        '0.1903 for the audited 30B, both head seeds; development folds 0-2 RP 0.2769 at weight 0.8'}
    recipe['vlm'] = {
        'model_directory': 'vlm-model', 'model_class': plan['vlm_class'], 'model_id': plan['vlm_repo'], 'model_revision': plan['vlm_revision'],
        'adapter': 'vlm/vlm-adapter.safetensors', 'adapter_sha256': sha256_file(context / 'vlm/vlm-adapter.safetensors'),
        'lora_target': plan['lora_target'], 'rank': plan['rank'], 'alpha': plan['alpha'], 'frames': plan['frames'], 'frame_width': 640,
        'height': plan['height'], 'width': plan['width'], 'max_length': plan['max_length'], 'template_kwargs': plan['template_kwargs'],
        'question_batch': 16, 'decode_workers': 6, 'atoms': atoms,
        **({'int8_scales': 'int8_scales.safetensors', 'lm_head_rows': 'lm_head_rows.safetensors',
            'weights': 'int8 per-row language-model linear weights (dequantized to bf16 at load), mtp weights dropped, lm_head reduced to the yes/no rows'} if int8 else {}),
        'note': 'question_batch 16 (research 96) keeps Qwen3.8 scoring near 64 GiB peak, inside 80 GB; padding and the causal mask make answers independent of batch'}
    if scorer is not None:  # entry 4: the audit-confirmed linear learned scorer
        sm = json.loads((scorer / 'manifest.json').read_text())
        if not sm['passed'] or sm['weights_sha256'] != sha256_file(scorer / 'linear-scorer.npz'):
            raise ValueError('Entry-4 scorer did not pass parity or changed')
        for name in ('linear-scorer.npz', 'manifest.json'):
            shutil.copyfile(scorer / name, context / 'components' / ('entry4-' + name))
        recipe['entry4'] = {'weights': 'components/entry4-linear-scorer.npz', 'weights_sha256': sm['weights_sha256'], 'dimension': sm['dimension'],
                            'training': sm['training']}
        if sm['dimension'] == 25:  # entry 5: the bundle plus 32-frame atoms and Qwen3.8 property answers, with a time guard
            recipe['entry4'].update(parser='v3', properties=True,
                                    verifier={'k': sm.get('verifier_k', 30), 'candidates': 'entry-3 recipe ranking (parser v2, fusion weight 0.8) with the 32-frame atom answers',
                                              'question': 'baseline/query_pairs.py verifier_question'},
                                    confirmation=sm.get('confirmation', 'see ENTRY5 records'))
            recipe['vlm'].update(frames=32, verifier_frames=16, question_batch=sm.get('question_batch', 8), properties=sm['properties'],
                                 guard=sm['guard'],
                                 note=('atoms and property questions on 32 frames, the verifier on 16; question batch kept small for the 80 GB '
                                       'card (answers do not depend on the batch); time guard: see baseline/entry3_vlm.py'))
            if sm.get('verifier', '').startswith('fine-tuned'):
                if verifier_adapter is None:
                    raise ValueError('The scorer expects the fine-tuned verifier adapter')
                shutil.copyfile(verifier_adapter, context / 'vlm/verifier-adapter.safetensors')
                recipe['vlm'].update(verifier_adapter='vlm/verifier-adapter.safetensors',
                                     verifier_adapter_sha256=sha256_file(context / 'vlm/verifier-adapter.safetensors'))
        elif sm['dimension'] == 23:  # the bundle: parser v3 and the zero-shot whole-query verifier on each query's top 30
            recipe['entry4'].update(parser='v3', verifier={'k': 30, 'candidates': 'entry-3 recipe ranking (parser v2, fusion weight 0.8)',
                                                           'question': 'baseline/query_pairs.py verifier_question'},
                                    confirmation='audit confirmation 2026-10-22 (fourth look at the audit set): RP 0.2480 versus 0.2286 for '
                                                 'the linear scorer and 0.2021 for the entry-3 recipe, both head seeds')
        else:
            recipe['entry4']['confirmation'] = ('audit confirmation 2026-10-22 (third look at the audit set): RP 0.2286 versus 0.2021 for the '
                                                'entry-3 recipe, both head seeds; folds 0-2 RP 0.2918 versus 0.2774')
    if all_clips:
        recipe['all_clips'] = ('Qwen3.8 adapter, t4 components and the scorer trained on all 2,066 annotated clips (learning-curve rule, '
                               '2026-10-22: one fold 0.2646/0.2656 versus two folds 0.2806/0.2863 on fold 0); no held-out set remains')
    write_json(context / 'native_recipe.json', recipe)
    (context / 'run.py').write_text("from baseline.entry3_run import main\n\nif __name__ == '__main__':\n    main()\n")
    (context / 'Dockerfile').write_text(dockerfile(shards, int8))
    submission = (context / 'submission.yaml').read_text()
    submission = submission.replace('  OMP_NUM_THREADS: "2"\n', '  OMP_NUM_THREADS: "2"\n  PYTORCH_CUDA_ALLOC_CONF: "expandable_segments:True"\n')
    submission = submission.replace('notes: "Native video embedding plus shared residual adapter; local-only user handoff"',
                                    'notes: "Native 8B ranking fused with t4 atoms and fine-tuned Qwen3.8-27B atom answers; local-only user handoff"')
    (context / 'submission.yaml').write_text(submission)
    (context / '.dockerignore').write_text('.cascade-build\n__pycache__\nHANDOFF.md\npackage_manifest.json\nvlm-model/README.md\n')
    with open(context / 'THIRD_PARTY.md', 'a') as f:
        f.write('''
Entry-3 additions (2026-10-21):

- **Qwen3.8-27B**: `Qwen/Qwen3.8-27B`, revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, Apache-2.0; its LICENSE file is
  included unchanged in `vlm-model/LICENSE`. A LoRA adapter (rank 32) fine-tuned on development clips is in `vlm/`.
'''
        + ('''  The packaged weights are derived from that revision (scripts/quantize_q38.py): language-model linear weights stored
  as int8 with per-row scales, the unused multi-token-prediction weights removed and lm_head reduced to the yes/no rows;
  `vlm-model/model_provenance.json` records the source files and the transform.
''' if int8 else '') + '''
- **Qwen3.8 Python environment** (`.venv-q38`): Transformers 5.18.0 (Apache-2.0), flash-linear-attention / fla-core 0.5.2
  (MIT), causal-conv1d 1.7.0 (BSD-3-Clause, built from its source distribution with nvcc 13.1 for sm_75-sm_121), einops
  (MIT) and the pinned dependencies in `requirements-q38.txt`. Installed distributions keep their own licences.
''')
    files = {str(p.relative_to(context)): sha256_file(p) for p in sorted(context.rglob('*')) if p.is_file() and p.name != 'package_manifest.json'}
    manifest.update(kind='local-entry3-build-context', recipe=recipe, files=files)
    write_json(context / 'package_manifest.json', manifest)
    (context / 'HANDOFF.md').write_text('''# Entry-3 build context

Unbuilt or locally built (--no-push) context for the audit-confirmed entry-3 recipe. Trained weights only (8B residual
head, atom probes, ego motion CNN and calibration, Qwen3.8 LoRA adapter); no annotations, qrels, query text, challenge
videos, known-ID vectors or credentials. Offline verification must pass before handoff.

The user alone publishes the artifact, accepts agreements or queues official evaluation.
''')
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--model-path', type=Path, default=Path('artifacts/models/embed-8b'))
    ap.add_argument('--checkpoint', type=Path, default=Path('runs/finalist-canonical-2026-10-11-v2/canonical-refit/results/embed-8b/checkpoint'))
    ap.add_argument('--training-manifest', type=Path, default=Path('runs/finalist-canonical-2026-10-11-v2/canonical-refit/results/embed-8b/training_manifest.json'))
    ap.add_argument('--features', type=Path, default=Path('runs/finalist-canonical-2026-10-11-v2/canonical-refit/results/embed-8b/features'))
    ap.add_argument('--components', type=Path, default=Path('runs/entry3/components'))
    ap.add_argument('--vlm-model', type=Path, default=Path('artifacts/models/qwen3.8-27b'))
    ap.add_argument('--wheel', type=Path, required=True)
    ap.add_argument('--int8', action='store_true', help='--vlm-model is the scripts/quantize_q38.py output')
    ap.add_argument('--entry4-scorer', type=Path, help='runs/entry4/scorer (scripts/fit_entry4_scorer.py output)')
    ap.add_argument('--adapter-run', type=Path, help='run directory holding the Qwen3.8 adapter job (default: the audit run)')
    ap.add_argument('--adapter-job', default='audit-vlm-train'); ap.add_argument('--all-clips', action='store_true', help='components/adapter fitted on all annotated clips')
    ap.add_argument('--verifier-adapter', type=Path, help='entry 5: the fine-tuned verifier adapter (runs/entry5/mtv-all-train results)')
    ap.add_argument('--build', action='store_true', help='Build local layers using --no-push; never publish')
    a = ap.parse_args()
    result = stage(a.out, a.model_path, a.checkpoint, a.training_manifest, a.features, a.components, a.vlm_model, a.wheel, a.int8, a.entry4_scorer,
                   (a.adapter_run, a.adapter_job) if a.adapter_run else None, a.all_clips, a.verifier_adapter)
    print(f"Staged {len(result['files'])} files at {a.out}")
    if a.build:
        from scripts.package_local import DEFAULT_DEVKIT
        subprocess.run([sys.executable, str(DEFAULT_DEVKIT / 'docker-prep-kit/build_submission.py'), '--context', str(a.out.resolve()),
                        '--base-image', BASE_IMAGE, '--image-tag', 'entry3-submission:local',
                        '--output-dir', str(a.out.resolve().parent / (a.out.name + '-artifact')), '--no-push'], check=True)


if __name__ == '__main__':
    main()
