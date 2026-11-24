"""Stage a verified native build context locally. Optional build always uses --no-push."""
from __future__ import annotations
import argparse,json,shutil,sys
from pathlib import Path,PurePosixPath
PROJECT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(PROJECT))
from baseline.io import sha256_file,write_json
from baseline.native_video import DOC_INSTRUCTION,QUERY_INSTRUCTION,validate_recipe
from scripts.package_local import BASE_IMAGE,local_build_command
from baseline.devkit import DEVKIT_COMMIT


def freeze_recipe(checkpoint,training_manifest,features,representation,model_path,helper):
    from safetensors import safe_open
    metadata=json.loads((checkpoint/'metadata.json').read_text());training=json.loads(training_manifest.read_text())
    if sha256_file(training_manifest)!=metadata['manifest_sha256']:raise ValueError('Checkpoint training manifest differs')
    if metadata.get('representation',representation)!=representation:raise ValueError('Checkpoint representation differs')
    weights=checkpoint/'adapter.safetensors'
    if sha256_file(weights)!=metadata['weights_sha256']:raise ValueError('Checkpoint weight hash differs')
    with safe_open(str(weights),framework='pt',device='cpu') as tensors:
        if set(tensors.keys())!={'down.weight','up.weight'}:raise ValueError('Unexpected adapter tensors')
        rank,dimension=tensors.get_slice('down.weight').get_shape()
        if tensors.get_slice('up.weight').get_shape()!=[dimension,rank]:raise ValueError('Adapter shape mismatch')
    if representation=='h1004':
        feature_manifest=features/'feature_manifest.json';feature=json.loads(feature_manifest.read_text())
        if training['feature_manifest_sha256']!=sha256_file(feature_manifest):raise ValueError('Checkpoint feature parent differs')
        if (feature['frames'],feature['height'],feature['width'],feature['actual_clock'],feature['dimension'])!=(4,256,448,True,dimension):raise ValueError('H100 feature recipe differs')
        if training['helper_sha256']!=sha256_file(helper):raise ValueError('Training helper differs')
        if feature['model_provenance_sha256']!=sha256_file(model_path/'model_provenance.json'):raise ValueError('Training model differs')
        spec=json.loads((model_path/'model_provenance.json').read_text())['spec']
        key='embed-8b' if dimension==4096 else 'qwen3-video'
        if spec!=json.loads((PROJECT/'configs/retrieval_models.json').read_text())[key]:raise ValueError('Pinned encoder differs')
        recipe={'representation':'h1004','frames':4,'height':256,'width':448,'dimension':dimension,'rank':rank,'max_length':2048,
            'document_instruction':DOC_INSTRUCTION,'query_instruction':QUERY_INSTRUCTION,'adapter_sha256':sha256_file(weights),
            'helper_sha256':sha256_file(helper),'model_id':spec['model_id'],'model_revision':spec['revision'],
            'model_provenance_sha256':sha256_file(model_path/'model_provenance.json'),'training_manifest_sha256':sha256_file(training_manifest),
            'feature_manifest_sha256':sha256_file(feature_manifest),'checkpoint_metadata_sha256':sha256_file(checkpoint/'metadata.json'),
            'raw_rank_weight':training['raw_rank_weight'],'rank_fusion_constant':60,'video_batch_size':4 if dimension==4096 else 2,'decode_workers':4,
            'attention_backend':feature.get('attention_backend','automatic')}
        validate_recipe(recipe);return recipe
    feature_manifest=features/'manifest.json';feature=json.loads(feature_manifest.read_text())
    parent=training.get('feature_manifest_sha256',training.get('feature_parents',{}).get(representation))
    if parent!=sha256_file(feature_manifest):raise ValueError('Checkpoint feature parent differs')
    extraction_path=features.parent/'experiment_manifest.json';extraction=json.loads(extraction_path.read_text())
    if sha256_file(extraction_path)!=feature['experiment_manifest_sha256']:raise ValueError('Feature extraction manifest differs')
    spec=json.loads((model_path/'model_provenance.json').read_text())['spec']
    expected=json.loads((PROJECT/'configs/retrieval_models.json').read_text())['qwen3-video']
    if spec!=expected:raise ValueError('Native pretrained model revision differs')
    if representation=='historical4':
        extra=extraction['extra']
        if (extra['frames'],extra['total_pixels'],extra['max_length'])!=(4,4*224*224,2048):raise ValueError('Historical feature recipe differs')
        for k,v in [('document_instruction',DOC_INSTRUCTION),('query_instruction',QUERY_INSTRUCTION)]:
            if extra[k]!=v:raise ValueError('Training instruction differs')
        expected_helper=extra['helper_sha256'];expected_model=extra['model_provenance_sha256']
    else:
        expected_helper=extraction['helper_sha256'];expected_model=extraction['model_provenance_sha256']
        variant=extraction['variant'];expected_variant='time4' if representation=='synthetic4' else representation
        if variant['id']!=expected_variant or (variant['height'],variant['width'])!=(256,448):raise ValueError('Temporal extraction recipe differs')
    if expected_helper!=sha256_file(helper) or expected_model!=sha256_file(model_path/'model_provenance.json'):raise ValueError('Model/helper differs from feature extraction')
    recipe={'representation':representation,'frames':8 if representation=='time8' else 4,'height':256,'width':448,
        'dimension':dimension,'rank':rank,'max_length':2048,'document_instruction':DOC_INSTRUCTION,'query_instruction':QUERY_INSTRUCTION,
        'adapter_sha256':sha256_file(weights),'helper_sha256':sha256_file(helper),'model_id':spec['model_id'],'model_revision':spec['revision'],
        'model_provenance_sha256':sha256_file(model_path/'model_provenance.json'),
        'training_manifest_sha256':sha256_file(training_manifest),'feature_manifest_sha256':sha256_file(feature_manifest),
        'feature_extraction_manifest_sha256':sha256_file(extraction_path),'checkpoint_metadata_sha256':sha256_file(checkpoint/'metadata.json')}
    validate_recipe(recipe);return recipe


def stage(context,model_path,checkpoint,training_manifest,features,representation='historical4'):
    if context.exists():raise ValueError('Choose a fresh local package directory')
    helper=PROJECT/'vendor/qwen3-vl-embedding/qwen3_vl_embedding.py'
    recipe=freeze_recipe(checkpoint,training_manifest,features,representation,model_path,helper)
    provenance=json.loads((model_path/'model_provenance.json').read_text());model_files=[]
    for name,digest in provenance['files'].items():
        relative=PurePosixPath(name)
        if relative.is_absolute() or '..' in relative.parts or any(p.startswith('.') for p in relative.parts):raise ValueError('Unsafe model file name')
        p=model_path/name
        if p.is_symlink() or not p.is_file() or p.suffix.lower() not in {'.json','.txt','.jinja','.model','.safetensors'}:raise ValueError('Unapproved model asset')
        if sha256_file(p)!=digest:raise ValueError(f'Pinned model asset changed: {name}')
        model_files.append(name)
    if not {'config.json','tokenizer.json','preprocessor_config.json','video_preprocessor_config.json'}.issubset(model_files):raise ValueError('Missing native model/processor files')
    if 'model.safetensors' not in model_files:
        index=model_path/'model.safetensors.index.json'
        if index.name not in model_files or not set(json.loads(index.read_text())['weight_map'].values()).issubset(model_files):raise ValueError('Missing native model shards')
    context.mkdir(parents=True);(context/'baseline').mkdir()
    for name in ['__init__.py','io.py','environment.py','dense.py','projection.py','temporal.py','video.py','native_video.py','native_ranking.py','native_run.py']:
        shutil.copyfile(PROJECT/'baseline'/name,context/'baseline'/name)
    for name in model_files+['model_provenance.json']:
        target=context/'model'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(model_path/name,target)
    (context/'checkpoint').mkdir()
    for name in ['adapter.safetensors','metadata.json']:shutil.copyfile(checkpoint/name,context/'checkpoint'/name)
    target=context/'vendor/qwen3-vl-embedding';target.mkdir(parents=True);shutil.copyfile(helper,target/helper.name)
    for name in ['LICENSE','NOTICE','THIRD_PARTY.md']:shutil.copyfile(PROJECT/name,context/name)
    shutil.copytree(PROJECT/'third_party_licenses',context/'third_party_licenses')
    shutil.copyfile(PROJECT/'requirements-native.txt',context/'requirements.txt')
    write_json(context/'native_recipe.json',recipe)
    (context/'run.py').write_text("from baseline.native_run import main\n\nif __name__ == '__main__':\n    main()\n")
    (context/'Dockerfile').write_text(f'''ARG BASE_IMAGE={BASE_IMAGE}
FROM ${{BASE_IMAGE}}
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libgomp1 && rm -rf /var/lib/apt/lists/*
RUN python -m venv /opt/app/.venv
COPY requirements.txt /opt/app/requirements.txt
RUN /opt/app/.venv/bin/pip install --no-cache-dir -r /opt/app/requirements.txt && /opt/app/.venv/bin/pip check
COPY baseline /opt/app/baseline
COPY vendor /opt/app/vendor
COPY model /opt/app/model
COPY checkpoint /opt/app/checkpoint
COPY native_recipe.json run.py LICENSE NOTICE THIRD_PARTY.md /opt/app/
COPY third_party_licenses /opt/app/third_party_licenses
''')
    (context/'submission.yaml').write_text('''entrypoint: ["/opt/app/.venv/bin/python", "/opt/app/run.py"]
env:
  DEVICE: "cuda"
  LD_LIBRARY_PATH: "/usr/lib/x86_64-linux-gnu:/usr/local/nvidia/lib64:/usr/local/nvidia/lib:/usr/lib64"
  HF_HUB_OFFLINE: "1"
  TRANSFORMERS_OFFLINE: "1"
  TOKENIZERS_PARALLELISM: "false"
  OMP_NUM_THREADS: "2"
  MKL_NUM_THREADS: "2"
cuda: "sm_80 binaries and actual A100/offline runtime verification pending"
notes: "Native video embedding plus shared residual adapter; local-only user handoff"
''')
    (context/'.dockerignore').write_text('.cascade-build\n__pycache__\nHANDOFF.md\npackage_manifest.json\n')
    (context/'HANDOFF.md').write_text('''# Native local build context

This is an unbuilt local context, not a verified challenge artifact. It includes pinned model/processor/helper assets and a hash-matched projection recipe. No annotations, qrels, released query text, raw challenge videos, known-ID vectors, or credentials are included. The head metadata contains provenance hashes only.

Build using the pinned devkit with --no-push. The entrypoint honors INPUT_DIR, OUTPUT_DIR and TOP_K; asset resolution does not depend on the current working directory or Docker ENV/WORKDIR. Run explicit output validation after offline delta reconstruction, then A100/sm_80 and full pixel-to-ranking runtime checks. Inspect the resolved CUDA wheel and dependency lock, not only these direct pins. Pretrained-component eligibility and final dependency-license review remain unresolved.

The user alone publishes the artifact, accepts agreements or queues official evaluation. Agents stop at this local context/artifact and handoff.
''')
    manifest={'kind':'local-native-build-context','official_submission_made':False,'built':False,'devkit_commit':DEVKIT_COMMIT,
        'base_image':BASE_IMAGE,'recipe':recipe,'files':{str(p.relative_to(context)):sha256_file(p)for p in sorted(context.rglob('*'))if p.is_file()}}
    write_json(context/'package_manifest.json',manifest);return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=Path('artifacts/handoff/native-historical-v0'))
    p.add_argument('--model-path',type=Path,default=Path('artifacts/models/qwen3-video'))
    p.add_argument('--checkpoint',type=Path,default=Path('runs/learning-prompt-v1/projection/final-checkpoint'))
    p.add_argument('--training-manifest',type=Path,default=Path('runs/learning-prompt-v1/projection/results/experiment_manifest.json'))
    p.add_argument('--features',type=Path,default=Path('runs/research-queue-v1/native/features'))
    p.add_argument('--representation',choices=['historical4','synthetic4','time4','time8','h1004'],default='historical4')
    p.add_argument('--build',action='store_true',help='Build local layers using --no-push; never publish')
    a=p.parse_args();result=stage(a.out,a.model_path,a.checkpoint,a.training_manifest,a.features,a.representation)
    print(f"Staged {len(result['files'])} verified files at {a.out}")
    if a.build:
        import subprocess
        subprocess.run(local_build_command(a.out),check=True)
