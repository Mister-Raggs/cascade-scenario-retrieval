"""Local-only offline delta reconstruction check; bounded inputs, explicit validation."""
from __future__ import annotations
import argparse, importlib.util, json, os, signal, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from baseline.io import sha256_file,unique_rows,write_json,write_jsonl
BASE='python:3.12@sha256:2575347025c314e37d89d4b353904edbe1824a6117b8eeffe52254879e4f6146'
OUT=ROOT/'runs/native-container-readiness-v1'
ARTIFACT=ROOT/'artifacts/handoff/native-readiness-v1'


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module


def execute(command,log,seconds):
    with log.open('w')as stream:
        process=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        try:return process.wait(timeout=seconds)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
            raise RuntimeError(f'Bounded container check exceeded {seconds}s')


def main():
    os.chdir(ROOT);OUT.mkdir(parents=True,exist_ok=False)
    input_dir=OUT/'input';output_dir=OUT/'output';input_dir.mkdir();output_dir.mkdir()
    parent=ROOT/'data/prepared-train-blip-v1/input'
    videos=unique_rows(parent/'videos.jsonl','video_id');queries=unique_rows(parent/'queries.jsonl','query_id')
    # Unfamiliar IDs and a fixed strided corpus sample. Local hardlinks avoid duplicate bytes;
    # Docker read-only mounting protects originals, without chmod on shared inodes.
    ids=sorted(videos)[::max(1,len(videos)//32)][:32];selected=list(queries)[:64]
    aliases={vid:f'unfamiliar-container-video-{i:03d}'for i,vid in enumerate(ids)}
    paths=[]
    for i,vid in enumerate(ids):
        source=(parent/videos[vid]['path']).resolve();target=input_dir/f'clip-{i:03d}.mp4'
        os.link(source,target);paths.append(target)
    write_jsonl(input_dir/'videos.jsonl',({'video_id':aliases[vid],'path':paths[i].name}for i,vid in enumerate(ids)))
    write_jsonl(input_dir/'queries.jsonl',({'query_id':f'unfamiliar-container-query-{i:03d}','text':queries[q]['text']}for i,q in enumerate(selected)))
    (input_dir/'qrels.jsonl').write_text('Poison labels: inference must never read this file.')
    before={p.name:sha256_file(p)for p in input_dir.iterdir()}
    reconstruction=load('_native_readiness_reconstruct',ROOT/'vendor/cascade-devkit/docker-prep-kit/reconstruct_submission.py')
    manifest=reconstruction.load_manifest(ARTIFACT);reconstruction.verify_base(BASE,manifest)
    spec=reconstruction.RunSpec(artifact_dir=ARTIFACT,base_image=BASE,input_dir=input_dir,output_dir=output_dir,network='none',gpus='all')
    command=reconstruction.docker_run_command(spec)
    command[2:2]=['--name','av-native-readiness-v1','--env','TOP_K=7']
    write_json(OUT/'status.json',{'stage':'offline-delta-reconstruction','videos':32,'queries':64,'top_k':7,'maximum_seconds':600})
    started=time.monotonic()
    code=execute(command,OUT/'reconstruction.log',600)
    after={p.name:sha256_file(p)for p in input_dir.iterdir()}
    validator=load('_native_readiness_validator',ROOT/'vendor/cascade-devkit/docker-prep-kit/validate_submission.py')
    errors,warnings=validator.validate_predictions(output_dir/'predictions.jsonl',input_dir/'queries.jsonl',input_dir/'videos.jsonl',7)if code==0 else(['Reconstruction failed'],[])
    if before!=after:errors.append('Read-only inputs changed')
    if code==0:
        predictions=unique_rows(output_dir/'predictions.jsonl','query_id')
        if any(len(r['video_ids'])!=len(set(r['video_ids']))for r in predictions.values()):errors.append('Duplicate video IDs')
        import numpy as np
        from scripts.feature_retrieval import load_vectors
        from scripts.pre_shutdown_checks import project
        from baseline.dense import rank_scores
        features=ROOT/'runs/research-queue-v1/native/features'
        feature=json.loads((features/'manifest.json').read_text())
        for name in ['aggregate.npz','queries-train.npz']:
            if sha256_file(features/name)!=feature['files'][name]:raise ValueError('Research feature bytes changed')
        vp={v:i for i,v in enumerate(feature['video_ids'])};qp={q:i for i,q in enumerate(feature['query_ids']['train'])}
        checkpoint=ROOT/'artifacts/handoff/native-historical-v0/checkpoint/adapter.safetensors'
        reference=project(checkpoint,load_vectors(features/'queries-train.npz')[[qp[q]for q in selected]])@project(checkpoint,load_vectors(features/'aggregate.npz')[[vp[v]for v in ids]]).T
        expected_ids=[aliases[v]for v in ids]
        matching=sum(predictions[f'unfamiliar-container-query-{i:03d}']['video_ids']==rank_scores(expected_ids,reference[i])[:7]for i in range(len(selected)))
        if matching!=len(selected):errors.append(f'Cached parent top7 ranking mismatch: {matching}/{len(selected)} exact')
    run=json.loads((output_dir/'run_manifest.json').read_text())if code==0 else None
    report={'kind':'Offline native delta reconstruction on RTX6000 Ada','passed':code==0 and not errors,
        'is_validation_score':False,'exit_code':code,'network':'none','input_mounted_read_only':True,
        'input_hashes_unchanged':before==after,'videos':32,'queries':64,'top_k':7,
        'elapsed_seconds_including_reconstruction':time.monotonic()-started,
        'manifest_sha256':sha256_file(ARTIFACT/'manifest.json'),'artifact_bytes':sum(p.stat().st_size for p in ARTIFACT.rglob('*')if p.is_file()),
        'cached_parent_top7_exact_query_count':matching if code==0 else None,
        'validator_errors':errors,'validator_warnings':warnings,'run_manifest':run,
        'limits':'Thirty-two real clips and sixty-four released training texts under unfamiliar IDs. No quality score, actual A100 execution or full-corpus six-hour certification.'}
    write_json(OUT/'report.json',report)
    # Inspect the exact built environment offline; retain dependency metadata, not credentials.
    inventory='''import importlib.metadata as m,json,torch,subprocess
assert "sm_80" in torch.cuda.get_arch_list(),"Missing sm_80 binaries"
assert torch.cuda.is_available(),"No container CUDA"
x=torch.tensor([1.,2.,3.],device="cuda")
torch.testing.assert_close((x.square()+1).cpu(),torch.tensor([2.,5.,10.]))
torch.cuda.synchronize()
print(json.dumps({"torch":torch.__version__,"cuda":torch.version.cuda,"compiled_arches":torch.cuda.get_arch_list(),"cuda_available":torch.cuda.is_available(),"kernel":"ok","device":torch.cuda.get_device_name(0),"capability":torch.cuda.get_device_capability(0),"packages":[{"name":d.metadata.get("Name"),"version":d.version,"license":d.metadata.get("License"),"license_expression":d.metadata.get("License-Expression"),"classifiers":d.metadata.get_all("Classifier",[])}for d in m.distributions()],"ffmpeg":subprocess.check_output(["ffmpeg","-version"],text=True)}))
'''
    cmd=['docker','run','--rm','--network','none','--gpus','all','av-native-readiness:local','/opt/app/.venv/bin/python','-c',inventory]
    inventory_result=subprocess.run(cmd,capture_output=True,text=True,timeout=90)
    if inventory_result.returncode:raise RuntimeError(inventory_result.stderr[-1000:])
    dependencies=json.loads(inventory_result.stdout);write_json(OUT/'dependency-inventory.json',dependencies)
    (OUT/'requirements-resolved.txt').write_text('\n'.join(sorted(f"{d['name']}=={d['version']}"for d in dependencies['packages']))+'\n')
    write_json(OUT/'status.json',{'stage':'complete'if report['passed']else'failed'})
    print(json.dumps({k:v for k,v in report.items()if k!='run_manifest'}),flush=True)
    if not report['passed']:raise RuntimeError('Offline container reconstruction failed')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifact',type=Path,default=ARTIFACT);p.add_argument('--out',type=Path,default=OUT)
    a=p.parse_args();ARTIFACT=a.artifact.resolve();OUT=a.out.resolve();main()
