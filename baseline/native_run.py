"""Offline native-video retrieval; no label access or known-corpus feature caches."""
from __future__ import annotations
import argparse,json,os,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from .dense import normalize,rank_scores
from .environment import environment
from .io import sha256_file,unique_rows,write_json,write_jsonl
from .native_video import NativeEncoder,validate_recipe
from .native_ranking import blended_scores


def load_projection(checkpoint,recipe,device):
    import torch
    from safetensors.torch import load_file
    from .projection import SharedProjection
    metadata=json.loads((checkpoint/'metadata.json').read_text())
    actual=sha256_file(checkpoint/'adapter.safetensors')
    if actual!=metadata['weights_sha256'] or actual!=recipe['adapter_sha256']:raise ValueError('Projection weights differ from frozen recipe')
    model=SharedProjection(recipe['dimension'],recipe['rank']).to(device).eval()
    model.load_state_dict(load_file(str(checkpoint/'adapter.safetensors'),device=device),strict=True)
    if not all(torch.isfinite(p).all() for p in model.parameters()):raise ValueError('Nonfinite projection weights')
    def project(vectors):
        values=[]
        with torch.inference_mode():
            for i in range(0,len(vectors),512):
                values.append(model(torch.as_tensor(vectors[i:i+512],dtype=torch.float32,device=device)).cpu().numpy())
        return normalize(np.concatenate(values))
    return project


def run(input_dir,output_dir,*,model_path,helper_path,checkpoint,recipe_path,top_k=100,device='auto',encoder_factory=NativeEncoder,projection_loader=load_projection):
    if isinstance(top_k,bool) or not isinstance(top_k,int) or top_k<0:raise ValueError('TOP_K must be a nonnegative integer')
    if output_dir.resolve().is_relative_to(input_dir.resolve()) or input_dir.resolve().is_relative_to(output_dir.resolve()):raise ValueError('Input and output directories must be separate')
    recipe=json.loads(recipe_path.read_text());validate_recipe(recipe)
    if sha256_file(helper_path)!=recipe['helper_sha256']:raise ValueError('Native helper changed')
    videos=unique_rows(input_dir/'videos.jsonl','video_id');queries=unique_rows(input_dir/'queries.jsonl','query_id')
    if not queries:raise ValueError('Query set is empty')
    if any(not isinstance(r.get('text'),str) for r in queries.values()):raise ValueError('Query text must be a string')
    ids=sorted(videos);paths=[]
    for vid in ids:
        relative=videos[vid].get('path')
        if not isinstance(relative,str) or Path(relative).is_absolute() or '..' in Path(relative).parts:raise ValueError('Expected a relative video path beneath INPUT_DIR')
        # Prepared local inputs may use symlinks to approved videos; never write through them.
        path=input_dir/relative
        if not path.is_file():raise FileNotFoundError(path)
        paths.append(path)
    output_dir.mkdir(parents=True,exist_ok=True);began=time.perf_counter()
    rows=[];documents=[];query_ids=list(queries);runtime=environment(device,load_torch=False)
    if ids and top_k:
        encoder=encoder_factory(model_path,helper_path,recipe,device)
        project=projection_loader(checkpoint,recipe,encoder.device)
        runtime=environment(encoder.device,load_torch=True)
        dimension=recipe['dimension']
        workers=recipe.get('decode_workers',1)
        if isinstance(workers,bool) or not isinstance(workers,int) or not 1<=workers<=4:raise ValueError('Invalid decode worker count')
        with ThreadPoolExecutor(max_workers=workers) as pool:
            size=recipe.get('video_batch_size',1)
            if isinstance(size,bool) or not isinstance(size,int) or not 1<=size<=4:raise ValueError('Invalid video batch size')
            pending=[pool.submit(encoder.decode,path) for path in paths[:size]]
            for start in range(0,len(ids),size):
                decoded=[future.result() for future in pending]
                pending=[pool.submit(encoder.decode,path) for path in paths[start+size:start+2*size]]
                if size==1:
                    vector,metadata=encoder.video(decoded[0]);vectors=[vector];metadata_rows=[metadata]
                else:vectors,metadata_rows=encoder.videos(decoded)
                for vid,path,vector,metadata in zip(ids[start:start+size],paths[start:start+size],vectors,metadata_rows,strict=True):
                    vector=normalize(vector)
                    if vector.shape!=(dimension,):raise ValueError('Video embedding dimension differs')
                    documents.append(vector);rows.append({'video_id':vid,'video_sha256':sha256_file(path),**metadata})
                write_json(output_dir/'progress.json',{'stage':'encoding-videos','completed':min(start+size,len(ids)),'total':len(ids)})
        raw_docs=np.stack(documents);docs=project(raw_docs);encoded_seconds=time.perf_counter()-began
        def predictions():
            for start in range(0,len(query_ids),16):
                selected=query_ids[start:start+16]
                q=normalize(encoder.queries([queries[qid]['text'] for qid in selected]))
                if q.shape!=(len(selected),dimension):raise ValueError('Query embedding dimensions differ')
                scores=blended_scores(q@raw_docs.T,project(q)@docs.T,recipe.get('raw_rank_weight',0))
                for qid,score in zip(selected,scores,strict=True):yield {'query_id':qid,'video_ids':rank_scores(ids,score)[:top_k]}
        write_jsonl(output_dir/'predictions.jsonl',predictions())
    else:
        encoded_seconds=0
        write_jsonl(output_dir/'predictions.jsonl',({'query_id':qid,'video_ids':[]}for qid in query_ids))
    write_jsonl(output_dir/'video_processing.jsonl',rows)
    report={'method':'native-video-shared-projection','recipe':recipe,'recipe_sha256':sha256_file(recipe_path),
        'is_validation_score':False,'num_videos':len(ids),'num_queries':len(queries),'top_k':top_k,
        'video_encoding_seconds':encoded_seconds,'total_seconds':time.perf_counter()-began,
        'queries_sha256':sha256_file(input_dir/'queries.jsonl'),'videos_manifest_sha256':sha256_file(input_dir/'videos.jsonl'),
        'environment':runtime,'predictions_sha256':sha256_file(output_dir/'predictions.jsonl'),'feature_reuse':'none; encode current input pixels and query text'}
    write_json(output_dir/'run_manifest.json',report);write_json(output_dir/'progress.json',{'stage':'complete'})
    return report


def main():
    # Offline settings are mandatory for both helper and tokenizer, before their imports/load.
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
    import torch
    torch.set_num_threads(2);torch.manual_seed(0)
    root=Path(__file__).resolve().parents[1]
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=Path(os.environ.get('INPUT_DIR','/input')))
    p.add_argument('--output',type=Path,default=Path(os.environ.get('OUTPUT_DIR','/output')))
    p.add_argument('--model-path',type=Path,default=root/'model')
    p.add_argument('--helper-path',type=Path,default=root/'vendor/qwen3-vl-embedding/qwen3_vl_embedding.py')
    p.add_argument('--checkpoint',type=Path,default=root/'checkpoint')
    p.add_argument('--recipe',type=Path,default=root/'native_recipe.json')
    p.add_argument('--top-k',type=int,default=int(os.environ.get('TOP_K','100')))
    p.add_argument('--device',choices=['auto','cuda','cpu'],default=os.environ.get('DEVICE','auto'))
    a=p.parse_args();report=run(a.input,a.output,model_path=a.model_path,helper_path=a.helper_path,checkpoint=a.checkpoint,
        recipe_path=a.recipe,top_k=a.top_k,device=a.device)
    print(f"Wrote {report['num_queries']} native rankings in {report['total_seconds']:.2f}s",flush=True)


if __name__=='__main__':main()
