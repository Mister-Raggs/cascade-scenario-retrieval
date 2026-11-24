"""Bounded development diagnostics and native readiness checks; never publish or submit."""
from __future__ import annotations
import argparse, gc, json, os, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from baseline.io import sha256_file, unique_rows, write_json, write_jsonl
from baseline.dense import normalize, rank_scores
from baseline.experiment_analysis import prefix_diagnostic
from scripts.feature_retrieval import load_vectors

COMPARISON = Path('runs/temporal-features-v2/analysis-v2/comparison')
CONTEXT = Path('artifacts/handoff/native-historical-v0')
INPUT = Path('data/prepared-train-blip-v1/input')
QRELS = INPUT.parent / 'labels/qrels.jsonl'


def source():
    manifest = json.loads((COMPARISON / 'experiment_manifest.json').read_text())
    videos = unique_rows(INPUT / 'videos.jsonl', 'video_id')
    queries = unique_rows(INPUT / 'queries.jsonl', 'query_id')
    labels = unique_rows(QRELS, 'query_id')
    if sha256_file(QRELS) != manifest['training_qrels_sha256']:
        raise ValueError('Training labels differ from frozen comparison')
    return manifest, videos, queries, labels


def project(path, values):
    import torch
    from safetensors.torch import load_file
    from baseline.projection import SharedProjection
    weights = load_file(str(path), device='cpu')
    rank, dim = weights['down.weight'].shape
    model = SharedProjection(dim, rank).eval()
    model.load_state_dict(weights, strict=True)
    with torch.inference_mode():
        return np.concatenate([model(torch.from_numpy(values[i:i+512])).numpy() for i in range(0, len(values), 512)])


def audit(out):
    import torch
    torch.set_num_threads(2)
    manifest, videos, queries, labels = source()
    query_ids = manifest['query_groups']['dev']
    slices = manifest['plan']['slices']
    results, examples = [], []
    for rep in manifest['plan']['representations']:
        if rep['id'] == 'budget16':
            continue
        feature_dir = Path(rep['directory'])
        feature = json.loads((feature_dir / 'manifest.json').read_text())
        if sha256_file(feature_dir / 'manifest.json') != manifest['feature_parents'][rep['id']]:
            raise ValueError('Feature manifest changed')
        for name in (rep['document_file'], 'queries-train.npz'):
            if sha256_file(feature_dir / name) != feature['files'][name]:
                raise ValueError('Feature bytes changed')
        v_all = load_vectors(feature_dir / rep['document_file'])
        q_all = load_vectors(feature_dir / 'queries-train.npz')
        q_positions = {q:i for i,q in enumerate(feature['query_ids']['train'])}
        v_positions = {v:i for i,v in enumerate(feature['video_ids'])}
        for fold, ids in enumerate(manifest['development_video_groups']):
            checkpoint = COMPARISON / 'development' / rep['id'] / f'fold-{fold}-seed-43' / 'best' / 'adapter.safetensors'
            metadata = json.loads(checkpoint.with_name('metadata.json').read_text())
            if sha256_file(checkpoint) != metadata['weights_sha256']:
                raise ValueError('Frozen checkpoint changed')
            if metadata['manifest_sha256'] != sha256_file(COMPARISON / 'experiment_manifest.json'):
                raise ValueError('Checkpoint comparison parent differs')
            vs = project(checkpoint, v_all[[v_positions[v] for v in ids]])
            qs = project(checkpoint, q_all[[q_positions[q] for q in query_ids]])
            scores = qs @ vs.T
            rows = []
            for i, qid in enumerate(query_ids):
                relevant = set(labels[qid]['relevant_video_ids']) & set(ids)
                if not relevant:
                    continue
                ranking = rank_scores(ids, scores[i])
                diagnostics = {str(k):prefix_diagnostic(ranking, relevant, k) for k in (20,50,100,200)}
                text = queries[qid]['text']
                rows.append({'query_id':qid, 'text':text, 'relevant_count':len(relevant),
                             'r_precision':diagnostics['100']['current_r_precision'], 'depths':diagnostics,
                             'slices':[s for s, words in slices.items() if any(w in text.lower() for w in words)]})
                if rep['id'] == 'historical4' and fold == 0:
                    examples.append({'query_id':qid, 'text':text, 'r_precision':rows[-1]['r_precision'],
                        'relevant_count':len(relevant), 'top5':[{'video_id':v,'relevant':v in relevant} for v in ranking[:5]],
                        'first_missed_positives':[v for v in ranking[len(relevant):] if v in relevant][:3],
                        'interpretation':'Released training-label diagnostic. Actor/order/cause failure requires manual visual/annotation review.'})
            result = {'representation':rep['id'], 'fold':fold, 'seed':43, 'epoch':metadata['epoch'],
                'checkpoint_sha256':sha256_file(checkpoint), 'queries':len(rows), 'candidate_videos':len(ids),
                'r_precision':float(np.mean([r['r_precision'] for r in rows])),
                'depths':{str(k):{field:float(np.mean([r['depths'][str(k)][field] for r in rows]))
                        for field in ('recall','oracle_r_precision','prefix_cannot_change_r_precision')} for k in (20,50,100,200)},
                'heuristic_slices':{s:{'queries':sum(s in r['slices'] for r in rows),
                    'r_precision':float(np.mean([r['r_precision'] for r in rows if s in r['slices']])) if any(s in r['slices'] for r in rows) else None}
                    for s in slices}}
            results.append(result)
            write_jsonl(out / f"{rep['id']}-fold-{fold}-per-query.jsonl", rows)
    historical = [r for r in results if r['representation']=='historical4']
    report = {'kind':'Frozen training-side excluded-video development error diagnostics', 'is_validation_score':False,
        'reserved_audit_used':False, 'validation_used':False, 'results':results,
        'historical_mean_r_precision':float(np.mean([r['r_precision'] for r in historical])),
        'historical_depths':{str(k):{f:float(np.mean([r['depths'][str(k)][f] for r in historical]))
                        for f in ('recall','oracle_r_precision','prefix_cannot_change_r_precision')} for k in (20,50,100,200)},
        'comparison_manifest_sha256':sha256_file(COMPARISON/'experiment_manifest.json'), 'qrels_sha256':sha256_file(QRELS),
        'code_sha256':sha256_file(Path(__file__)),
        'limits':'Reused development queries/corpus, one frozen seed per fold. Overlapping keyword slices are not verified event or causal categories. Oracle uses labels only to estimate a reranking ceiling; it is not a deployable method.'}
    examples.sort(key=lambda r:(r['r_precision'],r['query_id']))
    write_jsonl(out/'review-examples.jsonl', examples[:24])
    write_json(out/'report.json', report)
    print(json.dumps({k:v for k,v in report.items() if k!='results'}),flush=True)


def encoder(context=CONTEXT):
    from baseline.native_video import NativeEncoder
    recipe = json.loads((context/'native_recipe.json').read_text())
    return NativeEncoder(context/'model',context/'vendor/qwen3-vl-embedding/qwen3_vl_embedding.py',recipe,'cuda')


def parity(out):
    import torch
    from scripts.research_experiments import decode, encode_video, QUERY_INSTRUCTION
    from baseline.native_run import load_projection
    torch.set_num_threads(2);torch.manual_seed(0)
    manifest, videos, queries, labels = source()
    features = Path('runs/research-queue-v1/native/features')
    feature = json.loads((features/'manifest.json').read_text())
    if sha256_file(features/'manifest.json') != manifest['feature_parents']['historical4']:
        raise ValueError('Research feature parent differs')
    for name in ('aggregate.npz','queries-train.npz'):
        if sha256_file(features/name) != feature['files'][name]:
            raise ValueError('Research feature bytes differ')
    ids = feature['video_ids'][::max(1,len(videos)//8)][:8]
    qids = manifest['query_groups']['dev'][:16]
    vcache = load_vectors(features/'aggregate.npz'); qcache=load_vectors(features/'queries-train.npz')
    vp = {v:i for i,v in enumerate(feature['video_ids'])};qp={q:i for i,q in enumerate(feature['query_ids']['train'])}
    e=encoder();rows=[];native=[];research=[];started=time.monotonic()
    for vid in ids:
        path=INPUT/videos[vid]['path']
        vector, metadata=e.video(e.decode(path))
        reference, _, _=encode_video(e.embedder,decode(path,4)[1])
        cached=vcache[vp[vid]]
        rows.append({'video_id':vid,'video_sha256':sha256_file(path),'native_vs_live_max_abs':float(np.max(np.abs(vector-reference))),
            'native_vs_cached_max_abs':float(np.max(np.abs(vector-cached))), 'native_vs_cached_cosine':float(vector@cached),**metadata})
        native.append(vector);research.append(reference)
    nq=e.queries([queries[q]['text'] for q in qids])
    with torch.inference_mode():
        rq=normalize(e.embedder.process([{'text':queries[q]['text'],'instruction':QUERY_INSTRUCTION} for q in qids]).float().cpu().numpy())
    cq=qcache[[qp[q] for q in qids]]
    # BF16 attention kernels depend on padding/batch shape. Compare saved query
    # vectors using their original neighbours, rather than a newly regrouped batch.
    matched={}
    parent_order=feature['query_ids']['train']
    for start in sorted({(qp[q]//16)*16 for q in qids}):
        batch_ids=parent_order[start:start+16]
        batch=e.queries([queries[q]['text'] for q in batch_ids])
        matched.update(zip(batch_ids,batch,strict=True))
    mq=np.stack([matched[q] for q in qids])
    proj=load_projection(CONTEXT/'checkpoint',e.recipe,'cuda')
    scores=proj(nq)@proj(np.stack(native)).T;rscores=proj(rq)@proj(np.stack(research)).T
    rankings_equal=all(rank_scores(ids,a)==rank_scores(ids,b) for a,b in zip(scores,rscores,strict=True))
    passed=all(r['native_vs_live_max_abs']<1e-6 and r['native_vs_cached_cosine']>.9999 and r['native_vs_cached_max_abs']<.003 for r in rows)
    passed=passed and np.max(np.abs(nq-rq))<1e-6 and float(np.min(np.sum(mq*cq,axis=1)))>.9999 and np.max(np.abs(mq-cq))<.003 and rankings_equal
    report={'kind':'RTX CUDA BF16 native/research numerical parity', 'passed':bool(passed),'is_validation_score':False,
        'videos':rows,'queries':len(qids),'queries_native_vs_live_max_abs':float(np.max(np.abs(nq-rq))),
        'queries_native_vs_cache_min_cosine':float(np.min(np.sum(nq*cq,axis=1))), 'ranking_equal_to_live_parent':rankings_equal,
        'queries_original_batch_vs_cache_min_cosine':float(np.min(np.sum(mq*cq,axis=1))),
        'queries_original_batch_vs_cache_max_abs':float(np.max(np.abs(mq-cq))),
        'query_regrouping_drift_recorded':True,
        'score_max_abs':float(np.max(np.abs(scores-rscores))), 'cuda':torch.version.cuda,'torch':torch.__version__,
        'compiled_arches':torch.cuda.get_arch_list(),'sm80_present':'sm_80' in torch.cuda.get_arch_list(),
        'seconds':time.monotonic()-started, 'package_manifest_sha256':sha256_file(CONTEXT/'package_manifest.json'),
        'limits':'Eight real clips/sixteen released training queries. Host environment, not reconstructed artifact or actual A100; cache tolerances predeclared in script.'}
    write_json(out/'report.json',report);print(json.dumps(report),flush=True)
    if not passed:raise RuntimeError('CUDA parity failed')


def pilot(out):
    import torch
    from baseline.native_video import decode_source_times
    from scripts.refine_fusion import score_rankings
    torch.set_num_threads(2);torch.manual_seed(0)
    manifest,videos,queries,labels=source()
    # Fixed source cohort before reading its scores, development-only; no reserved audit.
    pool=sorted(manifest['development_video_groups'][0])
    rng=np.random.default_rng(20261024)
    ids=sorted(rng.choice(pool,size=32,replace=False).tolist())
    qids=[q for q in manifest['query_groups']['dev'] if set(labels[q]['relevant_video_ids'])&set(ids)][:64]
    if not qids:raise ValueError('Fixed development cohort has no positives')
    e=encoder();e.recipe={**e.recipe,'representation':'time4','frames':4}
    e.embedder.total_pixels=16*256*448
    started=time.monotonic();windows=[];whole=[];profiles=[]
    for index,vid in enumerate(ids):
        path=INPUT/videos[vid]['path'];began=time.monotonic()
        stamps,frames,duration,decoder=decode_source_times(path,8)
        vectors=[];metadata=[]
        for start in (0,4):
            vector,m=e.video((stamps[start:start+4],frames[start:start+4],duration,decoder))
            vectors.append(vector);metadata.append(m)
        e.recipe={**e.recipe,'representation':'time8','frames':8}
        v,m=e.video((stamps,frames,duration,decoder));whole.append(v)
        e.recipe={**e.recipe,'representation':'time4','frames':4}
        windows.append(np.stack(vectors))
        profiles.append({'video_id':vid,'video_sha256':sha256_file(path),'source_timestamps':stamps,
            'window_profiles':metadata,'whole8_profile':m,'decode_and_three_encodes_seconds':time.monotonic()-began})
        write_json(out/'progress.json',{'stage':'two-window-pilot','completed':index+1,'total':32})
    w=np.stack(windows);v=np.stack(whole);q=e.queries([queries[qid]['text'] for qid in qids])
    np.savez_compressed(out/'features.npz',windows=w,whole8=v,queries=q)
    match=np.array([[vid in labels[qid]['relevant_video_ids'] for vid in ids] for qid in qids],dtype=bool)
    window_scores=np.einsum('qd,vwd->qvw',q,w)
    methods={'whole8':q@v.T,'window_mean':q@normalize(w.mean(axis=1)).T,
             'window_max':window_scores.max(axis=2),'window_min':window_scores.min(axis=2)}
    measurements={}
    for name,scores in methods.items():
        ranking=np.argsort(-scores,axis=1,kind='stable');rp,ap=score_rankings(ranking,match)
        measurements[name]={'r_precision':float(np.nanmean(rp)),'map':float(np.nanmean(ap))}
        write_jsonl(out/f'{name}-predictions.jsonl',({'query_id':qid,'video_ids':[ids[i] for i in ranking[j]]}for j,qid in enumerate(qids)))
    report={'kind':'Two chronological four-frame subclip feasibility pilot', 'is_validation_score':False,
        'videos':len(ids),'queries':len(qids),'video_ids':ids,'query_ids':qids,'selection_seed':20261024,
        'training_subset_measurements':measurements,'seconds':time.monotonic()-started,'profiles':profiles,
        'feature_sha256':sha256_file(out/'features.npz'),'peak_gpu_reserved_bytes':torch.cuda.max_memory_reserved(),
        'order_preserved_in_saved_index':True, 'pooled_scoring_is_order_sensitive':False,
        'same_sampled_pixels_for_whole8_and_windows':True,'adapter_used':False,
        'comparison_manifest_sha256':sha256_file(COMPARISON/'experiment_manifest.json'),'qrels_sha256':sha256_file(QRELS),
        'limits':'32-video restricted candidate corpus, at most 64 previously studied development queries; raw embeddings only. Mean/max/min pooling cannot enforce event order. No promotion, full validation, training or overnight queue is authorized by this diagnostic.'}
    write_json(out/'report.json',report);write_json(out/'progress.json',{'stage':'complete'})
    print(json.dumps({k:v for k,v in report.items() if k not in {'profiles','video_ids','query_ids'}}),flush=True)


if __name__=='__main__':
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['audit','parity','pilot']);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    globals()[a.stage](a.out)
