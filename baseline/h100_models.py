"""Bounded development video scorers with explicit clocks and no visual fallback."""
from __future__ import annotations
from collections import OrderedDict
import importlib.util
import json
from pathlib import Path
import sys
import time
import numpy as np
from .io import sha256_file
from .native_video import decode_source_times, DOC_INSTRUCTION, QUERY_INSTRUCTION
from .temporal import timestamp_metadata
from .dense import normalize

def helper(name):
    path=Path('vendor/qwen3-vl-embedding')/(name+'.py')
    spec=importlib.util.spec_from_file_location('_h100_'+name,path)
    m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
    return m

class FrameStore:
    def __init__(self, directory, count=4, width=640):
        if count<2 or count%2:raise ValueError('An even frame count of at least two is required')
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True)
        self.memory=OrderedDict();self.count=count;self.width=width

    def get(self, video_id):
        from PIL import Image
        if video_id in self.memory:
            self.memory.move_to_end(video_id);return self.memory[video_id]
        source=Path('data/videos')/(video_id+'.mp4')
        target=self.directory/video_id;metadata=target/'metadata.json'
        source_hash=sha256_file(source)
        if metadata.exists():
            value=json.loads(metadata.read_text())
            if value['source_sha256']!=source_hash:raise ValueError('Frame cache parent changed')
            if len(value['frames_sha256'])!=self.count:raise ValueError('Frame cache has a different frame count')
            if value.get('width',640)!=self.width:raise ValueError('Frame cache has a different decode width')
            for name,digest in value['frames_sha256'].items():
                if sha256_file(target/name)!=digest:raise ValueError('Frame cache bytes changed')
            frames=[Image.open(target/f'{i}.png').convert('RGB') for i in range(self.count)]
        else:
            stamps,frames,duration,decoder=decode_source_times(source,self.count,self.width)
            target.mkdir(parents=True,exist_ok=True)
            for i,frame in enumerate(frames):frame.save(target/f'{i}.png')
            value={'video_id':video_id,'source_sha256':source_hash,'timestamps':stamps,'duration':duration,
                'decoder':decoder,'frames_sha256':{f'{i}.png':sha256_file(target/f'{i}.png') for i in range(self.count)}}
            if self.width!=640:value['width']=self.width  # 640-wide caches keep their original metadata
            temporary=metadata.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(metadata)
        result=(frames,value)
        self.memory[video_id]=result
        if len(self.memory)>32:self.memory.popitem(last=False)
        return result

def visual_batch(processor, conversations, metadata, max_length=2048, prefix_marker=None, template_kwargs=None):
    """`prefix_marker`: tokenize only the text before this marker (shared video prefix; questions are appended later).
    `template_kwargs`: extra chat-template switches (for example enable_thinking=False for thinking models)."""
    from qwen_vl_utils import process_vision_info
    text=processor.apply_chat_template(conversations,tokenize=False,add_generation_prompt=True,**(template_kwargs or {}))
    if prefix_marker is not None:
        if any(t.count(prefix_marker)!=1 for t in text):raise ValueError('Prefix marker must occur once per conversation')
        text=[t.split(prefix_marker)[0] for t in text]
    images,videos,kwargs=process_vision_info(conversations,image_patch_size=16,
        return_video_metadata=True,return_video_kwargs=True)
    if images is not None or videos is None or len(videos)!=len(conversations):
        raise ValueError('Missing/unexpected video input; text fallback forbidden')
    pixels=[v[0] for v in videos]
    actual=[timestamp_metadata(v['timestamps'],v['duration']) for v in metadata]
    for v,m in zip(metadata,actual,strict=True):
        times=processor._calculate_timestamps(m['frames_indices'].copy(),m['fps'])
        expected=[(v['timestamps'][i]+v['timestamps'][i+1])/2 for i in range(0,len(v['timestamps']),2)]
        if not np.allclose(times,expected,atol=1e-9):raise ValueError('Video clock mismatch')
    batch=processor(text=text,images=None,videos=pixels,video_metadata=actual,padding=True,
        truncation=False,do_resize=False,return_tensors='pt',**kwargs)
    if 'pixel_values_videos' not in batch or not batch['pixel_values_videos'].numel():raise ValueError('Empty visual tensors')
    lengths=batch['attention_mask'].sum(1).cpu().tolist()
    if max(lengths)>max_length:raise ValueError('Context cap exceeded; never truncate visual tokens')
    grids=batch['video_grid_thw'].cpu().tolist()
    return batch,{'sequence_tokens':lengths,'video_grids':grids,'actual_clock':True}

class VideoReranker:
    def __init__(self, path, instruction=None):
        import torch
        from transformers import Qwen3VLForConditionalGeneration,AutoProcessor
        m=helper('qwen3_vl_reranker')
        formatter=m.Qwen3VLReranker.__new__(m.Qwen3VLReranker)
        formatter.default_instruction='Given a search query, retrieve relevant candidates that answer the query.'
        formatter.total_pixels=4*256*448;formatter.min_pixels=4*32*32
        formatter.max_pixels=1800*32*32;formatter.max_frames=4;formatter.fps=1
        self.formatter=formatter
        self.processor=AutoProcessor.from_pretrained(path,padding_side='left',local_files_only=True,trust_remote_code=False)
        lm=Qwen3VLForConditionalGeneration.from_pretrained(path,dtype=torch.bfloat16,
            attn_implementation='sdpa',local_files_only=True,trust_remote_code=False).cuda().eval()
        vocab=self.processor.tokenizer.get_vocab()
        self.weight=(lm.lm_head.weight[vocab['yes']]-lm.lm_head.weight[vocab['no']]).detach().clone()
        self.model=lm.model.eval();self.instruction=instruction or formatter.default_instruction
        del lm

    def score(self,pairs,frames):
        groups=OrderedDict()
        for i,(query,video_id) in enumerate(pairs):groups.setdefault(query,[]).append((i,video_id))
        if len(groups)>1:
            result=[None]*len(pairs);tokens=[None]*len(pairs);grids=[None]*len(pairs)
            for query,items in groups.items():
                scores,info=self._score_same_query([(query,vid) for _,vid in items],frames)
                for j,(index,vid) in enumerate(items):
                    result[index]=scores[j];tokens[index]=info['sequence_tokens'][j];grids[index]=info['video_grids'][j]
            return result,{'sequence_tokens':tokens,'video_grids':grids,'actual_clock':True,'homogeneous_query_batches':True}
        return self._score_same_query(pairs,frames)

    def _score_same_query(self,pairs,frames):
        import torch
        conversations=[];metadata=[]
        for query,video_id in pairs:
            images,meta=frames.get(video_id)
            conversation=self.formatter.format_mm_instruction(query_text=query,doc_video=images,instruction=self.instruction)
            for item in conversation[1]['content']:
                if item['type']=='video':item.update(resized_height=256,resized_width=448,min_pixels=4*32*32)
            conversations.append(conversation);metadata.append(meta)
        batch,info=visual_batch(self.processor,conversations,metadata)
        batch={k:v.cuda() for k,v in batch.items()}
        with torch.inference_mode():
            hidden=self.model(**batch,use_cache=False).last_hidden_state[:,-1]
            scores=torch.sigmoid(hidden@self.weight).float().cpu().numpy()
        torch.cuda.synchronize()
        if scores.shape!=(len(pairs),) or not np.isfinite(scores).all():raise ValueError('Nonfinite/missing pair scores')
        return scores.tolist(),info

class VideoEmbedder:
    def __init__(self,path):
        import torch
        m=helper('qwen3_vl_embedding')
        self.model=m.Qwen3VLForEmbedding.from_pretrained(path,dtype=torch.bfloat16,
            attn_implementation='sdpa',local_files_only=True,trust_remote_code=False).cuda().eval()
        self.processor=m.Qwen3VLProcessor.from_pretrained(path,padding_side='right',local_files_only=True)
        self.formatter=m.Qwen3VLEmbedder.__new__(m.Qwen3VLEmbedder)
        for k,v in {'default_instruction':"Represent the user's input.",'total_pixels':4*256*448,
                    'min_pixels':4*32*32,'max_pixels':1800*32*32,'num_frames':4,'max_frames':4,'fps':1}.items():
            setattr(self.formatter,k,v)
        self.formatter.model=self.model;self.formatter.processor=self.processor;self.formatter.max_length=2048

    def videos(self,ids,frames,height=256,width=448):
        import torch
        conversations=[];metadata=[]
        for vid in ids:
            images,meta=frames.get(vid)
            c=self.formatter.format_model_input(video=images,instruction=DOC_INSTRUCTION)
            c[1]['content'][0].update(resized_height=height,resized_width=width,min_pixels=4*32*32)
            conversations.append(c);metadata.append(meta)
        batch,info=visual_batch(self.processor,conversations,metadata)
        batch={k:v.cuda() for k,v in batch.items()}
        with torch.inference_mode():
            output=self.model(**batch,use_cache=False)
            vectors=self.formatter._pooling_last(output.last_hidden_state,batch['attention_mask'])
        return normalize(vectors.float().cpu().numpy()),info

    def queries(self,texts):
        import torch
        with torch.inference_mode():
            vectors=self.formatter.process([{'text':t,'instruction':QUERY_INSTRUCTION} for t in texts])
        return normalize(vectors.float().cpu().numpy())
