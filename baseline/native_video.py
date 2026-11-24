"""Pixel-to-vector native encoding for frozen, explicitly named research recipes."""
from __future__ import annotations
import importlib.util
import json
import re
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path
import numpy as np
from .dense import normalize
from .temporal import sample_indices,timestamp_metadata
from .video import frame_times,read_frame

DOC_INSTRUCTION='Represent this driving video for retrieving road users, ego actions, traffic controls, and temporally ordered interactions.'
QUERY_INSTRUCTION='Retrieve driving videos matching every described event, road user, traffic control, and spatial or temporal relation.'


def validate_recipe(recipe):
    if recipe['representation'] not in {'historical4','synthetic4','time4','time8','h1004'}:raise ValueError('Unsupported native representation')
    frames=recipe['frames']
    if frames!=(8 if recipe['representation']=='time8' else 4):raise ValueError('Frame count differs from representation')
    expected=4096 if recipe['representation']=='h1004' and recipe.get('model_id')=='Qwen/Qwen3-VL-Embedding-8B' else 2048
    if recipe['max_length']!=2048 or recipe['dimension']!=expected:raise ValueError('Native model/context dimensions changed')
    if recipe['representation']=='h1004' and recipe.get('model_id') not in {'Qwen/Qwen3-VL-Embedding-8B','Qwen/Qwen3-VL-Embedding-2B'}:raise ValueError('Unknown H100 extraction model')
    weight=recipe.get('raw_rank_weight',0)
    if isinstance(weight,bool) or not isinstance(weight,(int,float)) or not np.isfinite(weight) or not 0<=weight<=1:raise ValueError('Invalid raw rank weight')
    if recipe.get('rank_fusion_constant',60)!=60:raise ValueError('Rank fusion constant changed')
    if recipe.get('attention_backend','automatic') not in {'automatic','math'}:raise ValueError('Unsupported attention backend')
    if recipe['document_instruction']!=DOC_INSTRUCTION or recipe['query_instruction']!=QUERY_INSTRUCTION:raise ValueError('Native instructions changed')
    if recipe['representation']!='historical4' and (recipe['height'],recipe['width'])!=(256,448):raise ValueError('Temporal spatial recipe changed')


def decode_source_times(path,count,width=640):
    # Matches the pinned temporal extraction; retain FFmpeg's actual source PTS.
    from fractions import Fraction
    from PIL import Image
    raw=subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries',
        'stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,duration','-of','json',str(path)],text=True)
    info=json.loads(raw)['streams'][0];fps=float(Fraction(info['avg_frame_rate']))
    if abs(fps-float(Fraction(info['r_frame_rate'])))>1e-6:raise ValueError('Constant-rate source metadata required')
    duration=float(info['duration']);total=int(info['nb_frames'])
    if abs(total/fps-duration)>1/fps+.001:raise ValueError('Source frame count/duration mismatch')
    indices=sample_indices(duration,fps,total,count);height=int(round(info['height']/info['width']*width/2)*2)
    frames=[];stamps=[]
    for index in indices:
        target=index/fps
        r=subprocess.run(['ffmpeg','-v','info','-threads','1','-copyts','-ss',str(target),'-i',str(path),
            '-frames:v','1','-vf',f'scale={width}:{height},showinfo','-vsync','0','-f','rawvideo','-pix_fmt','rgb24','pipe:1'],capture_output=True,check=True)
        times=re.findall(r'pts_time:([0-9.eE+-]+)',r.stderr.decode())
        if not times:raise ValueError('Decoder omitted source presentation time')
        stamp=float(times[0]);values=np.frombuffer(r.stdout,dtype=np.uint8)
        if abs(stamp-target)>max(.25,duration/count*.25):raise ValueError('Frame seek exceeded recorded tolerance')
        if values.size!=height*width*3:raise ValueError('Incomplete decoded frame')
        frames.append(Image.fromarray(values.reshape(height,width,3)));stamps.append(stamp)
    timestamp_metadata(stamps,duration)
    return stamps,frames,duration,{'fps':fps,'requested_source_indices':indices,
        'seek_errors_seconds':[s-i/fps for s,i in zip(stamps,indices,strict=True)]}


class NativeEncoder:
    def attention_context(self):
        if self.recipe.get('attention_backend','automatic')=='math':
            from torch.nn.attention import sdpa_kernel,SDPBackend
            return sdpa_kernel(SDPBackend.MATH)
        return nullcontext()

    def __init__(self,model_path,helper_path,recipe,device='auto'):
        import torch
        validate_recipe(recipe)
        resolved=('cuda' if torch.cuda.is_available() else 'cpu') if device=='auto' else device
        if resolved not in {'cpu','cuda'}:raise ValueError('Native inference supports cpu/cuda only')
        if resolved=='cuda' and not torch.cuda.is_available():raise ValueError('CUDA requested but unavailable')
        spec=importlib.util.spec_from_file_location('_packaged_native_qwen',helper_path)
        module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
        historical=recipe['representation']=='historical4'
        # Reproduce the pinned helper constructor with explicit placement and local-only
        # processor loading; its original constructor always auto-selects CUDA.
        e=module.Qwen3VLEmbedder.__new__(module.Qwen3VLEmbedder)
        e.max_length=2048;e.min_pixels=module.MIN_PIXELS;e.max_pixels=module.MAX_PIXELS
        e.total_pixels=4*224*224 if historical else 16*256*448
        e.fps=module.FPS;e.num_frames=e.max_frames=4 if historical else 16
        e.default_instruction="Represent the user's input."
        if recipe['representation']=='h1004':
            e.total_pixels=4*256*448;e.min_pixels=4*32*32;e.max_pixels=1800*32*32
            e.num_frames=e.max_frames=4;e.fps=1
        e.model=module.Qwen3VLForEmbedding.from_pretrained(str(model_path.resolve()),
            dtype=torch.bfloat16 if resolved=='cuda' else torch.float32,attn_implementation='sdpa',
            local_files_only=True,trust_remote_code=False).to(resolved).eval()
        e.processor=module.Qwen3VLProcessor.from_pretrained(str(model_path.resolve()),padding_side='right',local_files_only=True)
        self.embedder=e;self.recipe=recipe;self.device=resolved

    def decode(self,path):
        if self.recipe['representation']=='historical4':
            stamps=frame_times(path,4);return stamps,[read_frame(path,t) for t in stamps],None,{'clock':'historical helper list defaults'}
        return decode_source_times(path,self.recipe['frames'])

    def videos(self,items):
        """Batch the h1004 recipe with the same neighbours as cached extraction."""
        if self.recipe['representation']!='h1004':
            rows=[self.video(item) for item in items];return np.stack([x[0] for x in rows]),[x[1] for x in rows]
        import torch
        from qwen_vl_utils import process_vision_info
        e=self.embedder;conversations=[];actual=[];rows=[]
        for stamps,frames,duration,decoder in items:
            conversation=e.format_model_input(video=frames,instruction=DOC_INSTRUCTION)
            conversation[1]['content'][0].update(resized_height=256,resized_width=448,min_pixels=4*32*32)
            metadata=timestamp_metadata(stamps,duration)
            times=e.processor._calculate_timestamps(metadata['frames_indices'].copy(),metadata['fps'])
            if not np.allclose(times,[(stamps[i]+stamps[i+1])/2 for i in (0,2)],atol=1e-9):raise ValueError('Video clock changed')
            conversations.append(conversation);actual.append(metadata)
            rows.append({'sampled_timestamps':stamps,'processor_pair_times':times,'actual_clock':True,**decoder})
        texts=e.processor.apply_chat_template(conversations,tokenize=False,add_generation_prompt=True)
        images,video,kwargs=process_vision_info(conversations,image_patch_size=16,return_video_metadata=True,return_video_kwargs=True)
        if images is not None or video is None or len(video)!=len(items):raise ValueError('Missing batched pixels')
        batch=e.processor(text=texts,images=None,videos=[x[0] for x in video],video_metadata=actual,padding=True,
            truncation=False,do_resize=False,return_tensors='pt',**kwargs)
        if 'pixel_values_videos' not in batch or not batch['pixel_values_videos'].numel():raise ValueError('Missing pixels: text fallback is forbidden')
        tokens=batch['attention_mask'].sum(1).tolist()
        if max(tokens)>2048:raise ValueError('Batched context cap exceeded')
        for row,count,grid in zip(rows,tokens,batch['video_grid_thw'].tolist(),strict=True):row.update(sequence_tokens=count,video_grid=grid)
        batch={k:v.to(e.model.device) for k,v in batch.items()}
        with torch.inference_mode(),self.attention_context():
            hidden=e.model(**batch,use_cache=False).last_hidden_state
            vectors=normalize(e._pooling_last(hidden,batch['attention_mask']).float().cpu().numpy())
        return vectors,rows

    def video(self,decoded):
        import torch
        stamps,frames,duration,decoder=decoded;e=self.embedder;recipe=self.recipe
        if recipe['representation']=='historical4':
            batch=e._preprocess_inputs([e.format_model_input(video=frames,instruction=DOC_INSTRUCTION)])
            clock={'actual_clock':False}
        else:
            from qwen_vl_utils.vision_process import process_vision_info
            e.num_frames=e.max_frames=len(frames)
            conversations=[e.format_model_input(video=frames,instruction=DOC_INSTRUCTION)]
            conversations[0][1]['content'][0].update(resized_height=recipe['height'],resized_width=recipe['width'],min_pixels=4*32*32)
            text=e.processor.apply_chat_template(conversations,add_generation_prompt=True,tokenize=False)
            images,inputs,kwargs=process_vision_info(conversations,image_patch_size=16,return_video_metadata=True,return_video_kwargs=True)
            if images is not None or inputs is None or len(inputs)!=1:raise ValueError('Unexpected visual inputs')
            pixels,synthetic=inputs[0];actual=recipe['representation']!='synthetic4'
            metadata=timestamp_metadata(stamps,duration) if actual else synthetic
            pair_times=e.processor._calculate_timestamps(metadata['frames_indices'].copy(),metadata['fps'])
            expected=[(stamps[i]+stamps[i+1])/2 for i in range(0,len(stamps),2)]
            if actual and not np.allclose(pair_times,expected,atol=1e-9):raise ValueError('Timestamp metadata changed')
            batch=e.processor(text=text,images=None,videos=[pixels],video_metadata=[metadata],truncation=False,
                padding=True,do_resize=False,return_tensors='pt',**kwargs)
            clock={'actual_clock':actual,'processor_pair_times':pair_times}
        if 'pixel_values_videos' not in batch or not batch['pixel_values_videos'].numel():raise ValueError('Missing pixels: text fallback is forbidden')
        patches=int(batch['video_grid_thw'].prod(dim=-1).sum());tokens=int(batch['attention_mask'].sum())
        if tokens>2048:raise ValueError('Video exceeds context budget')
        batch={k:v.to(e.model.device) for k,v in batch.items()}
        with torch.inference_mode(),self.attention_context():
            if recipe['representation']=='h1004':
                output={'last_hidden_state':e.model(**batch,use_cache=False).last_hidden_state,'attention_mask':batch['attention_mask']}
            else:output=e.forward(batch)
            vector=normalize(e._pooling_last(output['last_hidden_state'],output['attention_mask']).float().cpu().numpy())[0]
        return vector,{'sampled_timestamps':stamps,'visual_patch_tokens':patches,'sequence_tokens':tokens,**decoder,**clock}

    def queries(self,texts):
        import torch
        with torch.inference_mode(),self.attention_context():
            vectors=self.embedder.process([{'text':t,'instruction':QUERY_INSTRUCTION} for t in texts])
        return normalize(vectors.float().cpu().numpy())
