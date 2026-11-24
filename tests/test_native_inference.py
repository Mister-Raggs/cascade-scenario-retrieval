"""Synthetic contract checks; these do not measure challenge retrieval accuracy."""
import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from safetensors.torch import save_file
from baseline.io import sha256_file,write_json,write_jsonl,unique_rows
from baseline.projection import SharedProjection
from baseline.native_run import run,load_projection
from baseline.native_video import DOC_INSTRUCTION,QUERY_INSTRUCTION,NativeEncoder

class PixelFixtureEncoder:
    """Synthetic encoding from current file bytes, with arbitrary caller IDs."""
    def __init__(self,*args):self.device='cpu'
    def decode(self,path):return path.read_bytes()
    def video(self,content):
        v=np.zeros(2048,dtype=np.float32);v[0 if content==b'first' else 1]=1
        return v,{'source_kind':'synthetic-fixture'}
    def queries(self,texts):
        values=np.zeros((len(texts),2048),dtype=np.float32)
        for i,t in enumerate(texts):values[i,0 if t=='first' else 1]=1
        return values

class NativeContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.input=self.root/'readonly-input';self.output=self.root/'output';self.input.mkdir()
        self.a=self.input/'one.mp4';self.a.write_bytes(b'first');(self.input/'two.mp4').write_bytes(b'second')
        write_jsonl(self.input/'videos.jsonl',[{'video_id':'unfamiliar-z','path':'one.mp4'},{'video_id':'unfamiliar-a','path':'two.mp4'}])
        write_jsonl(self.input/'queries.jsonl',[{'query_id':'new-q','text':'first'}])
        (self.input/'qrels.jsonl').write_text('POISON: must not be read')
        self.helper=self.root/'helper.py';self.helper.write_text('# synthetic fixture')
        self.checkpoint=self.root/'checkpoint';self.checkpoint.mkdir();self.weights=self.checkpoint/'adapter.safetensors'
        torch.manual_seed(4);save_file(SharedProjection(2048,128).state_dict(),str(self.weights))
        write_json(self.checkpoint/'metadata.json',{'weights_sha256':sha256_file(self.weights)})
        self.recipe={'representation':'historical4','frames':4,'height':256,'width':448,'dimension':2048,'rank':128,'max_length':2048,
            'document_instruction':DOC_INSTRUCTION,'query_instruction':QUERY_INSTRUCTION,'adapter_sha256':sha256_file(self.weights),'helper_sha256':sha256_file(self.helper)}
        self.config=self.root/'recipe.json';write_json(self.config,self.recipe)
    def invoke(self,**kwargs):
        return run(self.input,self.output,model_path=self.root/'model',helper_path=self.helper,checkpoint=self.checkpoint,
            recipe_path=self.config,device='cpu',encoder_factory=PixelFixtureEncoder,**kwargs)
    def test_new_ids_current_pixels_cap_reordering_and_read_only_inputs(self):
        original={p.name:sha256_file(p)for p in self.input.iterdir()}
        for p in self.input.iterdir():p.chmod(0o444)
        self.input.chmod(0o555)
        self.addCleanup(self.input.chmod,0o755)
        self.invoke(top_k=1)
        prediction=unique_rows(self.output/'predictions.jsonl','query_id')['new-q'];self.assertEqual(prediction['video_ids'],['unfamiliar-z'])
        self.assertEqual(original,{p.name:sha256_file(p)for p in self.input.iterdir()})
        self.input.chmod(0o755);self.a.chmod(0o644);self.a.write_bytes(b'second')
        # Both byte-derived vectors tie; lexical video ID breaks ties independently of manifest order.
        self.invoke(top_k=100);self.assertEqual(unique_rows(self.output/'predictions.jsonl','query_id')['new-q']['video_ids'],['unfamiliar-a','unfamiliar-z'])
        write_jsonl(self.input/'videos.jsonl',[{'video_id':'unfamiliar-a','path':'two.mp4'},{'video_id':'unfamiliar-z','path':'one.mp4'}])
        self.invoke(top_k=100);self.assertEqual(unique_rows(self.output/'predictions.jsonl','query_id')['new-q']['video_ids'],['unfamiliar-a','unfamiliar-z'])
    def test_zero_cap_and_empty_corpus_avoid_model_loading(self):
        with patch.object(PixelFixtureEncoder,'__init__',side_effect=AssertionError('must not load')):
            self.invoke(top_k=0)
            write_jsonl(self.input/'videos.jsonl',[]);self.invoke(top_k=3)
        self.assertEqual(unique_rows(self.output/'predictions.jsonl','query_id')['new-q']['video_ids'],[])
    def test_batch_recipe_processes_partial_tail_without_changing_rankings(self):
        class BatchFixture(PixelFixtureEncoder):
            def decode(self,path):
                import time
                value=path.read_bytes()
                if value==b'second':time.sleep(.02)
                return value
            def videos(self,items):
                rows=[self.video(item) for item in items]
                return np.stack([x[0] for x in rows]),[x[1] for x in rows]
        self.recipe['video_batch_size']=4;self.recipe['decode_workers']=4;self.recipe['raw_rank_weight']=.25;write_json(self.config,self.recipe)
        result=run(self.input,self.output,model_path=self.root/'model',helper_path=self.helper,checkpoint=self.checkpoint,
            recipe_path=self.config,device='cpu',encoder_factory=BatchFixture,top_k=1)
        self.assertEqual(result['num_videos'],2)
        self.assertEqual(unique_rows(self.output/'predictions.jsonl','query_id')['new-q']['video_ids'],['unfamiliar-z'])
    def test_duplicate_ids_escape_paths_and_modified_assets_rejected(self):
        write_jsonl(self.input/'queries.jsonl',[{'query_id':'dup','text':'a'}]*2)
        with self.assertRaisesRegex(ValueError,'duplicate query_id'):self.invoke(top_k=1)
        write_jsonl(self.input/'queries.jsonl',[{'query_id':'q','text':'a'}])
        write_jsonl(self.input/'videos.jsonl',[{'video_id':'v','path':'../outside.mp4'}])
        with self.assertRaisesRegex(ValueError,'relative video path'):self.invoke(top_k=1)
        self.helper.write_text('changed')
        with self.assertRaisesRegex(ValueError,'helper changed'):self.invoke(top_k=1)
    def test_checkpoint_hash_and_shape_checked(self):
        changed={**self.recipe,'adapter_sha256':'not the weights'}
        with self.assertRaisesRegex(ValueError,'weights differ'):load_projection(self.checkpoint,changed,'cpu')
        bad=SharedProjection(2048,8);save_file(bad.state_dict(),str(self.weights))
        self.recipe['adapter_sha256']=sha256_file(self.weights);write_json(self.checkpoint/'metadata.json',{'weights_sha256':sha256_file(self.weights)})
        with self.assertRaises(RuntimeError):load_projection(self.checkpoint,self.recipe,'cpu')
    def test_missing_visual_tensors_do_not_fall_back_to_text(self):
        class Fake:
            def format_model_input(self,**kwargs):return []
            def _preprocess_inputs(self,*args):return {'attention_mask':torch.ones(1,3)}
        encoder=NativeEncoder.__new__(NativeEncoder);encoder.embedder=Fake();encoder.recipe=self.recipe
        with self.assertRaisesRegex(ValueError,'Missing pixels'):encoder.video(([1,2,3,4],[object()]*4,None,{}))
