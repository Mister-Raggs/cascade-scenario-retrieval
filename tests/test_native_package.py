"""Synthetic packaging contract checks; no real model inference or challenge scores."""
import json,tempfile,unittest
from pathlib import Path
import torch
from safetensors.torch import save_file
from baseline.io import sha256_file,write_json
from baseline.native_video import DOC_INSTRUCTION,QUERY_INSTRUCTION
from baseline.projection import SharedProjection
from scripts.package_native_local import stage,freeze_recipe,PROJECT
from scripts.package_local import local_build_command

class NativePackagingTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.model=self.root/'model';self.model.mkdir()
        for n in ['config.json','tokenizer.json','preprocessor_config.json','video_preprocessor_config.json']:write_json(self.model/n,{'synthetic_fixture':True})
        (self.model/'model.safetensors').write_bytes(b'synthetic fixture, not real pretrained weights')
        files={p.name:sha256_file(p) for p in self.model.iterdir()}
        spec=json.loads((PROJECT/'configs/retrieval_models.json').read_text())['qwen3-video']
        write_json(self.model/'model_provenance.json',{'spec':spec,'files':files})
        (self.model/'token').write_text('must never enter package')
        self.helper=PROJECT/'vendor/qwen3-vl-embedding/qwen3_vl_embedding.py'
        self.features=self.root/'extraction/features';self.features.mkdir(parents=True)
        extraction=self.features.parent/'experiment_manifest.json'
        write_json(extraction,{'extra':{'frames':4,'total_pixels':4*224*224,'max_length':2048,
            'document_instruction':DOC_INSTRUCTION,'query_instruction':QUERY_INSTRUCTION,
            'helper_sha256':sha256_file(self.helper),'model_provenance_sha256':sha256_file(self.model/'model_provenance.json')}})
        write_json(self.features/'manifest.json',{'experiment_manifest_sha256':sha256_file(extraction)})
        self.training=self.root/'training.json';write_json(self.training,{'feature_manifest_sha256':sha256_file(self.features/'manifest.json')})
        self.checkpoint=self.root/'head';self.checkpoint.mkdir()
        save_file(SharedProjection(2048,128).state_dict(),str(self.checkpoint/'adapter.safetensors'))
        write_json(self.checkpoint/'metadata.json',{'manifest_sha256':sha256_file(self.training),'weights_sha256':sha256_file(self.checkpoint/'adapter.safetensors')})
        (self.checkpoint/'qrels.jsonl').write_text('must not enter package')
    def test_context_allowlist_contract_and_no_push(self):
        out=self.root/'context';result=stage(out,self.model,self.checkpoint,self.training,self.features)
        self.assertFalse((out/'model/token').exists());self.assertFalse((out/'checkpoint/qrels.jsonl').exists())
        self.assertFalse((out/'data').exists());self.assertFalse((out/'scripts').exists())
        self.assertEqual(result['recipe']['adapter_sha256'],sha256_file(self.checkpoint/'adapter.safetensors'))
        self.assertIn('--no-push',local_build_command(out))
        import yaml
        config=yaml.safe_load((out/'submission.yaml').read_text());self.assertTrue(config['entrypoint'][0].startswith('/'))
        self.assertFalse({'INPUT_DIR','OUTPUT_DIR','TOP_K'}&set(config['env']))
        self.assertFalse(result['built']);self.assertFalse(result['official_submission_made'])
    def test_mismatched_feature_parent_and_changed_model_fail_before_staging(self):
        write_json(self.features/'manifest.json',{'experiment_manifest_sha256':'changed'})
        with self.assertRaisesRegex(ValueError,'feature parent'):freeze_recipe(self.checkpoint,self.training,self.features,'historical4',self.model,self.helper)
        # Restore the original parent, then corrupt one declared model asset.
        write_json(self.features/'manifest.json',{'experiment_manifest_sha256':sha256_file(self.features.parent/'experiment_manifest.json')})
        (self.model/'model.safetensors').write_bytes(b'changed')
        out=self.root/'context'
        with self.assertRaisesRegex(ValueError,'asset changed'):stage(out,self.model,self.checkpoint,self.training,self.features)
        self.assertFalse(out.exists())
    def test_h100_sharded_8b_package_preserves_fusion_and_rejects_missing_shard(self):
        spec=json.loads((PROJECT/'configs/retrieval_models.json').read_text())['embed-8b']
        (self.model/'model.safetensors').unlink()
        (self.model/'model-00001-of-00001.safetensors').write_bytes(b'synthetic shard')
        write_json(self.model/'model.safetensors.index.json',{'weight_map':{'fake':'model-00001-of-00001.safetensors'}})
        files={p.name:sha256_file(p) for p in self.model.iterdir() if p.name not in {'token','model_provenance.json'}}
        write_json(self.model/'model_provenance.json',{'spec':spec,'files':files})
        fm=self.features/'feature_manifest.json'
        write_json(fm,{'frames':4,'height':256,'width':448,'actual_clock':True,'dimension':4096,'model_provenance_sha256':sha256_file(self.model/'model_provenance.json')})
        write_json(self.training,{'feature_manifest_sha256':sha256_file(fm),'helper_sha256':sha256_file(self.helper),'raw_rank_weight':.25})
        save_file(SharedProjection(4096,128).state_dict(),str(self.checkpoint/'adapter.safetensors'))
        write_json(self.checkpoint/'metadata.json',{'manifest_sha256':sha256_file(self.training),'weights_sha256':sha256_file(self.checkpoint/'adapter.safetensors'),'representation':'h1004'})
        out=self.root/'sharded-context';result=stage(out,self.model,self.checkpoint,self.training,self.features,'h1004')
        self.assertEqual(result['recipe']['dimension'],4096)
        self.assertEqual(result['recipe']['raw_rank_weight'],.25)
        self.assertEqual(result['recipe']['video_batch_size'],4)
        self.assertTrue((out/'baseline/native_ranking.py').exists())
        self.assertFalse((out/'checkpoint/qrels.jsonl').exists())
        (self.model/'model-00001-of-00001.safetensors').unlink()
        with self.assertRaises(ValueError):stage(self.root/'broken-context',self.model,self.checkpoint,self.training,self.features,'h1004')
