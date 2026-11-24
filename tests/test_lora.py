import tempfile
import unittest
import torch
from torch import nn
from baseline.lora import attach_language_lora, adapter_state, load_adapter_state, multi_positive_loss
from baseline.lora_data import check_partitions, sample_batch


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = nn.Module();self.visual.q_proj=nn.Linear(4,4)
        self.language_model=nn.Module();self.language_model.layers=nn.ModuleList([nn.Module()])
        layer=self.language_model.layers[0];layer.self_attn=nn.Module()
        layer.self_attn.q_proj=nn.Linear(4,4);layer.self_attn.v_proj=nn.Linear(4,4)

    def forward(self, x):
        a=self.language_model.layers[0].self_attn
        return a.q_proj(x)+a.v_proj(x)


class LoRATests(unittest.TestCase):
    def test_noop_then_optimizer_changes_adapters_only_and_roundtrips(self):
        torch.manual_seed(43);model=TinyModel();x=torch.randn(3,4)
        original=model(x).detach().clone();visual=model.visual.q_proj.weight.detach().clone()
        targets=attach_language_lora(model,rank=2,alpha=4)
        self.assertEqual(len(targets),2);torch.testing.assert_close(model(x),original,rtol=0,atol=0)
        frozen={name:p.detach().clone() for name,p in model.named_parameters() if not p.requires_grad}
        opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.01)
        model(x).square().mean().backward();opt.step()
        self.assertFalse(torch.equal(model(x),original))
        for name,p in model.named_parameters():
            if name in frozen:
                torch.testing.assert_close(p,frozen[name],rtol=0,atol=0);self.assertIsNone(p.grad)
        torch.testing.assert_close(model.visual.q_proj.weight,visual,rtol=0,atol=0)
        state=adapter_state(model);expected=model(x).detach().clone()
        for name,p in model.named_parameters():
            if p.requires_grad:p.data.zero_()
        load_adapter_state(model,state);torch.testing.assert_close(model(x),expected,rtol=0,atol=0)
        with self.assertRaises(ValueError):attach_language_lora(model,rank=2)
        with self.assertRaises(ValueError):load_adapter_state(model,{})

    def test_positive_and_ambiguous_cells_are_not_negatives(self):
        scores=torch.tensor([[.3,.7,100.,.2]],requires_grad=True)
        positive=torch.tensor([[True,True,False,False]])
        negative=torch.tensor([[False,False,False,True]])
        loss=multi_positive_loss(scores,positive,negative);loss.backward()
        self.assertEqual(scores.grad[0,2].item(),0)
        self.assertTrue((scores.grad[0,:2]<0).all())
        self.assertGreater(scores.grad[0,3].item(),0)
        with self.assertRaises(ValueError):multi_positive_loss(scores,positive,positive)

    def test_batches_obey_video_exclusion_and_all_known_positives(self):
        qrels={'q1':{'relevant_video_ids':['a','b','held']},'q2':{'relevant_video_ids':['b']}}
        batch=sample_batch(['q1','q2'],['a','b','c','d','e'],qrels,query_batch=2,excluded={'q1':['c']})
        self.assertNotIn('held',batch['video_ids'])
        for qid,p,n in zip(batch['query_ids'],batch['positive_mask'],batch['negative_mask']):
            for vid,isp,isn in zip(batch['video_ids'],p,n):
                self.assertEqual(isp,vid in qrels[qid]['relevant_video_ids'])
                if isp:self.assertFalse(isn)
                if qid=='q1' and vid=='c':self.assertFalse(isn)

    def test_relevance_group_and_video_leakage_fail(self):
        parent={'query_groups':{'fit':['a'],'dev':['b']},'development_video_groups':[['v1'],['v2']]}
        with self.assertRaises(ValueError):check_partitions(parent,{'a':{'relevant_video_ids':['v1']},'b':{'relevant_video_ids':['v1']}})
        parent['development_video_groups']=[['v1'],['v1']]
        with self.assertRaises(ValueError):check_partitions(parent,{'a':{'relevant_video_ids':['v1']},'b':{'relevant_video_ids':['v2']}})


if __name__=='__main__':unittest.main()
