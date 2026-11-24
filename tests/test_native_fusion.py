import unittest
import numpy as np
from baseline.native_ranking import blended_scores
from baseline.native_video import validate_recipe,DOC_INSTRUCTION,QUERY_INSTRUCTION

class NativeFusionTests(unittest.TestCase):
    def test_full_corpus_fusion_matches_research_reciprocal_ranks(self):
        from scripts.refine_fusion import reciprocal_matrix
        raw=np.array([[.9,.7,.1,.4]],dtype=np.float32);head=np.array([[.4,.1,.8,.7]],dtype=np.float32)
        expected=.25*reciprocal_matrix(np.argsort(-raw,axis=1,kind='stable'),4)+.75*reciprocal_matrix(np.argsort(-head,axis=1,kind='stable'),4)
        np.testing.assert_array_equal(blended_scores(raw,head,.25),expected)
        np.testing.assert_array_equal(blended_scores(raw,head,0),head)
        np.testing.assert_array_equal(blended_scores(raw,head,1),raw)
        with self.assertRaises(ValueError):blended_scores(raw,np.full_like(head,np.nan),.25)

    def test_model_dimensions_and_blend_are_checked_together(self):
        recipe={'representation':'h1004','frames':4,'height':256,'width':448,'dimension':4096,'max_length':2048,
            'model_id':'Qwen/Qwen3-VL-Embedding-8B','document_instruction':DOC_INSTRUCTION,'query_instruction':QUERY_INSTRUCTION,'raw_rank_weight':.25}
        validate_recipe(recipe)
        with self.assertRaises(ValueError):validate_recipe({**recipe,'model_id':'Qwen/Qwen3-VL-Embedding-2B'})
        with self.assertRaises(ValueError):validate_recipe({**recipe,'raw_rank_weight':float('nan')})
        with self.assertRaises(ValueError):validate_recipe({**recipe,'dimension':2048})

if __name__=='__main__':unittest.main()
