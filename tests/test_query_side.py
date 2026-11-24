"""Query-side experiment isolation and gating; all inputs here are synthetic."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from scripts.query_side import training_masks, comparison_gate, evaluate_query_side, query_loss
from tests.fixtures import fixture, SETTINGS


class QuerySideTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_ambiguous_and_known_positive_pairs_are_never_negatives(self):
        qrels = {'a': {'relevant_video_ids': ['v1', 'held']}, 'b': {'relevant_video_ids': ['held']}}
        pos, neg, keep = training_masks(['a', 'b'], ['v1', 'v2', 'v3'], qrels, {'a': ['v2', 'held']})
        self.assertEqual(pos.tolist(), [[True, False, False], [False, False, False]])
        self.assertEqual(neg.tolist(), [[False, False, True], [True, True, True]])
        self.assertFalse((pos & neg).any())
        self.assertEqual(keep.tolist(), [True, False])

    def test_share_weighted_microbatches_reproduce_full_batch_gradient(self):
        g = torch.Generator().manual_seed(0)
        weights = torch.randn(4, 3, generator=g, requires_grad=True)
        inputs = torch.randn(6, 4, generator=g); videos = torch.nn.functional.normalize(torch.randn(5, 3, generator=g), dim=-1)
        pos = torch.zeros(6, 5, dtype=torch.bool); pos[torch.arange(6), torch.arange(6) % 5] = True
        neg = ~pos; anchors = torch.randn(6, 3, generator=g)
        settings = {'temperature': .07, 'anchor_weight': .1}
        encode = lambda ix: torch.nn.functional.normalize(inputs[ix] @ weights, dim=-1)
        query_loss(encode(slice(None)), videos, pos, neg, anchors, settings).backward()
        full = weights.grad.clone(); weights.grad = None
        for ix in [slice(0, 2), slice(2, 6)]:
            n = ix.stop - ix.start
            (query_loss(encode(ix), videos, pos[ix], neg[ix], anchors[ix], settings) * n / 6).backward()
        torch.testing.assert_close(weights.grad, full)

    def test_gate_requires_mean_gain_majority_no_regressing_fold_and_map(self):
        row = lambda fold, d, m=.0: {'fold': fold, 'r_precision_delta': d, 'map_delta': m}
        self.assertTrue(comparison_gate([row(0, .01), row(0, .01), row(1, .005), row(1, -.001)], .001)['gate_passed'])
        self.assertFalse(comparison_gate([row(0, .02), row(0, .02), row(1, -.001), row(1, -.001)], .001)['gate_passed'])
        self.assertFalse(comparison_gate([row(0, .05), row(0, -.001), row(0, -.001)], .001)['gate_passed'])
        self.assertFalse(comparison_gate([row(0, .01, -.01), row(0, .01, -.01)], .001)['gate_passed'])
        self.assertFalse(comparison_gate([row(0, .0005), row(0, .0005)], .001)['gate_passed'])

    def test_identical_query_vectors_never_pass_and_cells_are_complete(self):
        parent, qrels, qids, vids, q, v = fixture()
        plan = {'instructions': {'canonical': 'c', 'alt': 'a'}, 'adapter': {'held_fold': 0},
                'instruction_folds': [0, 1], 'head_seeds': [43, 44], 'raw_weights': [0, .25, 1],
                'primary_raw_weight': .25, 'head_epochs': 2, 'head_training': SETTINGS, 'head_device': 'cpu',
                'minimum_rp_gain': .001, 'reference_comparison': {'arm_raw_weight': 1, 'frozen_raw_weight': .25}}
        def vectors(name):
            return q.copy()
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / 'evaluate'; (out / 'results').mkdir(parents=True)
            (Path(directory) / 'query-manifest.json').write_text(json.dumps({'query_ids': qids}))
            def verified(root, job, name):
                return Path(directory) / name
            with patch('scripts.query_side.lora_pilot.inputs', return_value=(parent, {}, qrels)), \
                 patch('scripts.query_side.frozen', return_value=({'query_ids': qids, 'video_ids': vids}, q, v, {})), \
                 patch('scripts.query_side.verified_result', side_effect=verified), \
                 patch('scripts.query_side.load_vectors', side_effect=vectors), \
                 patch('scripts.query_side.finish') as finished:
                evaluate_query_side(out, plan)
            report = finished.call_args.args[2]
            cells = json.loads((out / 'results/cells.json').read_text())
        self.assertFalse(report['gate_passed'])
        self.assertEqual(set(report['arms']), {'query-lora', 'alt'})
        self.assertEqual(len(report['arms']['query-lora']['comparisons']), 2 * 3)
        self.assertEqual(len(report['arms']['alt']['comparisons']), 2 * 2 * 3)
        self.assertTrue(all(r['r_precision_delta'] == 0 for arm in report['arms'].values() for r in arm['comparisons']))
        self.assertEqual(len([c for c in cells if c['arm'] == 'frozen']), 2 * 2 * 3)
        reference = report['reference_comparison']
        self.assertEqual([(r['fold'], r['seed']) for r in reference['comparisons']], [(0, 43), (0, 44)])
        self.assertEqual(report['gate_passed'], reference['gate_passed'])


if __name__ == '__main__':
    unittest.main()
