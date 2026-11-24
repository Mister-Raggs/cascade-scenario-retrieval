"""Packaged structured recipe matches the validated research harness; synthetic inputs only."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from baseline.native_ranking import blended_scores
from baseline.probe import fit_logistic, standardize
from baseline.structured_recipe import BLOCKS, Probes, fuse, structured_scores, window_feature
from scripts.structured_eval import atom_probabilities, late_probabilities


class StructuredRecipeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        rng = np.random.default_rng(0); self.n = 80
        self.y = np.stack([rng.random(self.n) < .3, rng.random(self.n) < .5, np.zeros(self.n, bool)], 1)
        self.x = {b: rng.normal(size=(self.n, d)) + 2 * self.y[:, :1] for b, d in zip(BLOCKS, (6, 4, 8))}
        self.atoms = ['ego|action=reverse', 'agent|type=car', 'obj|type=stop_sign']

    def probes(self, directory):
        out = {'atoms': np.asarray(self.atoms), 'prior': np.clip(self.y.mean(0), 1e-3, 1 - 1e-3), 'is_prior': np.asarray([False, False, True])}
        for b in BLOCKS:
            x = self.x[b]; mean, std = x.mean(0), x.std(0) + 1e-6; z = (x - mean) / std
            w = np.zeros((x.shape[1], 3), np.float32); c = np.zeros(3, np.float32)
            for j in range(2):
                w[:, j], c[j] = fit_logistic(z, self.y[:, j])
            out.update({f'{b}_mean': mean, f'{b}_std': std, f'{b}_weight': w, f'{b}_bias': c})
        path = Path(directory) / 'p.npz'; np.savez(path, **out); return Probes(path)

    def test_packaged_probabilities_match_research_late_fusion(self):
        with tempfile.TemporaryDirectory() as d:
            probes = self.probes(d)
        packaged = probes.probabilities(self.x)
        rows = np.arange(self.n)
        for j in range(2):
            research = late_probabilities([self.x[b] for b in BLOCKS], rows, rows, self.y[:, j])
            np.testing.assert_allclose(packaged[:, j], research, rtol=1e-4, atol=1e-6)
        self.assertTrue(np.allclose(packaged[:, 2], packaged[0, 2]))

    def test_fusion_matches_research_formula_and_unparsed_keep_submitted(self):
        with tempfile.TemporaryDirectory() as d:
            probes = self.probes(d)
        p = probes.probabilities(self.x)
        texts = ['The ego vehicle reverses while a car is visible.', 'A unicorn juggles.']
        scores, mask = structured_scores(texts, p, probes)
        self.assertEqual(mask.tolist(), [True, False])
        np.testing.assert_allclose(scores[0], np.log(p[:, 0]) + np.log(p[:, 1]))
        rng = np.random.default_rng(1); raw = rng.random((2, self.n)); submitted = rng.random((2, self.n))
        fused = fuse(raw, submitted, scores, mask)
        np.testing.assert_allclose(fused[0], blended_scores(scores[:1], submitted[:1], .5)[0])
        np.testing.assert_allclose(fused[1], submitted[1])
        self.assertEqual(window_feature(np.ones((5, 3))).shape, (6,))


if __name__ == '__main__':
    unittest.main()
