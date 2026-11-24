"""Packaged entry-3 recipe matches the research code paths; synthetic inputs only."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from baseline.entry3_recipe import EgoNet, late, structured_scores, t4_probabilities, with_vlm
from baseline.probe import fit_logistic, standardize
from baseline.structured_recipe import BLOCKS
from scripts.fit_entry3 import train_motionnet
from scripts.structured_v2 import late as research_late, logistic_prob, motionnet_logits


class Entry3RecipeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        rng = np.random.default_rng(0); self.n = 60
        self.y = np.stack([rng.random(self.n) < .4, rng.random(self.n) < .3], 1)
        self.series = (rng.normal(size=(self.n, 20, 3)) + self.y[:, :1, None]).astype(np.float32)
        self.fit, self.held = np.arange(40), np.arange(40, 60)

    def ego_data(self):
        conv, head, mean, std, predict = train_motionnet(self.series, self.fit, self.y, 0, 3)
        oof = predict(self.fit); cal_mean, cal_std = oof.mean(0), oof.std(0) + 1e-6; z = (oof - cal_mean) / cal_std
        w0, b0 = fit_logistic(z, self.y[self.fit, 0])
        return {'ego_atoms': np.asarray(['ego|action=stop', 'ego|action=uturn']), 'ego_series_length': np.asarray(18),
                'ego_mean': mean, 'ego_std': std, 'ego_conv0_weight': conv[0].weight.detach().numpy(), 'ego_conv0_bias': conv[0].bias.detach().numpy(),
                'ego_conv2_weight': conv[2].weight.detach().numpy(), 'ego_conv2_bias': conv[2].bias.detach().numpy(),
                'ego_head_weight': head.weight.detach().numpy(), 'ego_head_bias': head.bias.detach().numpy(), 'ego_cal_mean': cal_mean,
                'ego_cal_std': cal_std, 'ego_cal_weight': np.stack([w0, np.zeros(2)]), 'ego_cal_bias': np.asarray([b0, 0.]),
                'ego_cal_prior': np.asarray([0., .02]), 'ego_cal_is_prior': np.asarray([False, True])}, oof

    def test_trained_cnn_equals_research_logits(self):
        _, _, _, _, predict = train_motionnet(self.series, self.fit, self.y, 0, 3)
        self.assertTrue(np.array_equal(predict(self.held), motionnet_logits(self.series, self.fit, self.held, self.y, 0, 3)))

    def test_ego_net_truncates_and_calibrates_like_research(self):
        data, oof = self.ego_data(); net = EgoNet(data)
        short = self.series[:, :18]  # the package truncates to the stored research length
        logits = np.zeros((self.n, 2), np.float32); logits[self.fit] = oof
        logits[self.held] = net.logits(list(short[self.held]))
        want = logistic_prob(logits, self.fit, self.held, self.y[:, 0])
        got = net.probabilities(list(self.series[self.held]))
        self.assertLess(np.abs(got['ego|action=stop'] - want).max(), 1e-5)
        self.assertTrue(np.allclose(got['ego|action=uturn'], .02))

    def test_late_fusion_scores_and_vlm(self):
        rng = np.random.default_rng(1); p, q = rng.uniform(.1, .9, 5), rng.uniform(.1, .9, 5)
        self.assertTrue(np.allclose(late([p, q]), research_late([p, q])))
        fused = with_vlm({'a': p, 'b': q}, {'a': q, 'z': p})
        self.assertTrue(np.allclose(fused['a'], late([p, q])) and np.array_equal(fused['b'], q) and 'z' not in fused)
        scores, mask = structured_scores(['The ego vehicle stops.', 'zzz unparseable'], {'ego|action=stop': p})
        self.assertTrue(mask[0] and not mask[1] and np.allclose(scores[0], np.log(p)) and not scores[1].any())

    def test_t4_adds_ego_cnn_only_to_its_atoms(self):
        data, _ = self.ego_data(); rng = np.random.default_rng(2)
        feats = {b: rng.normal(size=(self.n, d)) for b, d in zip(BLOCKS, (4, 3, 5))}
        atoms = ['ego|action=stop', 'agent|type=car']
        for b in BLOCKS:
            z = standardize(feats[b], self.fit); w, c = fit_logistic(z[self.fit], self.y[self.fit, 0])
            data.update({f'{b}_mean': feats[b][self.fit].mean(0), f'{b}_std': feats[b][self.fit].std(0) + 1e-6,
                         f'{b}_weight': np.stack([w, w], 1).astype(np.float32), f'{b}_bias': np.asarray([c, c], np.float32)})
        data.update(atoms=np.asarray(atoms), prior=np.asarray([.4, .3]), is_prior=np.asarray([False, False]))
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'a.npz'; np.savez(path, **data)
            out = t4_probabilities(path, {b: feats[b][self.held] for b in BLOCKS}, list(self.series[self.held]))
        self.assertFalse(np.allclose(out['ego|action=stop'], out['agent|type=car']))


if __name__ == '__main__':
    unittest.main()
