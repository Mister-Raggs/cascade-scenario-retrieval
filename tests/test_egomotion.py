"""Ego-motion signal and label utilities; synthetic inputs only."""
import unittest

import numpy as np
import torch

from baseline.annotation_events import parse_time, events, window_labels
from baseline.egomotion import phase_shift, grid_flow, signals, clip_vector
from baseline.probe import auc, cross_validated


class EgomotionTests(unittest.TestCase):
    def test_phase_correlation_recovers_known_translation(self):
        rng = np.random.default_rng(0)
        base = rng.random((96, 96)).astype(np.float32)
        moved = np.roll(base, shift=(3, -5), axis=(0, 1))
        dy, dx, confidence = phase_shift(base, moved)
        self.assertAlmostEqual(dy, 3, delta=.2); self.assertAlmostEqual(dx, -5, delta=.2)
        self.assertGreater(confidence, .5)

    def test_uniform_pan_reads_as_yaw_not_expansion(self):
        rng = np.random.default_rng(1)
        frame = rng.random((120, 160)).astype(np.float32)
        frames = np.stack([np.roll(frame, 2 * i, axis=1) for i in range(4)])
        s = signals(grid_flow(frames, rows=3, cols=4))
        self.assertTrue(np.allclose(s[:, 0], 2, atol=.3))
        self.assertTrue(np.allclose(s[:, 1], 0, atol=.5))
        self.assertEqual(clip_vector(s, bins=3).shape, (6 * 7 + 6 * 3,))

    def test_timestamps_and_window_overlap_labels(self):
        self.assertEqual(parse_time('0:9.5'), 9.5); self.assertEqual(parse_time('1:02.0'), 62.0)
        self.assertEqual(parse_time('20.167', duration=20.0), 20.0); self.assertIsNone(parse_time(''))
        bundle = {'video': {'clip_id': 'c', 'duration_s': 20.0}, 'annotation': {
            'ego_vehicle': {'actions': [{'type': 'oxd:Stop', 'start_timestamp': '0:12.0', 'end_timestamp': '0:15.0'}]},
            'agents': [{'type': 'oxd:Car', 'visibility_start_timestamp': '', 'visibility_end_timestamp': '',
                        'actions': [{'action_type': 'fst:Park', 'start_timestamp': '0:0.0', 'end_timestamp': '0:4.0'}]}]}}
        _, _, items = events(bundle)
        self.assertIn(('ego_act:oxd:Stop', 12.0, 15.0), items)
        self.assertIn(('agent:oxd:Car', 0.0, 20.0), items)
        matrix = window_labels(items, [(0, 5), (10, 13), (15, 20)], ['ego_act:oxd:Stop', 'agent_act:fst:Park'])
        self.assertEqual(matrix, [[False, True], [True, False], [False, False]])

    def test_auc_handles_ties_and_probe_separates_signal(self):
        self.assertEqual(auc([1, 1, 1, 1], [True, False, True, False]), .5)
        self.assertEqual(auc([.1, .9, .2, .8], [False, True, False, True]), 1.0)
        torch.set_num_threads(1)
        rng = np.random.default_rng(2); y = rng.random(200) < .3
        x = rng.normal(size=(200, 5)); x[:, 0] += 3 * y
        self.assertGreater(auc(cross_validated(x, y, np.arange(200) % 5), y), .9)


if __name__ == '__main__':
    unittest.main()
