"""Mirror semantics and verifier pair sampling; synthetic inputs only."""
import unittest

import numpy as np

from baseline.mirror import mirror_atom, mirror_text
from baseline.query_pairs import clip_query_pools, sample_queries, verifier_question


class MirrorTests(unittest.TestCase):
    def test_atoms_swap_left_and_right_only(self):
        self.assertEqual(mirror_atom('agent|action=turn_left|type=car'), 'agent|action=turn_right|type=car')
        self.assertEqual(mirror_atom('light|color=green|shape=arrow_right'), 'light|color=green|shape=arrow_left')
        self.assertEqual(mirror_atom('ego|action=stop'), 'ego|action=stop')
        self.assertEqual(mirror_atom(mirror_atom('ego|action=change_lane_left')), 'ego|action=change_lane_left')

    def test_text_swaps_whole_words_and_keeps_case(self):
        self.assertEqual(mirror_text('Left turn, then the ego vehicle changes to the right lane.'),
                         'Right turn, then the ego vehicle changes to the left lane.')
        self.assertEqual(mirror_text('A bright light.'), 'A bright light.')


class PairTests(unittest.TestCase):
    def setUp(self):
        self.vids = ['a', 'b', 'c', 'd']
        self.truth = {'x': np.array([1, 1, 1, 0], bool), 'y': np.array([1, 1, 0, 0], bool)}
        self.qrels = {'q1': {'relevant_video_ids': ['a']}, 'q2': {'relevant_video_ids': ['c', 'd']}}
        self.parsed = {'q1': ['x', 'y'], 'q2': ['x']}

    def test_near_misses_have_all_atoms_but_are_not_relevant_and_stay_in_fit(self):
        pools = clip_query_pools(['q1', 'q2'], self.qrels, self.parsed, self.truth, self.vids, np.array([0, 1, 2]))
        self.assertEqual(pools[0], (['q1'], ['q2']))
        self.assertEqual(pools[1], ([], ['q1', 'q2']))
        self.assertEqual(pools[2], (['q2'], []))
        self.assertNotIn(3, pools)

    def test_samples_never_label_a_relevant_clip_negative(self):
        pools = clip_query_pools(['q1', 'q2'], self.qrels, self.parsed, self.truth, self.vids, np.array([0, 1, 2]))
        relevant_of = {'q1': {0}, 'q2': {2, 3}}
        for seed in range(20):
            for i in (0, 1, 2):
                for q, label in sample_queries(pools, i, ['q1', 'q2'], relevant_of, np.random.default_rng(seed), 2):
                    self.assertEqual(label, int(i in relevant_of[q]))
        self.assertTrue(verifier_question('A car stops.').endswith('"A car stops."'))


if __name__ == '__main__':
    unittest.main()
