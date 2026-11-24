"""Score-normalization behavior and metric parity; synthetic inputs only."""
import unittest

import numpy as np

from baseline.hubness import normalize, ranking_metrics
from baseline.post_submission import evaluate
from baseline.projection import SharedProjection
from tests.fixtures import fixture


class HubnessTests(unittest.TestCase):
    def test_hub_video_is_demoted_but_identity_is_unchanged(self):
        # Video 0 is a hub: moderately similar to every query. Video 1 is a strong match for query 1 only.
        scores = np.asarray([[.60, .50, .10], [.60, .58, .10], [.60, .10, .55]])
        self.assertEqual(np.argmax(scores[1]), 0)
        for method in [{'kind': 'dual-softmax', 'temperature': .05}, {'kind': 'inverted-softmax', 'temperature': .05},
                       {'kind': 'csls', 'k': 2}]:
            self.assertEqual(np.argmax(normalize(scores, method)[1]), 1, method)
        np.testing.assert_array_equal(normalize(scores, {'kind': 'none'}), scores.astype(np.float32))
        with self.assertRaises(ValueError):
            normalize(scores, {'kind': 'unknown'})

    def test_metrics_match_existing_evaluation_including_ties(self):
        parent, qrels, qids, vids, q, v = fixture()
        state = SharedProjection(4, 2).state_dict()
        reference = evaluate(q, v, qids, vids, qrels, state, 1)
        score = ranking_metrics(q @ v.T, qids, vids, qrels)
        self.assertEqual(score['r_precision'], reference['r_precision'])
        self.assertEqual(score['map'], reference['map'])


if __name__ == '__main__':
    unittest.main()
