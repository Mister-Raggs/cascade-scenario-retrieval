"""Synthetic vector tests; no challenge scores or downloads."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from baseline.dense import fuse, normalize, rank_scores
from scripts.feature_retrieval import load_vectors, rankings, save_vectors


class DenseTests(unittest.TestCase):
    def test_ties_and_full_ranking_include_negative_similarities(self):
        self.assertEqual(rank_scores(["b", "c", "a"], np.array([1, -1, 1])), ["a", "b", "c"])

    def test_fusion_preserves_corpus_and_missing_lexical_results(self):
        ranking = fuse(["a", "b", "c"], ["c", "b", "a"], ["a"], .5)
        self.assertEqual(ranking[0], "a")
        self.assertEqual(set(ranking), {"a", "b", "c"})
        self.assertEqual(fuse(["a", "b", "c"], ["c", "b", "a"], [], 1), ["c", "b", "a"])

    def test_frame_max_captures_a_matching_frame_and_rejects_nonfinite_cache(self):
        frames = normalize(np.array([[[1, 0], [0, 1]], [[-1, 0], [0, -1]]]))
        self.assertEqual(list(rankings(["a", "b"], frames, np.array([[0, 1]]), "max")), [["a", "b"]])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cache.npz"
            save_vectors(path, frames)
            np.testing.assert_allclose(load_vectors(path), frames)
            save_vectors(path, np.array([[float("nan")]]))
            with self.assertRaises(ValueError):
                load_vectors(path)


if __name__ == "__main__":
    unittest.main()
