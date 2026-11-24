import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from baseline.devkit import load_module
from scripts.robustness_retrieval import partitions, read_rankings, score_group, summarize


class RobustnessTests(unittest.TestCase):
    def test_video_restriction_changes_r_and_skips_no_positive_queries(self):
        # Full q0: R=2, first two are x,a => RP=.5. Subset a,b: R=1 => RP=1.
        ranking = np.array([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=np.int32)
        labels = [{1, 3}, {0}]
        metrics = load_module("metrics")
        full = score_group(ranking, labels, range(4), metrics)
        subset = score_group(ranking, labels, [1, 2], metrics)
        self.assertEqual(full[0][0], .5)
        self.assertEqual(subset[0][0], 1)
        self.assertTrue(np.isnan(subset[0][1]))
        report = summarize({"B": subset[:2], "BV": subset[:2]}, subset[2])
        self.assertEqual(report["num_queries_without_relevance"], 1)
        self.assertEqual(report["num_scored_queries"], 1)
        self.assertEqual(report["recipes"]["B"]["map"], 1)
        self.assertEqual(report["comparisons"]["BV_minus_B"]["mean_r_precision_delta"], 0)

    def test_partition_is_exhaustive_disjoint_and_repeatable(self):
        groups = partitions(2066, 2, 17)
        self.assertEqual([len(x) for x in groups], [1033, 1033])
        self.assertEqual(set(groups[0]) & set(groups[1]), set())
        self.assertEqual(set(groups[0]) | set(groups[1]), set(range(2066)))
        self.assertTrue(np.array_equal(groups[0], partitions(2066, 2, 17)[0]))
        self.assertFalse(np.array_equal(groups[0], partitions(2066, 2, 29)[0]))

    def test_missing_duplicate_or_partial_predictions_cannot_bias_comparison(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "predictions.jsonl"
            valid = {"query_id": "q", "video_ids": ["b", "a"]}
            path.write_text(json.dumps(valid) + "\n")
            self.assertEqual(read_rankings(path, ["q"], ["a", "b"]).tolist(), [[1, 0]])
            for rows, query_ids in (([valid, valid], ["q"]), ([valid], ["q", "missing"]),
                    ([{"query_id": "q", "video_ids": ["a", "a"]}], ["q"]),
                    ([{"query_id": "q", "video_ids": ["a"]}], ["q"])):
                path.write_text("\n".join(json.dumps(row) for row in rows))
                with self.assertRaises(ValueError):
                    read_rankings(path, query_ids, ["a", "b"])

    def test_paired_coverage_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            summarize({"B": (np.array([.5, np.nan]), np.array([.5, np.nan])),
                       "BV": (np.array([.5, .5]), np.array([.5, .5]))}, np.array([1, 0]))


if __name__ == "__main__":
    unittest.main()
