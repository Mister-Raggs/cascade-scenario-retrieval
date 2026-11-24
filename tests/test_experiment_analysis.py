import unittest
from baseline.experiment_analysis import interaction, paired_delta, prefix_diagnostic


class ExperimentAnalysisTests(unittest.TestCase):
    def test_prefix_ceiling_changes_at_r_cutoff(self):
        ranking = ["bad", "a", "b", "c"]
        self.assertEqual(prefix_diagnostic(ranking, {"a", "b"}, 2)["oracle_r_precision"], .5)
        self.assertEqual(prefix_diagnostic(ranking, {"a", "b"}, 3)["oracle_r_precision"], 1)
        self.assertEqual(prefix_diagnostic(ranking, {"a", "b", "c"}, 2)["oracle_r_precision"], 2 / 3)
        self.assertIsNone(prefix_diagnostic(ranking, set(), 2))

    def test_combination_must_beat_both_not_just_baseline(self):
        result = interaction([.1], [.2], [.3], [.25])
        self.assertAlmostEqual(result["interaction_delta"], -.15)
        self.assertAlmostEqual(result["combined_minus_a"], .05)
        self.assertAlmostEqual(result["combined_minus_b"], -.05)

    def test_paired_counts_and_validation(self):
        result = paired_delta([0, .5, 1], [.5, .5, .5], replicates=100)
        self.assertEqual(result["improved_queries"], 1)
        self.assertEqual(result["regressed_queries"], 1)
        self.assertEqual(result["mean_delta"], 0)
        with self.assertRaises(ValueError):
            paired_delta([1], [1, 2])


if __name__ == "__main__":
    unittest.main()
