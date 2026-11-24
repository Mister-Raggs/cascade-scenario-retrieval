import unittest
from baseline.temporal import timestamp_metadata,sample_indices


class TemporalTests(unittest.TestCase):
    def test_irregular_source_times_survive_metadata(self):
        times=[.4,1.9,4.1,8.7]
        m=timestamp_metadata(times,10)
        recovered=[i/m['fps'] for i in m['frames_indices']]
        self.assertEqual(recovered,times)
        self.assertAlmostEqual((recovered[2]+recovered[3])/2,6.4)

    def test_invalid_order_or_range_fails(self):
        for times in [[1,1],[2,1],[-1,2],[1,10]]:
            with self.assertRaises(ValueError):timestamp_metadata(times,10)

    def test_sampling_covers_clip_midpoints(self):
        self.assertEqual(sample_indices(20,30,600,4),[75,225,375,525])
        with self.assertRaises(ValueError):sample_indices(1,2,2,8)
