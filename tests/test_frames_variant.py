"""Frame-count generalization keeps the 4-frame path and rejects mismatched caches; synthetic only."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from baseline.h100_models import FrameStore


class FrameCountTests(unittest.TestCase):
    def test_odd_or_tiny_frame_counts_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            for count in (1, 3, 5):
                with self.assertRaises(ValueError):
                    FrameStore(directory, count)
            self.assertEqual(FrameStore(directory).count, 4)

    def test_cache_written_for_another_frame_count_is_not_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / 'clip').mkdir()
            (root / 'clip' / 'metadata.json').write_text(json.dumps(
                {'source_sha256': 'x', 'frames_sha256': {f'{i}.png': 'y' for i in range(4)}}))
            with patch('baseline.h100_models.sha256_file', return_value='x'):
                with self.assertRaisesRegex(ValueError, 'different frame count'):
                    FrameStore(root, 8).get('clip')


if __name__ == '__main__':
    unittest.main()
