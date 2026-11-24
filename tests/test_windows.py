"""Window geometry and windowed-probe evaluation wiring; synthetic inputs only."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from baseline.windows import window_bounds, window_times
from scripts import window_run


class WindowTests(unittest.TestCase):
    def test_bounds_partition_clip_and_frames_stay_inside(self):
        bounds = window_bounds(20.0, 5)
        self.assertEqual(bounds[0][0], 0.0); self.assertEqual(bounds[-1][1], 20.0)
        self.assertTrue(all(a[1] == b[0] for a, b in zip(bounds, bounds[1:])))
        times = window_times(4.0, 8.0, 4)
        self.assertEqual(times, [4.5, 5.5, 6.5, 7.5])
        with self.assertRaises(ValueError):
            window_times(0, 4, 3)

    def test_localized_signal_favors_windowed_probe(self):
        torch.set_num_threads(1)
        rng = np.random.default_rng(0); n, k, d = 60, 5, 6
        vids = [f'v{i}' for i in range(n)]
        active = rng.random(n) < .4; where = rng.integers(0, k, n)
        windowed = rng.normal(size=(n, k, d)).astype(np.float32)
        windowed[np.arange(n)[active], where[active], 0] += 4
        pooled = windowed.mean(1)
        events = {v: {'duration': 20.0, 'events': ([('ego_act:oxd:Stop', 4.0 * where[i] + 1, 4.0 * where[i] + 2)] if active[i] else [])
                                                    + [('agent:oxd:Car', 0.0, 20.0)]} for i, v in enumerate(vids)}
        bounds = {v: [{'bounds': [4.0 * w, 4.0 * (w + 1)]} for w in range(k)] for v in vids}
        plan = {'windows': k, 'feature_directory': 'f', 'annotation_directory': 'a', 'minimum_positive_clips': 5,
                'seed': 0, 'folds': 3, 'minimum_auc_gain': .03}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); out = root / 'evaluate'; (out / 'results').mkdir(parents=True)
            (root / 'window-manifest.json').write_text(json.dumps({'keys': [f'{v}@{w}' for v in vids for w in range(k)]}))
            (root / 'windows.json').write_text(json.dumps(bounds))
            vectors = {'window-vectors.npz': windowed.reshape(n * k, d), 'aggregate.npz': pooled}
            with patch.object(window_run, 'development_videos', return_value=vids), \
                 patch.object(window_run, 'verified_result', side_effect=lambda r, j, name: root / name), \
                 patch.object(window_run, 'load_vectors', side_effect=lambda p: vectors[Path(p).name]), \
                 patch.object(window_run.Path, 'read_text', new=lambda self, *a, **kw: (
                     json.dumps({'video_ids': vids}) if self.name == 'feature_manifest.json' else Path.open(self).read())), \
                 patch.object(window_run, 'load_events', return_value=events), \
                 patch.object(window_run, 'finish') as finished:
                window_run.evaluate_windows(out, plan)
        report = finished.call_args.args[2]
        stop = next(r for r in report['labels'] if r['label'] == 'ego_act:oxd:Stop')
        self.assertGreater(stop['clip_auc_windowed'], stop['clip_auc_pooled'])
        self.assertIn('agent', report['by_group'])


if __name__ == '__main__':
    unittest.main()
