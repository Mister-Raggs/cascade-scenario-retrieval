"""Time windows over a clip, source-timed frame decoding inside each window, and a FrameStore-compatible cache.

Cache keys are '<video_id>@<window>' so the existing encoder (which reads `<directory>/<key>/metadata.json`)
can encode windows unchanged.
"""
from __future__ import annotations

from collections import OrderedDict
from fractions import Fraction
import json
from pathlib import Path
import re
import subprocess

import numpy as np

from .io import sha256_file


def window_bounds(duration, windows):
    edges = np.linspace(0.0, duration, windows + 1)
    return [(float(edges[i]), float(edges[i + 1])) for i in range(windows)]


def window_times(start, end, frames):
    """Evenly spaced frame centres inside [start, end)."""
    if frames < 2 or frames % 2 or end <= start:
        raise ValueError('Need an even frame count of at least two and a positive window')
    return [start + (i + .5) * (end - start) / frames for i in range(frames)]


def probe(path):
    raw = subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                                   'stream=width,height,avg_frame_rate,duration,nb_frames', '-of', 'json', str(path)], text=True)
    info = json.loads(raw)['streams'][0]
    return {'width': int(info['width']), 'height': int(info['height']), 'fps': float(Fraction(info['avg_frame_rate'])),
            'duration': float(info['duration']), 'frames': int(info['nb_frames'])}


def decode_at(path, targets, info=None, width=640):
    """Decode one frame at each target time, keeping FFmpeg's reported source presentation time."""
    from PIL import Image
    info = info or probe(path)
    height = int(round(info['height'] / info['width'] * width / 2) * 2)
    frames, stamps = [], []
    for target in targets:
        index = min(int(round(target * info['fps'])), info['frames'] - 1)
        seek = index / info['fps']
        r = subprocess.run(['ffmpeg', '-v', 'info', '-threads', '1', '-copyts', '-ss', str(seek), '-i', str(path),
                            '-frames:v', '1', '-vf', f'scale={width}:{height},showinfo', '-vsync', '0', '-f', 'rawvideo',
                            '-pix_fmt', 'rgb24', 'pipe:1'], capture_output=True, check=True)
        times = re.findall(r'pts_time:([0-9.eE+-]+)', r.stderr.decode())
        values = np.frombuffer(r.stdout, dtype=np.uint8)
        if not times or values.size != height * width * 3:
            raise ValueError('Incomplete decoded frame or missing presentation time')
        stamp = float(times[0])
        if abs(stamp - seek) > .25:
            raise ValueError('Frame seek exceeded tolerance')
        frames.append(Image.fromarray(values.reshape(height, width, 3))); stamps.append(stamp)
    if any(b <= a for a, b in zip(stamps, stamps[1:])):
        raise ValueError('Window frame timestamps must increase')
    return frames, stamps, info


class WindowFrameStore:
    """FrameStore-compatible cache of per-window frames; `count` frames per window."""

    def __init__(self, directory, windows, count=4, video_root='data/videos'):
        self.directory = Path(directory); self.directory.mkdir(parents=True, exist_ok=True)
        self.windows, self.count, self.video_root = windows, count, Path(video_root)
        self.memory = OrderedDict()

    def keys(self, video_id):
        return [f'{video_id}@{w}' for w in range(self.windows)]

    def prepare(self, video_id):
        """Decode and cache every window of one clip; returns the window metadata list."""
        source = self.video_root / (video_id + '.mp4'); source_hash = sha256_file(source)
        info = probe(source); duration = info['duration']; out = []
        for w, (start, end) in enumerate(window_bounds(duration, self.windows)):
            target = self.directory / f'{video_id}@{w}'; metadata = target / 'metadata.json'
            if metadata.exists() and json.loads(metadata.read_text())['source_sha256'] == source_hash:
                out.append(json.loads(metadata.read_text())); continue
            frames, stamps, _ = decode_at(source, window_times(start, end, self.count), info)
            target.mkdir(parents=True, exist_ok=True)
            for i, frame in enumerate(frames):
                frame.save(target / f'{i}.png')
            value = {'video_id': video_id, 'window': w, 'window_bounds': [start, end], 'source_sha256': source_hash,
                     'timestamps': stamps, 'duration': duration, 'decoder': {'fps': info['fps']},
                     'frames_sha256': {f'{i}.png': sha256_file(target / f'{i}.png') for i in range(self.count)}}
            temporary = metadata.with_suffix('.tmp'); temporary.write_text(json.dumps(value, indent=2) + '\n'); temporary.replace(metadata)
            out.append(value)
        return out

    def get(self, key):
        from PIL import Image
        if key in self.memory:
            self.memory.move_to_end(key); return self.memory[key]
        target = self.directory / key; value = json.loads((target / 'metadata.json').read_text())
        if len(value['frames_sha256']) != self.count:
            raise ValueError('Window cache has a different frame count')
        for name, digest in value['frames_sha256'].items():
            if sha256_file(target / name) != digest:
                raise ValueError('Window frame bytes changed')
        result = ([Image.open(target / f'{i}.png').convert('RGB') for i in range(self.count)], value)
        self.memory[key] = result
        if len(self.memory) > 64:
            self.memory.popitem(last=False)
        return result


class CropFrameStore(WindowFrameStore):
    """FrameStore-compatible cache of fixed spatial crops of the canonical frames; keys '<video_id>@<crop>'."""

    def __init__(self, directory, crops, base_directory, count=4, video_root='data/videos'):
        super().__init__(directory, len(crops), count, video_root)
        self.crops, self.base_directory = crops, Path(base_directory)

    def prepare(self, video_id):
        from .h100_models import FrameStore
        frames, meta = FrameStore(self.base_directory, self.count).get(video_id); out = []
        for c, (x0, y0, x1, y1) in enumerate(self.crops):
            target = self.directory / f'{video_id}@{c}'; metadata = target / 'metadata.json'
            if metadata.exists() and json.loads(metadata.read_text())['source_sha256'] == meta['source_sha256']:
                out.append(json.loads(metadata.read_text())); continue
            target.mkdir(parents=True, exist_ok=True)
            for i, frame in enumerate(frames):
                w, h = frame.size
                frame.crop((int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h))).save(target / f'{i}.png')
            value = {**{k: meta[k] for k in ('video_id', 'source_sha256', 'timestamps', 'duration')}, 'crop': [x0, y0, x1, y1],
                     'frames_sha256': {f'{i}.png': sha256_file(target / f'{i}.png') for i in range(self.count)}}
            temporary = metadata.with_suffix('.tmp'); temporary.write_text(json.dumps(value, indent=2) + '\n'); temporary.replace(metadata)
            out.append(value)
        return out
