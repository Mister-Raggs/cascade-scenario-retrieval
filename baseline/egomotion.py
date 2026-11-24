"""Dependency-light ego-motion features from video: FFT phase correlation over a frame grid.

Each consecutive frame pair gives a translation per grid cell. Derived signals approximate yaw
(distant-region horizontal shift), forward motion (radial expansion) and lateral motion.
"""
from __future__ import annotations

import json
import subprocess

import numpy as np


def decode_gray(path, fps=4.0, width=256):
    info = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                                               'stream=width,height', '-of', 'json', str(path)], text=True))['streams'][0]
    height = int(round(info['height'] / info['width'] * width / 2) * 2)
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', str(path), '-vf', f'fps={fps},scale={width}:{height}',
                          '-f', 'rawvideo', '-pix_fmt', 'gray', 'pipe:1'], capture_output=True, check=True).stdout
    frames = np.frombuffer(raw, dtype=np.uint8)
    if frames.size % (width * height):
        raise ValueError('Incomplete decoded frame stream')
    frames = frames.reshape(-1, height, width).astype(np.float32)
    if len(frames) < 2:
        raise ValueError('Need at least two frames for motion')
    return frames


def phase_shift(a, b):
    """Subpixel translation (dy, dx) that maps patch a onto patch b, plus peak confidence."""
    window = np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1]))
    fa, fb = np.fft.fft2((a - a.mean()) * window), np.fft.fft2((b - b.mean()) * window)
    cross = fb * np.conj(fa)
    cross /= np.abs(cross) + 1e-9
    corr = np.fft.ifft2(cross).real
    peak = np.unravel_index(np.argmax(corr), corr.shape)
    shift = []
    for axis, p in enumerate(peak):
        n = corr.shape[axis]
        prev = corr[(p - 1) % n, peak[1]] if axis == 0 else corr[peak[0], (p - 1) % n]
        nxt = corr[(p + 1) % n, peak[1]] if axis == 0 else corr[peak[0], (p + 1) % n]
        denom = prev - 2 * corr[peak] + nxt
        offset = 0.5 * (prev - nxt) / denom if abs(denom) > 1e-12 else 0.0
        value = p + offset
        shift.append(value - n if value > n / 2 else value)
    return shift[0], shift[1], float(corr[peak])


def grid_flow(frames, rows=3, cols=4):
    """[pairs, rows, cols, 3] of (dy, dx, confidence)."""
    t, h, w = frames.shape
    hs, ws = h // rows, w // cols
    out = np.zeros((t - 1, rows, cols, 3), dtype=np.float32)
    for i in range(t - 1):
        for r in range(rows):
            for c in range(cols):
                a = frames[i, r * hs:(r + 1) * hs, c * ws:(c + 1) * ws]
                b = frames[i + 1, r * hs:(r + 1) * hs, c * ws:(c + 1) * ws]
                out[i, r, c] = phase_shift(a, b)
    return out


def signals(flow):
    """Per-pair derived signals [pairs, 6]: yaw, expansion, lateral, vertical, magnitude, confidence."""
    dy, dx, conf = flow[..., 0], flow[..., 1], flow[..., 2]
    cols = dx.shape[2]
    yaw = dx[:, 0, :].mean(1)
    expansion = (dx[:, :, cols // 2:].mean((1, 2)) - dx[:, :, :cols // 2].mean((1, 2))) + (dy[:, -1, :].mean(1) - dy[:, 0, :].mean(1))
    lateral = dx[:, -1, :].mean(1) - yaw
    vertical = dy.mean((1, 2))
    magnitude = np.sqrt(dx ** 2 + dy ** 2).mean((1, 2))
    return np.stack([yaw, expansion, lateral, vertical, magnitude, conf.mean((1, 2))], axis=1)


def clip_vector(series, bins=5):
    """Fixed-length summary: global statistics plus coarse temporal profile of each signal."""
    s = np.asarray(series, dtype=np.float32)
    stats = [s.mean(0), s.std(0), s.min(0), s.max(0), np.quantile(s, .1, axis=0), np.quantile(s, .9, axis=0),
             np.abs(np.diff(s, axis=0)).mean(0) if len(s) > 1 else np.zeros(s.shape[1], np.float32)]
    profile = [chunk.mean(0) for chunk in np.array_split(s, bins)]
    return np.concatenate(stats + profile).astype(np.float32)
