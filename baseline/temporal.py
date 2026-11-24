"""Timestamp metadata and deterministic sampling for video embedding experiments."""
from __future__ import annotations
import numpy as np


def timestamp_metadata(timestamps, duration):
    stamps=np.asarray(timestamps,dtype=float)
    if stamps.ndim!=1 or len(stamps)<2 or not np.isfinite(stamps).all():
        raise ValueError('At least two finite frame timestamps required')
    if np.any(np.diff(stamps)<=0) or stamps[0]<0 or stamps[-1]>=duration:
        raise ValueError('Frame timestamps must increase inside the clip')
    # The processor derives seconds from indices/fps, then averages temporal pairs.
    return {'fps':1000.,'frames_indices':(stamps*1000).tolist(),
            'total_num_frames':int(np.ceil(duration*1000))}


def sample_indices(duration,fps,total_frames,count):
    if min(duration,fps,total_frames,count)<=0 or count%2:
        raise ValueError('Positive source metadata and an even frame count required')
    indices=np.rint((np.arange(count)+.5)*duration/count*fps).astype(int)
    indices=np.clip(indices,0,total_frames-1)
    if len(set(indices))!=count:raise ValueError('Source has too few distinct frames')
    return indices.tolist()
