"""Time-localized training labels read directly from CASCADE annotation JSON (no devkit needed).

Labels use raw annotation vocabulary values; they are training targets only and are never shipped
in an inference container.
"""
from __future__ import annotations

import json
from pathlib import Path


def parse_time(value, duration=None):
    """Annotation timestamps are 'm:s.s' or decimal-second strings; empty means unset."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if ':' in text:
        minutes, seconds = text.split(':', 1)
        seconds = int(minutes) * 60 + float(seconds)
    else:
        seconds = float(text)
    return seconds if duration is None else min(max(seconds, 0.0), duration)


def _interval(item, start_key, end_key, duration):
    start, end = parse_time(item.get(start_key), duration), parse_time(item.get(end_key), duration)
    start = 0.0 if start is None else start
    end = duration if end is None else end
    return (start, end) if end > start else None


def events(bundle):
    """Return (clip_id, duration, [(label, start, end), ...]) for one annotation bundle dict."""
    video, a = bundle['video'], bundle.get('annotation') or {}
    duration = float(video.get('duration_s') or 20.0)
    out = []
    def add(label, span):
        if span:
            out.append((label, round(span[0], 3), round(span[1], 3)))
    for act in (a.get('ego_vehicle') or {}).get('actions') or []:
        add('ego_act:' + str(act.get('type')), _interval(act, 'start_timestamp', 'end_timestamp', duration))
    for agent in a.get('agents') or []:
        kind = str(agent.get('type'))
        add('agent:' + kind, _interval(agent, 'visibility_start_timestamp', 'visibility_end_timestamp', duration))
        for act in agent.get('actions') or []:
            action = str(act.get('action_type') or act.get('type'))
            span = _interval(act, 'start_timestamp', 'end_timestamp', duration)
            add('agent_act:' + action, span)
            add(f'agent_act:{kind}|{action}', span)
    for obj in a.get('traffic_objects') or []:
        add('obj:' + str(obj.get('type')), _interval(obj, 'visibility_start_timestamp', 'visibility_end_timestamp', duration))
    for light in a.get('traffic_lights') or []:
        for head in light.get('signal_heads') or []:
            for state in head.get('state_sequence') or []:
                span = _interval(state, 'start_timestamp', 'end_timestamp', duration)
                add('light:' + str(state.get('color')), span)
                if state.get('shape'):
                    add(f"light_shape:{state.get('color')}|{state.get('shape')}", span)
    return video['clip_id'], duration, out


def load_events(directory):
    result = {}
    for path in sorted(Path(directory).glob('*.json')):
        clip, duration, items = events(json.loads(path.read_text()))
        result[clip] = {'duration': duration, 'events': items}
    return result


def window_labels(items, windows, labels):
    """Binary matrix [window, label]: a label is active if any of its intervals overlaps the window."""
    index = {label: i for i, label in enumerate(labels)}
    matrix = [[False] * len(labels) for _ in windows]
    for label, start, end in items:
        if label not in index:
            continue
        for w, (ws, we) in enumerate(windows):
            if start < we and end > ws:
                matrix[w][index[label]] = True
    return matrix
