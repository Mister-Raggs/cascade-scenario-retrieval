"""Property atoms (2026-10-22): the qualifiers the structure atoms drop (baseline/query_struct.py keys keep entity, type
and action only).

Query side (from the parser's DSL): an agent with prop.type (signal, slow, double_parked) gives
'agent|prop=<p>|type=<t>' (type optional); a light with state = flashing, affects_ego = true or ego_in_on_yellow = true
gives 'light|color=<c>|state=flashing', 'light|affects_ego=true|color=<c>' or 'light|color=<c>|ego_in_on_yellow=true'.
Label side (raw CASCADE annotation JSON): agents[].properties[].property_type and signal-head states (type Flashing,
yellow_on_ego_path, ego_in_intersection_on_yellow). Labels are training targets only and never shipped.
"""
from __future__ import annotations

import json
from pathlib import Path
import re

from .annotation_events import _interval
from .query_struct import _expand, _values, atom_key

_GROUP = re.compile(r'\b(agent|light)\(')


def _norm(text):
    return str(text).strip().lower().replace(' ', '_')


def labels(bundle):
    """Set of property labels for one annotation bundle."""
    video, a = bundle['video'], bundle.get('annotation') or {}
    duration = float(video.get('duration_s') or 20.0); out = set()
    for agent in a.get('agents') or []:
        for prop in agent.get('properties') or []:
            if _interval(prop, 'start_timestamp', 'end_timestamp', duration):
                out.add(f"agent_prop:{agent.get('type')}|{_norm(prop.get('property_type'))}")
    for light in a.get('traffic_lights') or []:
        for head in light.get('signal_heads') or []:
            for state in head.get('state_sequence') or []:
                if not _interval(state, 'start_timestamp', 'end_timestamp', duration):
                    continue
                color = state.get('color')
                if _norm(state.get('type')) == 'flashing':
                    out.add(f'light_state:{color}|flashing')
                if str(state.get('yellow_on_ego_path')) == 'True':
                    out.add(f'light_flag:{color}|affects_ego')
                if str(state.get('ego_in_intersection_on_yellow')) == 'True':
                    out.add(f'light_flag:{color}|ego_in_on_yellow')
    return out


def load_labels(directory):
    result = {}
    for path in sorted(Path(directory).glob('*.json')):
        bundle = json.loads(path.read_text()); result[bundle['video']['clip_id']] = labels(bundle)
    return result


def _groups(dsl):
    """(entity, inner text) for every agent(...) / light(...) group, respecting nested parentheses."""
    for m in _GROUP.finditer(dsl):
        depth, start = 1, m.end()
        for i in range(start, len(dsl)):
            depth += dsl[i] == '('; depth -= dsl[i] == ')'
            if depth == 0:
                yield m.group(1), dsl[start:i]; break


def query_atoms(dsl):
    """Sorted property atom keys of one parsed query (its DSL)."""
    out = set()
    for entity, inner in _groups(dsl):
        fields = {}
        for part in re.split(r', (?![^(]*\))', inner):
            name, _, value = part.partition(' = ') if ' = ' in part else part.partition(' in ')
            fields[name.strip()] = _values(value)
        if entity == 'agent' and 'prop.type' in fields:
            keep = {'prop': fields['prop.type']}
            if 'type' in fields:
                keep['type'] = fields['type']
            out.add(atom_key('agent', keep))
        if entity == 'light':
            base = {'color': fields['color']} if 'color' in fields else {}
            if fields.get('state') == ('flashing',):
                out.add(atom_key('light', {**base, 'state': ('flashing',)}))
            for flag in ('affects_ego', 'ego_in_on_yellow'):
                if fields.get(flag) == ('true',):
                    out.add(atom_key('light', {**base, flag: ('true',)}))
    return sorted(out)


def present(key, labels):
    entity, *parts = key.split('|')
    fields = {p.split('=')[0]: tuple(p.split('=')[1].split(',')) for p in parts}
    if entity == 'agent':
        types = _expand('agent_type', fields['type']) if 'type' in fields else None
        return any(l.startswith('agent_prop:') and (types is None or l[11:].split('|')[0] in types) and l.split('|')[1] in fields['prop']
                   for l in labels)
    colors = {c for v in fields.get('color', ()) for c in _expand('light_color', (v,))} if 'color' in fields else None
    for tag, prefix in (('state', 'light_state:'), ('affects_ego', 'light_flag:'), ('ego_in_on_yellow', 'light_flag:')):
        if tag in fields:
            want = 'flashing' if tag == 'state' else tag
            return any(l.startswith(prefix) and l.split('|')[1] == want and (colors is None or l[len(prefix):].split('|')[0] in colors)
                       for l in labels)
    raise ValueError(key)


_AGENT_PROP = {'signal': 'signaling (turn signal, hazard lights or a hand signal)', 'slow': 'moving slowly', 'double_parked': 'double-parked'}
_WHO = {'car': 'a car', 'truck': 'a truck or heavy-duty vehicle', 'ped': 'a pedestrian', 'bicycle': 'a cyclist', 'bus': 'a bus',
        'motorcycle': 'a motorcycle', 'vehicle': 'another vehicle', 'emergency': 'an emergency vehicle'}


def question(key):
    """Natural-language yes/no question for a property atom (zero-shot VLM probing)."""
    entity, *parts = key.split('|')
    fields = {p.split('=')[0]: p.split('=')[1].split(',') for p in parts}
    if entity == 'agent':
        who = ' or '.join(_WHO.get(t, 'a ' + t.replace('_', ' ')) for t in fields['type']) if 'type' in fields else 'any road user other than the camera car'
        return f"Is {who} {' or '.join(_AGENT_PROP[p] for p in fields['prop'])} at any point in this video?"
    color = fields.get('color', [''])[0]
    if 'state' in fields:
        return f'Is a flashing {color} traffic signal shown at any point in this video?'
    if 'affects_ego' in fields:
        return f'Is a {color} traffic signal shown that applies to the camera car\'s own path at any point in this video?'
    return f'Is the camera car already inside the intersection when the traffic signal is {color}?'
