"""Structure trees from the parser's generated DSL, and clip-level atom matching on annotation events.

Trees: ('atom', key) | ('and', [nodes]) | ('while', a, b) | ('before', a, b). Atom keys are canonical
strings such as 'ego|action=turn_left' or 'agent|type=car|action=turn_right', so probes can be shared.
Properties, amounts and light states are dropped from keys (not yet perceived); they only narrow matches.
"""
from __future__ import annotations

import json
from pathlib import Path
import re

ALIASES = json.loads((Path(__file__).parent / 'dsl_aliases.json').read_text())
_VALUE = r'(?:"[^"]*"|\([^)]*\)|[a-z_]+)'


def _values(text):
    text = text.strip()
    if text.startswith('('):
        return tuple(v.strip().strip('"') for v in text[1:-1].split(','))
    return (text.strip('"'),)


def _split_top(text, op):
    depth, parts, start = 0, [], 0
    for i, ch in enumerate(text):
        depth += ch == '('; depth -= ch == ')'
        if depth == 0 and text.startswith(f' {op} ', i):
            parts.append(text[start:i]); start = i + len(op) + 2
    return parts + [text[start:]] if parts else None


def _strip(text):
    text = text.strip()
    while text.startswith('(') and text.endswith(')'):
        depth = 0
        for i, ch in enumerate(text):
            depth += ch == '('; depth -= ch == ')'
            if depth == 0 and i < len(text) - 1:
                return text
        text = text[1:-1].strip()
    return text


def atom_key(entity, fields):
    return '|'.join([entity] + [f'{k}={",".join(sorted(v))}' for k, v in sorted(fields.items())])


def _atom(text):
    m = re.fullmatch(r'ego\.action (?:=|in) (' + _VALUE + ')', text)
    if m:
        return atom_key('ego', {'action': _values(m.group(1))})
    m = re.fullmatch(r'obj\.type (?:=|in) (' + _VALUE + ')', text)
    if m:
        return atom_key('obj', {'type': _values(m.group(1))})
    m = re.fullmatch(r'light\.color = ([a-z]+)', text)
    if m:
        return atom_key('light', {'color': (m.group(1),)})
    m = re.fullmatch(r'light\((.*)\)', text)
    if m:
        fields = dict(p.split(' = ') for p in m.group(1).split(', '))
        keep = {k: (fields[k],) for k in ('color', 'shape') if k in fields}
        return atom_key('light', keep)
    if text == 'agent.type = road_user':
        return atom_key('agent', {})
    m = re.fullmatch(r'agent\((.*)\)', text)
    if m:
        fields = {}
        for part in re.split(r', (?![^(]*\))', m.group(1)):
            name, _, value = part.partition(' = ') if ' = ' in part else part.partition(' in ')
            if name in ('type', 'action.type'):
                fields[name.split('.')[0]] = _values(value)
        return atom_key('agent', fields)
    raise ValueError('Unrecognized DSL atom: ' + text)


def tree(dsl):
    text = _strip(dsl)
    for op in ('before', 'then', 'while'):
        parts = _split_top(text, op)
        if parts and len(parts) == 2:
            return ('before' if op in ('before', 'then') else 'while', tree(parts[0]), tree(parts[1]))
    parts = _split_top(text, 'and')
    if parts:
        return ('and', [tree(p) for p in parts])
    return ('atom', _atom(text))


def atoms(node):
    if node[0] == 'atom':
        return [node[1]]
    children = node[1] if node[0] == 'and' else [node[1], node[2]]
    return [a for child in children for a in atoms(child)]


def _expand(table, values):
    out = set()
    for value in values:
        out.update(ALIASES[table].get(value, [value]))
    return out


def _action_ok(raw, aliases):
    for alias in aliases:
        if alias in ('jaywalk', 'jaywalk_flag'):
            if '(jaywalk' in raw:
                return True
            continue
        if any(raw == v or raw.startswith(v + ' (') for v in ALIASES['action_type'].get(alias, [alias])):
            return True
    return False


def atom_present(key, labels):
    """Clip-level truth of an atom given a clip's set of event labels (timing ignored)."""
    entity, *parts = key.split('|')
    fields = {p.split('=')[0]: tuple(p.split('=')[1].split(',')) for p in parts}
    if entity == 'ego':
        return any(l.startswith('ego_act:') and _action_ok(l[8:], fields['action']) for l in labels)
    if entity == 'obj':
        types = _expand('obj_type', fields['type'])
        return any(l.startswith('obj:') and l[4:] in types for l in labels)
    if entity == 'light':
        colors = {c for v in fields.get('color', ()) for c in ALIASES['light_color'].get(v, [v])}
        if 'shape' in fields:
            shapes = {s for v in fields['shape'] for s in ALIASES['light_shape'].get(v, [v])}
            return any(l.startswith('light_shape:') and l[12:].split('|')[0] in colors and l[12:].split('|')[1] in shapes
                       for l in labels)
        return any(l.startswith('light:') and l[6:] in colors for l in labels)
    if entity == 'agent':
        types = _expand('agent_type', fields['type']) if 'type' in fields else None
        if 'action' not in fields:
            return any(l.startswith('agent:') and (types is None or l[6:] in types) for l in labels)
        for l in labels:
            if l.startswith('agent_act:') and '|' in l:
                kind, action = l[10:].split('|', 1)
                if (types is None or kind in types) and _action_ok(action, fields['action']):
                    return True
        return False
    raise ValueError(key)
