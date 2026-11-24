"""Parser v2: wider paraphrase and causal-frame coverage, used only when the frozen v1 parser fails.

Every query v1 parses keeps its exact v1 translation, so recipes built on v1 are unchanged.
Causal frames ("B since A", "A, so B", "A, leading X to V") are mapped to cause-before-effect.
"""
from __future__ import annotations

import re

from .query_parser import translate as translate_v1, Unparsed

PARAPHRASES = [
    (r'\bcan be seen\b', 'is visible'), (r'\bproceeds on foot\b', 'walks'), (r'\bremains standing\b', 'stands'),
    (r'\bmakes a turn\b', 'turns'), (r'\bmakes a (left|right) turn\b', r'turns \1'),
    (r'\bmakes a lane change\b', 'changes lanes'), (r'\bmakes a (left|right) lane change\b', r'changes to the \1 lane'),
    (r'\bmoves along (its|their) lane\b', r'drives in \1 lane'), (r'\bis driving in (its|their) lane\b', r'drives in \1 lane'),
    (r'\bhas stopped but is not parked\b', 'stops or is stopped but is not parked'),
    (r'\bcomes to a stop or is stopped but is not parked\b', 'stops or is stopped but is not parked'),
    (r'\bis not moving\b', 'stops or is stopped but is not parked'), (r'\btravels slowly\b', 'is moving slowly'),
    (r'\bhas double-parked\b', 'is double-parked'), (r'\bhas a signal on\b', 'is signaling'),
    (r'\bperforms a parking maneuver\b', 'parks'), (r'\bmoves backward\b', 'reverses'),
    (r'\bpasses another road user\b', 'overtakes'), (r'\bis moving slightly sideways\b', 'makes a small lateral move'),
    (r'\bmay (turn|change|proceed|stop|yield)\b', lambda m: _third_person(m.group(1))),
    (r'\b(?:that|the) same (road user|person|car|vehicle|pedestrian|cyclist)\b', r'a \1'),
    (r'\bthe movement of a boom gate is visible\b', 'a boom gate is visible'), (r'\ba boom gate is moving\b', 'a boom gate is visible'),
    (r'\ba (red|yellow|green) traffic light is visible\b', r'a traffic light shows \1'),
    (r'\ba flashing (red|yellow|green) traffic light is visible\b', r'a flashing \1 traffic light is active'),
    (r'\ba (red|yellow|green) (left|right|upward)-arrow signal is visible\b', r'a \1 \2-arrow signal is active'),
    (r'^at some point, ', ''),
]
CAUSAL = [
    re.compile(r'^(?P<b>.+?) (?:since|because) (?P<a>.+)$'),
    re.compile(r'^(?P<a>.+?), so (?P<b>.+)$'),
    re.compile(r'^(?P<a>.+?); in response, (?P<b>.+)$'),
    re.compile(r'^(?P<a>.+?), (?:leading|prompting|causing) (?P<s>the ego vehicle|an? [a-z -]+?) to (?P<v>.+)$'),
]


def _third_person(verb):
    first, _, rest = verb.partition(' ')
    first = first + ('es' if first.endswith(('s', 'sh', 'ch', 'x')) else 's')
    return (first + ' ' + rest).strip()


def _rewrite(text):
    t = text.strip().rstrip('.').strip()
    t = t[0].lower() + t[1:] if t else t
    for pattern, repl in PARAPHRASES:
        t = re.sub(pattern, repl, t)
    return t


def translate(text):
    try:
        return translate_v1(text)
    except Unparsed:
        pass
    t = _rewrite(text)
    try:
        return translate_v1(t + '.')
    except Unparsed:
        pass
    for pattern in CAUSAL:
        m = pattern.match(t)
        if not m:
            continue
        g = m.groupdict()
        b = g['b'] if 'b' in g and g.get('b') else f"{g['s']} {_third_person(g['v'])}"
        try:
            return translate_v1(f"After {g['a']}, {b}.")
        except Unparsed:
            continue
    raise Unparsed(t)
