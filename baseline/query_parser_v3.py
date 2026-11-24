"""Parser v3 (2026-10-22): wider coverage, used only when parser v2 fails, so every query v2 parses keeps its exact v2
translation. Patterns were written from the released validation-split texts only (the development queries used for
fold evaluation were not inspected).

Adds: scene prefixes ("Here,", "In this scene,"), progressive and synonym forms ("is decelerating", "decelerates"),
a few lexicon gaps ("is in view", "on duty", "continues to follow", the cleft "It is because A that B"), and sequence
or causal frames mapped, like v2's, to cause-before-effect "After A, B": "A; (some time) afterward, B",
"A; consequently/therefore/as a result, B", "Because/Since A, B", "A, which causes/leads X to V", "A and then B".
"""
from __future__ import annotations

import re

from .query_parser import translate as translate_v1, Unparsed
from .query_parser_v2 import _rewrite, _third_person, translate as translate_v2

PREFIXES = re.compile(r'^(?:here|in this scene|in this video|at some point|at one point), ')
SYNONYMS = [
    (r'\b(?:is decelerating|decelerates|is slowing down)\b', 'slows down'), (r'\bis stopping\b', 'stops'),
    (r'\bis yielding\b', 'yields'), (r'\bis turning (left|right)\b', r'turns \1'), (r'\bis turning\b', 'turns'),
    (r'\bis reversing\b', 'reverses'), (r'\bis walking\b', 'walks'), (r'\bis parking\b', 'parks'),
    (r'\bis changing lanes\b', 'changes lanes'), (r'\bis changing to the (left|right) lane\b', r'changes to the \1 lane'),
    (r'\bis following another road user\b', 'follows another road user'), (r'\bis jaywalking\b', 'jaywalks'),
    (r'\bcontinues to follow\b', 'follows'), (r'\bis in view\b', 'is visible'),
    (r'\ba road worker is (?:currently )?(?:on duty|working)\b', 'a road worker is visible'),
]
FRAMES = [
    re.compile(r'^(?P<a>.+?); (?:some time |shortly )?(?:afterward|afterwards|after that|later), (?P<b>.+)$'),
    re.compile(r'^(?P<a>.+?); (?:consequently|therefore|as a result|thus|hence), (?P<b>.+)$'),
    re.compile(r'^(?:because|since) (?P<a>.+?), (?P<b>.+)$'),
    re.compile(r'^(?P<a>.+?),? which (?:causes|leads|prompts) (?P<s>the ego vehicle|an? [a-z -]+?) to (?P<v>.+)$'),
    re.compile(r'^(?P<a>.+?),? and then (?P<b>.+)$'),
]


def _v3_rewrite(t):
    t = PREFIXES.sub('', t)
    t = re.sub(r'^it is because (.+?) that (.+)$', r'because \1, \2', t)
    for pattern, repl in SYNONYMS:
        t = re.sub(pattern, repl, t)
    return t


def translate(text):
    try:
        return translate_v2(text)
    except Unparsed:
        pass
    t = _v3_rewrite(_rewrite(text))
    try:
        return translate_v2(t + '.')
    except Unparsed:
        pass
    for pattern in FRAMES:
        m = pattern.match(t)
        if not m:
            continue
        g = m.groupdict()
        b = g['b'] if g.get('b') else f"{g['s']} {_third_person(g['v'])}"
        for a, bb in ((g['a'], b), (_v3_rewrite(g['a']), _v3_rewrite(b))):
            try:
                return translate_v2(f'After {a}, {bb}.')
            except Unparsed:
                continue
    raise Unparsed(t)


__all__ = ['translate', 'Unparsed', 'translate_v1']
