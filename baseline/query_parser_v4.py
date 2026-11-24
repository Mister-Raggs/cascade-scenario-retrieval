"""Parser v4 (2026-10-22): parser v3 plus time-limited sequences, used only when v3 fails, so every query v3 parses
keeps its exact v3 translation. Patterns were written from the released validation-split texts only.

Time limits ("within 10 seconds", "no more than 10 seconds after", "with no more than 10 seconds between them") are
dropped and the sequence is kept as cause-before-effect "After A, B"; "finishes changing lanes" and "starts to change"
become the plain actions. The limit itself is not modelled (atoms carry no timing).
"""
from __future__ import annotations

import re

from .query_parser import Unparsed
from .query_parser_v3 import translate as translate_v3

TIMED = re.compile(r'\b(?:within|at most|no more than) \d+ seconds?\b')
LIMIT = r'(?:within|at most|no more than) \d+ seconds?'
FRAMES = [
    re.compile(rf'^{LIMIT} after (?P<a>.+?), (?P<b>.+)$'),
    re.compile(rf'^no more than \d+ seconds? pass from when (?P<a>.+?) until (?P<b>.+)$'),
    re.compile(rf'^(?:after|once) (?P<a>.+?), (?P<b>.+?) {LIMIT}$'),
    re.compile(rf'^(?P<b>.+?) (?:{LIMIT} of when|{LIMIT} after) (?P<a>.+)$'),
    re.compile(rf'^(?P<a>.+?) before (?P<b>.+?), with no more than \d+ seconds? between them$'),
    re.compile(rf'^(?P<a>.+?)(?:, and|;) (?:{LIMIT} of that, )?(?P<b>.+?)(?: {LIMIT})?$'),
]


def _plain(t):
    t = re.sub(r'\bfinishes changing lanes\b', 'changes lanes', t)
    t = re.sub(r'\bstarts to change\b', 'changes', t)
    return t


def translate(text):
    try:
        return translate_v3(text)
    except Unparsed:
        pass
    t = text.strip().rstrip('.').strip(); t = t[0].lower() + t[1:] if t else t
    if not TIMED.search(t):
        raise Unparsed(t)
    t = _plain(t)
    for pattern in FRAMES:
        m = pattern.match(t)
        if m:
            try:
                return translate_v3(f"After {m.group('a')}, {m.group('b')}.")
            except Unparsed:
                continue
    raise Unparsed(t)


__all__ = ['translate', 'Unparsed']
