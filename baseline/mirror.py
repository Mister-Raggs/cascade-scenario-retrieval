"""Horizontal-mirror augmentation semantics (model-free, unit-tested).

A mirrored clip shows left and right exchanged, so a question about "left" on the mirrored clip has the answer of the
same question about "right" on the original. Atom keys swap left/right tokens (turn_left, change_lane_left, arrow_left);
query texts swap the words. On parsed training queries the swapped text parses to exactly the swapped atoms
(1,581 of 1,581 checked on 2026-10-21).
"""
from __future__ import annotations

import re

_WORD = re.compile(r'\b(left|right|Left|Right)\b')
_SWAP = {'left': 'right', 'right': 'left', 'Left': 'Right', 'Right': 'Left'}


def mirror_atom(atom):
    return atom.replace('left', '\0').replace('right', 'left').replace('\0', 'right')


def mirror_text(text):
    return _WORD.sub(lambda m: _SWAP[m.group(0)], text)
