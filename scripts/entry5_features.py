"""Entry-5 recipe features shared by the final scorer fit and the audit check (2026-10-23).

ARMS names the verifier each recipe uses, always on candidates ranked as the container ranks them (entry-3 recipe with
the 32-frame atom answers):
  J1p     zero-shot verifier (each fold's atom adapter; audit-mode adapter for the audit set), top 30
  J2-30   fine-tuned multi-task verifier (mtv adapters; runs/entry5/mtv-audit-train for the audit set), top 30, from the
          top-50 runs restricted to each query's top 30
  J2-50   fine-tuned verifier, top 50
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.atoms_views_eval import view
from scripts.candidates32 import top_k
from scripts.learned_scorer import OUT, load, verify_matrix
from scripts.props_val import VV, with_props

ARMS = {'J1p': ('zs-c50', 30), 'J2-30': ('ft-c50', 30), 'J2-50': ('ft-c50', 50)}


def verifier(d, arm, kind, f):
    """kind: 'dev', 'val' or 'audit'. The top-50 answers, restricted to the top 30 when the arm uses 30 candidates."""
    variant, k = ARMS[arm]; prefix = {'dev': '', 'val': 'val-', 'audit': 'audit-'}[kind]
    m = verify_matrix(d, VV / f'{prefix}{variant}-f{f}/verify.npz', 50)
    if k == 30:
        v2 = load(f) if kind == 'dev' else load(f, OUT / 'val-v2') if kind == 'val' else load('audit')
        top = top_k(view(v2, f, '32'), 30); keep = np.zeros(m.shape, bool)
        pos = {v: j for j, v in enumerate(d['held'])}; qi = {q: i for i, q in enumerate(d['dev'])}
        for v, qs in top.items():
            for q in qs:
                keep[qi[q], pos[v]] = True
        m = np.where(keep, m, np.nan)
    return m


def group(arm, kind, f):
    root = OUT / 'val' if kind == 'val' else OUT / 'p3'
    d = with_props(view(load(f, root), f, '32'), f, 'qwen'); d['verify'] = verifier(d, arm, kind, f); d['group'] = f'{kind}-{f}'
    return d
