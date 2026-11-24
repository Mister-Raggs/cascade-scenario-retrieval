"""Shared synthetic partition fixture for the head and query-side tests."""
import numpy as np

SETTINGS = {'rank': 2, 'batch_size': 2, 'learning_rate': .01, 'weight_decay': .001,
            'temperature': .1, 'anchor_weight': 1}


def fixture():
    parent = {'query_groups': {'fit': ['q1', 'q2', 'q3'], 'dev': ['qd'], 'audit': ['qa']},
              'development_video_groups': [['held'], ['v1', 'v2', 'n1', 'n2']], 'audit_video_groups': [['audit']]}
    qrels = {'q1': {'relevant_video_ids': ['v1', 'audit']}, 'q2': {'relevant_video_ids': ['v2']},
             'q3': {'relevant_video_ids': ['held', 'v1']},
             'qd': {'relevant_video_ids': ['held', 'v2']}, 'qa': {'relevant_video_ids': ['audit']}}
    qids = ['q1', 'q2', 'q3', 'qd', 'qa']; vids = ['held', 'v1', 'v2', 'n1', 'n2', 'audit']
    q = np.asarray([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
    v = np.asarray([[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [.5, .5, .5, .5], [0, 0, 0, 1]], dtype=np.float32)
    return parent, qrels, qids, vids, q, v
