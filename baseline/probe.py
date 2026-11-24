"""Cross-validated linear attribute probes (torch LBFGS logistic regression) and AUC."""
from __future__ import annotations

import numpy as np
import torch


def standardize(x, train):
    mean, std = x[train].mean(0), x[train].std(0) + 1e-6
    return (x - mean) / std


def fit_logistic(x, y, weight_decay=1e-2):
    x = torch.as_tensor(x, dtype=torch.float32); y = torch.as_tensor(y, dtype=torch.float32)
    w = torch.zeros(x.shape[1], requires_grad=True); b = torch.zeros(1, requires_grad=True)
    prior = float(y.mean().clamp(1e-3, 1 - 1e-3))
    optimizer = torch.optim.LBFGS([w, b], max_iter=100, line_search_fn='strong_wolfe')
    pos_weight = torch.tensor((1 - prior) / prior)
    def closure():
        optimizer.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(x @ w + b, y, pos_weight=pos_weight) + weight_decay * w.square().sum()
        loss.backward(); return loss
    optimizer.step(closure)
    return w.detach().numpy(), float(b.detach())


def auc(scores, labels):
    scores, labels = np.asarray(scores, dtype=float), np.asarray(labels, dtype=bool)
    positives, negatives = labels.sum(), (~labels).sum()
    if not positives or not negatives:
        return float('nan')
    order = np.argsort(scores, kind='stable'); ranks = np.empty(len(scores))
    # Average ranks over ties so constant scores give exactly 0.5.
    sorted_scores = scores[order]; i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1; i = j + 1
    return float((ranks[labels].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def cross_validated(features, labels, folds, minimum_positives=3):
    """Out-of-fold probabilities for one binary label; features are standardized within each fold."""
    labels = np.asarray(labels, dtype=bool); out = np.zeros(len(labels))
    for fold in np.unique(folds):
        train, test = folds != fold, folds == fold
        if labels[train].sum() < minimum_positives:
            out[test] = labels[train].mean(); continue
        x = standardize(features, train)
        w, b = fit_logistic(x[train], labels[train])
        out[test] = 1 / (1 + np.exp(-(x[test] @ w + b)))
    return out
