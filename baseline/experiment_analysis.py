"""Paired retrieval diagnostics; descriptive evidence, not hidden-test inference."""
import numpy as np


def prefix_diagnostic(ranking, relevant, k):
    relevant = set(relevant)
    if not relevant:
        return None
    if len(set(ranking)) != len(ranking):
        raise ValueError("Duplicate ranked IDs")
    r = len(relevant)
    hits = len(set(ranking[:k]) & relevant)
    current = len(set(ranking[:r]) & relevant) / r
    ceiling = min(hits, r) / r if r < k else current
    return {"recall": hits / r, "oracle_r_precision": ceiling,
            "current_r_precision": current, "prefix_cannot_change_r_precision": r >= k}


def paired_delta(base, changed, *, seed=0, replicates=1000):
    base, changed = np.asarray(base, dtype=float), np.asarray(changed, dtype=float)
    if base.shape != changed.shape or base.ndim != 1 or not len(base):
        raise ValueError("Paired nonempty equal-length arrays required")
    if not np.isfinite(base).all() or not np.isfinite(changed).all():
        raise ValueError("Nonfinite metrics")
    delta = changed - base
    rng = np.random.default_rng(seed)
    draws = [float(delta[rng.integers(0, len(delta), len(delta))].mean())
             for _ in range(replicates)]
    return {"mean_delta": float(delta.mean()), "improved_queries": int((delta > 1e-12).sum()),
            "regressed_queries": int((delta < -1e-12).sum()),
            "unchanged_queries": int((np.abs(delta) <= 1e-12).sum()),
            "paired_query_bootstrap_95_interval": np.quantile(draws, [.025, .975]).tolist(),
            "bootstrap_seed": seed, "bootstrap_replicates": replicates,
            "interpretation": "Descriptive query bootstrap; shared videos/templates can correlate queries. Not hidden-test generalization evidence."}


def interaction(base, a, b, combined):
    arrays = [np.asarray(x, dtype=float) for x in (base, a, b, combined)]
    if any(x.shape != arrays[0].shape for x in arrays):
        raise ValueError("All factorial cells require identical query coverage")
    base, a, b, combined = arrays
    residual = combined - a - b + base
    return {"interaction_delta": float(residual.mean()),
            "combined_minus_a": float((combined - a).mean()),
            "combined_minus_b": float((combined - b).mean()),
            "interpretation": "Positive residual indicates super-additive score change for these fixed recipes; negative indicates sub-additivity. Bounded R-Precision and ranking effects also contribute. Compatibility requires combined to beat both individual recipes."}
