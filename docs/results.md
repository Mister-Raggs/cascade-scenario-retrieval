# Results

R-Precision (RP): for a query with R relevant clips, the fraction of the top R results that are relevant. Hidden-test
numbers come from the official leaderboard; everything else is a held-out measurement on the released data. The
populations differ, so numbers are comparable only within a table.

## Hidden test

| Entry | What it added | RP |
| --- | --- | --- |
| Organizers' baseline | | 0.264 |
| 1 | Embedding similarity with a learned projection head | 0.243 |
| 2 | Atoms answered by linear probes on frozen video features | 0.266 |
| 3 | Atoms answered by the fine-tuned VLM, plus an ego-motion CNN and a wider parser | 0.350 |
| 4 | Learned linear scorer, parser v3, zero-shot whole-query verifier (top 30) | 0.398 |
| 5 | Property atoms, 32-frame answers, fine-tuned verifier on the top 50 | **0.434** |

Entry 5 by tier (Easy / Medium / Hard): 0.562 / 0.439 / 0.309, against 0.551 / 0.392 / 0.261 for entry 4. MAP 0.447,
Recall@10 0.394. First on the public leaderboard when it closed; final results pending.

## How changes were judged

- **Development folds:** three folds of clips × two projection-head seeds = 6 cells. A change was adopted only if mean RP
  rose by more than 0.001, at least 4 of 6 cells improved, no fold got worse on average and MAP did not drop. Each rule
  was written into its script before the run.
- **Audit set:** 773 queries × 517 clips held out of everything, used five times in total, each time with the pass rule
  committed first.
- **Validation queries (entry 5):** the 711 development queries turned out to be only 64 scenarios (paraphrases with
  identical relevant clips), so the 3,108 validation queries (298 scenarios) became the primary test. Scenarios were split
  into two seeded halves, so every test is on unseen clips and unseen scenarios (`scripts/val_eval.py`).

Three ideas rejected on development queries passed on validation queries:

| Idea | Development folds | Validation queries |
| --- | --- | --- |
| Property atoms (VLM, zero-shot) | −0.0003, 3/6 | +0.0059, 6/6 |
| Fine-tuned verifier (vs zero-shot) | +0.0014, 4/6 | +0.0088, 6/6 |
| Verifier on the top 50 (vs top 30) | +0.0004, 4/6 | +0.0071, 6/6 |

## Ablations

| Change | Test | Effect | Cells |
| --- | --- | --- | --- |
| Linear scorer vs the entry-3 rank fusion | development | +0.0144 | 6/6 |
| a 2-layer MLP instead | development | +0.0075 | 4/6 |
| a bilinear query-text model instead | development | −0.0052 | 3/6 |
| 32-frame atom answers (vs 16) | validation | +0.005 | 6/6 |
| Scorer trained with 25% of validation queries | validation | +0.007 | 6/6 |
| Property atoms | validation | +0.006 | 6/6 |
| Fine-tuned verifier on the top 50 | validation | +0.017 | 6/6 |
| Entry-4 recipe → entry 5 | validation | 0.1757 → 0.2111 | 6/6 |
| | development | 0.3016 → 0.3207 | 6/6 |
| | audit | 0.2480 → 0.2814 | one look |

## Headroom

Replacing one component with ground truth shows what perfecting it is worth (`scripts/entry5_headroom.py`). Perfect atom
answers reach about 0.64 RP on development queries but only about 0.41 on validation queries: the atom representation
loses meaning (order, binding, qualifiers), which pointed to property atoms and the whole-query verifier. On development
folds, a verifier whose within-candidate AUC rose from 0.69 to 0.86 was estimated to be worth about +0.05 RP.

## Engineering

| Stage (A100, 650 clips × 2,000 queries) | Entry 4 | Entry 5 |
| --- | --- | --- |
| Embeddings, windows and motion | 44 min | 30 min |
| Atom and property questions | 28 min | 48 min |
| Verifier (top 30 / top 50: 60,000 / 100,000 pairs) | 29 min | 50 min |
| **Total** | **1h48m** | **2h14m** |
| Peak GPU memory | 62 GiB | 60.5 GiB |

- int8 base weights plus dropping unused weights: 55.6 → 27.9 GB, at −0.0018 RP (MAP +0.0012) on fold 0 of the entry-3
  recipe. The artifact repository was rejected at 63.3 GB and accepted at 44.06 GB; the limit is not documented.
- Container vs research code: embedding and motion features within 1.5e-8 (motion exactly), scorer features within 1e-6,
  top-10 rankings identical on 64 of 64 validation queries (`scripts/verify_entry3_package.py`).

## What did not work

Each row failed a rule fixed before its result, or was closed as a diagnostic. Development-fold RP unless marked.

| Idea | Result |
| --- | --- |
| Cosmos-Reason2-32B, Gemma 4 31B as atom answerers (fold-0 pilots, top-half LoRA) | −0.025, −0.007; Cosmos matched or slightly beat Qwen3.8's per-atom AUC yet ranked worse |
| Qwen3-VL-30B-A3B ensemble with Qwen3.8 | helped development, flat on validation |
| 64 frames for the verifier (two layouts) | AUC 0.741 / 0.727 vs 0.754 at 32 frames (fold 0) |
| 32 frames for the verifier | +0.0007 development, −0.0005 validation |
| Higher resolution (448×800) | flat |
| LoRA on the 8B embedding model | −0.032 / −0.034 on fold 0 (566 fitting queries; overfits) |
| Mirror augmentation or averaging | −0.0004 (development); −0.0012 atoms, −0.0025 verifier (validation) |
| MLP, bilinear and gradient-boosted scorers; calibration, rank and per-family features; bagging | failed or unstable |
| Ego-motion supervision for the ego CNN | +0.0002, ego AUC unchanged |
| Paraphrase consensus | ≈ 0 even with an oracle |
| Perfect event timing (oracle) | +0.005 on validation: not worth per-window perception |
| Atom adapter variants (6,000 steps, rank 64, lr 2e-4, ego specialist) | trade-offs; 6,000 steps: validation +0.008, development −0.025 |
| Verifier scale-up (more training data and steps) | −0.0026, 2/6 on unseen validation scenarios |
