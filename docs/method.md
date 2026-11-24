# Method

The final system (entry 5) has three stages. The code is in `baseline/`; the container entrypoint is
`baseline/entry3_run.py`.

Running example (synthetic, written in the dataset's style and checked against it):

> A signaling bus turns left while an adult pedestrian walks.

| Step | What the example becomes |
| --- | --- |
| Parse (A) | two atoms, a bus turning left and an adult pedestrian walking, joined by "while"; one property, the bus is signaling |
| Embed and probe (A) | raw and projected query-clip similarities; probe log-odds for each atom |
| Ask (B) | one yes/no question per atom and property ("Does a bus turn left at any point in this video?"), scored as yes-minus-no log-odds from 32 frames |
| Verify (B) | if the clip is in this query's top 50, the verifier answers the whole query from 16 frames |
| Score (C) | 25 features through one linear layer; the score sets the clip's place in the query's top 100 |

## Stage A: embeddings, probes and parsing

**Embeddings.** Qwen3-VL-Embedding-8B, frozen, encodes each clip from 4 frames at 256×448 and each query with a
retrieval instruction. A residual low-rank projection head (rank 128) is trained on the training queries with a
positive-aware softmax loss (each relevant clip is contrasted only with non-relevant clips) plus an anchor term that keeps
projected vectors near the raw ones. Both the raw and the projected similarity are kept as features.
Code: `baseline/native_video.py`, `baseline/projection.py`, `baseline/post_submission.py`.

**Probes.** Per-atom logistic regressions on three frozen blocks: the pooled clip embedding, embeddings of 5 time windows,
and an optical-flow motion descriptor (8 fps, 4×6 grid). Block log-odds are averaged per atom. A small temporal CNN on the
motion series adds calibrated probabilities for the ego vehicle's own actions. Code: `baseline/structured_recipe.py`,
`baseline/entry3_recipe.py`.

**Parsing.** Regular-expression rules (no learned model) translate each query into the devkit's DSL and then into a tree
of atoms joined by temporal relations. There are 95 atom types in four families (other road users, traffic lights,
objects, the ego vehicle), such as `agent|action=turn_left|type=bus`, and 10 property types for the qualifiers that atoms
drop (signaling, slow, double-parked, flashing or ego-affecting yellow lights). 97.1% of validation queries parse.
Code: `baseline/query_parser_v3.py`, `baseline/query_struct.py`, `baseline/property_atoms.py`.

## Stage B: answers from Qwen3.8-27B

Each atom becomes a question (`baseline/atom_questions.py`). Qwen3.8-27B with a LoRA adapter (rank 32, α 64, on attention
q/v and the linear-attention qkv projections) answers from 32 frames at 352×640. Nothing is generated: the score is
`logsumexp(yes-token logits) − logsumexp(no-token logits)` at the first answer position.

**Shared video prefix.** The video occupies thousands of tokens and a question a few dozen, so each clip's prefix is
encoded once and its key/value cache is shared by all its questions, which run as batched suffixes. The first clip of
every run checks that the split prompt's tokens and logits match the full prompt's. The model's linear-attention layers
update their cache in place, so training through the shared prefix needed an out-of-place cache update
(`baseline/entry3_vlm.py`, `scripts/vlm_atom_v2.py`).

**Verifier.** For each query's top 50 clips under a cheaper ranking (embedding similarities rank-fused with summed atom
log-probabilities), a second adapter answers "Does this video match the following description?" from 16 frames. It is
trained on atom questions together with whole-query questions whose negatives include **near misses**: clips where every
parsed atom of the query is present but which are still not relevant (wrong order, binding or qualifier). The runtime
swaps adapters between the atom pass and the verifier pass. Code: `scripts/vlm_multitask.py`, `baseline/query_pairs.py`.

## Stage C: a linear scorer

For every query-clip pair, 25 features (`baseline/entry4_scorer.py`):

- for up to two of the query's atoms: probe, VLM and averaged log-odds, a VLM-available flag and the family (16);
- the raw and projected similarities and the summed atom log-probability, z-scored within the query, a parsed flag and
  the atom count (5);
- the summed property log-probability and a has-property flag (2);
- the verifier log-odds z-scored within the query's candidates, and an in-candidates flag (2).

One linear layer is trained with a listwise softmax loss (Adam, lr 1e-2, 10 epochs) on **out-of-fold** features: every
VLM answer in its training data comes from an adapter that never trained on that clip, so the scorer never learns to
over-trust in-sample answers. Code: `scripts/learned_scorer.py`, `scripts/fit_entry5_scorer.py`.

## Training data

Only the released CASCADE data: 2,066 annotated clips (camera video from the PhysicalAI-AV dataset), 7,454
training-split queries and 3,108 validation-split queries. Atom labels are clip-level truths derived from the CASCADE
annotations; properties are answered zero-shot.

| Component | Settings |
| --- | --- |
| Projection head | rank 128, 4 epochs, lr 3e-4, temperature 0.05 |
| Atom adapter | LoRA rank 32, α 64, AdamW lr 1e-4, 6,006 steps, 16 questions per clip per step, 16 frames, all clips |
| Verifier adapter | same LoRA settings, 4,001 steps, 10 atom and 6 whole-query questions per step, 16 frames |
| Scorer | out-of-fold features of the development and audit queries plus 25% of the validation queries |

The base Qwen3.8-27B weights ship in int8 (one scale per output row; unused weights dropped; 55.6 → 27.9 GB). The adapters
are trained in bf16. Code: `scripts/quantize_q38.py`.

## Runtime

The submission is a container that reads only the clips and query texts, runs offline on one A100 80 GB and must finish
within 6 hours. A rehearsal at the estimated test scale (650 clips × 2,000 queries) took 2h14m with a peak of 60.5 GiB. A time guard
degrades instead of failing: if the atom pass is projected past 3 hours, the remaining clips are answered from 16 frames,
and at 4h15m the verifier stops, leaving neutral verifier features for queries it has not reached.
