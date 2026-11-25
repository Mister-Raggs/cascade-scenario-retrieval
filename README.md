# CASCADE scenario retrieval

Text-to-video retrieval for driving scenarios, built for the AV Causal Scenario Retrieval Challenge (NeurIPS 2026 World
Models in Physical AI workshop). Given a plain-English description such as

> A signaling bus turns left while an adult pedestrian walks.

the system ranks dashcam clips so that the ones matching every part of it (who, what, which qualifier, and in which
order) come first. The final entry scored **0.434 R-Precision** on the hidden test set, up from 0.243 for the first entry and 0.264 for the organizers' baseline.

## How it works

1. **Parse.** Regular-expression rules turn each query into yes/no facts ("atoms": a bus turns left, an adult
   pedestrian walks) joined by temporal relations, plus qualifiers ("properties": the bus is signaling).
2. **Ask.** A LoRA-fine-tuned Qwen3.8-27B answers every fact for every clip from 32 frames. Each clip's video prefix is
   encoded once and shared by all of its questions, and the answer is read as yes-minus-no log-odds without generating
   text.
3. **Verify.** For each query's top 50 clips, a second adapter checks the whole sentence. It is trained on near misses:
   clips where every fact is present but the order or the actor is wrong.
4. **Score.** A single linear layer over 25 features (fact answers, small probes on frozen video features, and
   Qwen3-VL-Embedding-8B similarities), trained with a listwise loss on out-of-fold features, produces the ranking.

Everything runs offline in one container on one A100 80 GB: 2h14m at test scale in a rehearsal, inside the 6-hour limit.

| Entry | What it added | Hidden-test R-Precision |
| --- | --- | --- |
| 1 | Embedding similarity with a learned projection head | 0.243 |
| 2 | Facts answered by linear probes on frozen video features | 0.266 |
| 3 | Facts answered by the fine-tuned VLM, plus an ego-motion CNN and a wider parser | 0.350 |
| 4 | Learned linear ranker and a zero-shot whole-sentence check | 0.398 |
| 5 | Property facts, 32 frames, a fine-tuned check on the top 50 | **0.434** |

Details: [docs/method.md](docs/method.md) (pipeline, training data, settings) and [docs/results.md](docs/results.md)
(evaluation protocol, ablations, headroom, engineering, and what did not work).

## Repository layout

```
baseline/   the runtime: the container entrypoint (entry3_run.py), the VLM stage (entry3_vlm.py), the scorer
            (entry4_scorer.py), the query parsers and fact definitions, embedding and probe code
scripts/    research code behind the reported results: adapter training (vlm_atom_v2.py, vlm_multitask.py), the learned
            scorer (learned_scorer.py, fit_entry5_scorer.py), validation-query evaluation (val_queries.py, val_eval.py),
            headroom analysis (entry5_headroom.py), int8 conversion (quantize_q38.py), packaging and offline verification
configs/    the experiment plans behind the final recipe (adapters in entry4/ and entry5/)
tests/      unit tests (synthetic fixtures only)
docs/       method and results
```

Each research script states its evaluation rule in its docstring; rules were fixed before results were seen.

## Setup

Python 3.12 and `ffmpeg`/`ffprobe` are required.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
git clone https://github.com/nv-tlabs/cascade-devkit.git vendor/cascade-devkit
git -C vendor/cascade-devkit checkout 776102861a4fc587c7ca2141647d4562b93b7f15
.venv/bin/python -m unittest discover -s tests
```

The VLM stage uses a separate environment pinned in `scripts/package_entry3.py` (Transformers 5.18.0 and
flash-linear-attention).

**Models** (not included):

- `Qwen/Qwen3-VL-Embedding-8B`, revision `2c4565515e0f265c6511776e7193b22c0968ddc7`;
- `Qwen/Qwen3.8-27B`, revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.

**Data** (not included): the CASCADE dataset is available from NVIDIA under the NVIDIA Autonomous Vehicle Dataset License
(gated; non-commercial research; no redistribution). This repository contains no clips, annotations or dataset queries,
and the research scripts expect intermediate outputs (features, adapters, predictions) that are not distributed.

## License

Apache-2.0 (see [LICENSE](LICENSE) and [NOTICE](NOTICE)). Third-party components keep their own licences
([THIRD_PARTY.md](THIRD_PARTY.md), [third_party_licenses/](third_party_licenses/)).
