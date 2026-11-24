"""Shrink the pinned Qwen3.8-27B checkpoint for the size-limited entry-3 package (2026-10-21).

  lossless  drop the multi-token-prediction (mtp.*) weights, which inference never loads; keep only the lm_head rows of
            the yes/no answer tokens (the yes-minus-no log-odds read nothing else), saved separately
  int8      language-model 2-D linear weights stored as int8 with one float32 scale per output row (symmetric, round to
            nearest); the runtime multiplies them back to bf16 after loading, so the model code path is unchanged
Embeddings, the vision tower, norms and every non-2-D tensor stay bf16. Output keeps the original tensor keys and shard
names, plus int8_scales.safetensors, lm_head_rows.safetensors and model_provenance.json (source revision and every file
hash). Usage: quantize_q38.py --source DIR --out DIR
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from baseline.io import sha256_file

SMALL = ('config.json', 'generation_config.json', 'chat_template.jinja', 'merges.txt', 'vocab.json', 'tokenizer.json',
         'tokenizer_config.json', 'preprocessor_config.json', 'video_preprocessor_config.json', 'LICENSE')


def quantizable(key, shape):
    return (len(shape) == 2 and key.endswith('.weight') and 'language_model.layers.' in key and 'norm' not in key
            and 'mtp' not in key)


def main():
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    from transformers import AutoTokenizer
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, required=True); ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=False)
    provenance = json.loads((a.source / 'model_provenance.json').read_text())
    index = json.loads((a.source / 'model.safetensors.index.json').read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(a.source), local_files_only=True)
    answer_ids = sorted({tokenizer.encode(w, add_special_tokens=False)[0] for w in ['yes', 'Yes', ' yes', ' Yes', 'no', 'No', ' no', ' No']})
    weight_map, scales, stats = {}, {}, {'int8': 0, 'bf16': 0, 'dropped': 0}
    for shard in sorted(set(index['weight_map'].values())):
        if sha256_file(a.source / shard) != provenance['files'][shard]:
            raise ValueError('Pinned shard changed: ' + shard)
        out = {}
        with safe_open(str(a.source / shard), 'pt') as f:
            for key in f.keys():
                w = f.get_tensor(key)
                if key.startswith('mtp.') or '.mtp.' in key:
                    stats['dropped'] += w.numel() * 2; continue
                if key == 'lm_head.weight':
                    save_file({'weight': w[answer_ids].contiguous(), 'token_ids': torch.tensor(answer_ids)}, str(a.out / 'lm_head_rows.safetensors'))
                    stats['dropped'] += w.numel() * 2; continue
                if quantizable(key, w.shape):
                    x = w.float(); s = (x.abs().amax(1) / 127).clamp(min=1e-12)
                    out[key] = torch.round(x / s[:, None]).clamp(-127, 127).to(torch.int8).contiguous(); scales[key] = s
                    stats['int8'] += w.numel()
                else:
                    out[key] = w.contiguous(); stats['bf16'] += w.numel() * 2
                weight_map[key] = shard
        save_file(out, str(a.out / shard), metadata={'format': 'pt'}); print('wrote', shard, flush=True)
    save_file(scales, str(a.out / 'int8_scales.safetensors'))
    json.dump({'metadata': {'quantization': 'int8 per-row symmetric (language-model 2-D linear weights)'}, 'weight_map': weight_map},
              open(a.out / 'model.safetensors.index.json', 'w'), indent=2)
    for name in SMALL:
        shutil.copyfile(a.source / name, a.out / name)
    files = {p.name: sha256_file(p) for p in sorted(a.out.iterdir()) if p.is_file() and p.name != 'model_provenance.json'}
    json.dump({**{k: provenance[k] for k in ('repo', 'revision', 'declared_license', 'license_files')}, 'files': files,
               'derived_from_files': provenance['files'], 'transform': __doc__.split('Embeddings')[0].strip(),
               'answer_token_ids': answer_ids, 'stats_bytes': {'int8': stats['int8'], 'bf16': stats['bf16'], 'dropped_bf16': stats['dropped']}},
              open(a.out / 'model_provenance.json', 'w'), indent=2)
    print(json.dumps(stats), flush=True)


if __name__ == '__main__':
    main()
