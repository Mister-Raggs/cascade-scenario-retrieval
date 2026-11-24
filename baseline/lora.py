"""Small language-only low-rank adapters; original model weights stay frozen."""
from __future__ import annotations

import math
import re
import torch
from torch import nn
from torch.nn import functional as F

TARGET = re.compile(r'(?:^|\.)language_model\.layers\.\d+\.self_attn\.(q_proj|v_proj)$')


class LoRALinear(nn.Module):
    def __init__(self, base, rank=8, alpha=16):
        super().__init__()
        if not isinstance(base, nn.Linear) or isinstance(rank, bool) or not isinstance(rank, int) or not 0 < rank <= min(base.in_features, base.out_features):
            raise ValueError('LoRA requires a Linear layer and a valid positive rank')
        if isinstance(alpha, bool) or not math.isfinite(alpha) or alpha <= 0:
            raise ValueError('LoRA alpha must be finite and positive')
        self.base = base.requires_grad_(False)
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, values):
        original = self.base(values)
        update = F.linear(F.linear(values.float(), self.lora_A), self.lora_B)
        return original + (self.scale * update).to(original.dtype)


VISION_TARGET = re.compile(r'(?:^|\.)visual\.blocks\.\d+\.attn\.(qkv|proj)$')


def attach_language_lora(model, rank=8, alpha=16):
    return attach_lora(model, TARGET, rank, alpha)


def attach_vision_lora(model, rank=16, alpha=32, last_blocks=None):
    """LoRA on vision-tower attention qkv/proj; `last_blocks` limits it to the final N blocks."""
    pattern = VISION_TARGET
    if last_blocks is not None:
        depth = 1 + max(int(re.search(r'blocks\.(\d+)\.', n).group(1)) for n, _ in model.named_modules() if VISION_TARGET.search(n))
        keep = '|'.join(str(i) for i in range(depth - last_blocks, depth))
        pattern = re.compile(r'(?:^|\.)visual\.blocks\.(?:' + keep + r')\.attn\.(qkv|proj)$')
    return attach_lora(model, pattern, rank, alpha)


def attach_lora(model, pattern, rank, alpha):
    if any(isinstance(module, LoRALinear) for module in model.modules()):
        raise ValueError('Adapters already attached')
    selected = [(name, module) for name, module in model.named_modules() if pattern.search(name)]
    if not selected or any(not isinstance(module, nn.Linear) for _, module in selected):
        raise ValueError('Expected Linear projections matching the adapter target')
    adapters = [(name, LoRALinear(module, rank, alpha)) for name, module in selected]
    model.requires_grad_(False)
    for name, adapter in adapters:
        adapter.lora_A.requires_grad_(True)
        adapter.lora_B.requires_grad_(True)
        parent, child = name.rsplit('.', 1)
        setattr(model.get_submodule(parent), child, adapter)
    return [name for name, _ in adapters]


def adapter_state(model):
    return {name: p.detach().cpu().contiguous().clone() for name, p in model.named_parameters()
            if name.endswith(('.lora_A', '.lora_B'))}


def load_adapter_state(model, state):
    parameters = dict(model.named_parameters())
    expected = {name for name in parameters if name.endswith(('.lora_A', '.lora_B'))}
    if not expected or set(state) != expected:
        raise ValueError('Adapter parameter names differ')
    if any(state[name].shape != parameters[name].shape or not torch.isfinite(state[name]).all() for name in expected):
        raise ValueError('Invalid adapter shape or values')
    with torch.no_grad():
        for name in expected:
            parameters[name].copy_(state[name].to(device=parameters[name].device, dtype=parameters[name].dtype))


def multi_positive_loss(scores, positives, negatives):
    """Each known positive competes only against allowed qrel negatives.

    Other known positives and masked ambiguous cells never become negatives.
    """
    if scores.ndim != 2 or scores.shape != positives.shape or scores.shape != negatives.shape:
        raise ValueError('Contrastive score/mask shapes differ')
    if positives.dtype != torch.bool or negatives.dtype != torch.bool or (positives & negatives).any():
        raise ValueError('Positive and negative masks must be disjoint boolean tensors')
    if not torch.isfinite(scores).all() or not positives.any(1).all() or not negatives.any(1).all():
        raise ValueError('Every query needs finite scores, a known positive and an allowed negative')
    log_negatives = torch.logsumexp(scores.masked_fill(~negatives, -torch.inf), dim=1, keepdim=True)
    terms = F.softplus(log_negatives - scores).masked_fill(~positives, 0)
    return (terms.sum(1) / positives.sum(1)).mean()
