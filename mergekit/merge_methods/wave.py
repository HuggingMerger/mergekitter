# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only

from typing import List, Optional

import torch
import torch.nn.functional as F

from mergekit.architecture import WeightInfo
from mergekit.merge_methods.easy_define import merge_method
from mergekit.merge_methods.rectify_embed import rectify_embed_sizes


@merge_method(name="wave", pretty_name="WAVE (Ultra-Low Memory)")
def wave_merge(
    tensors: List[torch.Tensor],
    base_tensor: torch.Tensor,
    output_weight: WeightInfo,
    weight: List[float] = 1.0,
    synergy: float = 0.5,
    epsilon: float = 1e-6,
    entropy: float = 0.1,
) -> torch.Tensor:
    all_tensors = [base_tensor] + tensors
    rectify_embed_sizes(output_weight, all_tensors)
    base_tensor, tensors = all_tensors[0], all_tensors[1:]

    device, dtype = base_tensor.device, base_tensor.dtype
    num_models = len(tensors)

    sum_delta = torch.zeros_like(base_tensor, dtype=torch.float32)
    sum_sq_delta = torch.zeros_like(base_tensor, dtype=torch.float32)

    for t in tensors:
        delta = (t - base_tensor).float()
        sum_delta.add_(delta)
        sum_sq_delta.add_(delta.pow(2))

    # Var = E[X^2] - (E[X])^2
    variance = (sum_sq_delta / num_models) - (sum_delta / num_models).pow(2)
    del sum_sq_delta, sum_delta  # Free buffers

    if entropy > 0:
        variance.mul_(1.0 + (torch.rand_like(variance) * 2 - 1) * entropy)

    # Thresholding
    if synergy >= 1.0:
        mask = variance > epsilon
    elif synergy > 0.0:
        numel = variance.numel()
        sample = variance.view(-1)[torch.randint(0, numel, (min(numel, 250_000),))]
        mask = variance >= torch.quantile(sample.float(), 1.0 - synergy)
    else:
        mask = torch.zeros_like(variance, dtype=torch.bool)
    del variance

    best_delta = torch.zeros_like(base_tensor)
    max_importance = torch.full_like(base_tensor, float("-inf"), dtype=torch.float32)
    stable_delta = torch.zeros_like(base_tensor)
    total_weight = 0.0

    for i, t in enumerate(tensors):
        delta = t - base_tensor
        w = weight[i]
        stable_delta.add_(delta, alpha=w)
        total_weight += w

        # Only compute importance where the mask is actually True
        imp = _compute_single_importance(t, base_tensor)
        is_better = imp > max_importance
        best_delta = torch.where(is_better, delta, best_delta)
        max_importance = torch.where(is_better, imp, max_importance)

    tensors.clear()
    stable_delta.div_(max(total_weight, 1e-6))
    return base_tensor + torch.where(mask, best_delta, stable_delta)


def _compute_single_importance(param, base):
    p = F.softmax(param.float(), dim=-1) + 1e-8
    q = F.softmax(base.float(), dim=-1) + 1e-8
    kl = torch.sum(p * torch.log(p / q), dim=-1)
    if kl.dim() < param.dim():
        kl = kl.unsqueeze(-1)
    return (param - base).abs().float() * kl
