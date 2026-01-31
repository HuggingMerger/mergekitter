# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: BUSL-1.1

from typing import List, Optional

import torch

from mergekit.architecture import WeightInfo
from mergekit.merge_methods.easy_define import merge_method
from mergekit.merge_methods.rectify_embed import rectify_embed_sizes


@merge_method(name="core_space", pretty_name="Core Space Merge")
def core_space_merge(
    tensors: List[torch.Tensor],
    base_tensor: torch.Tensor,
    output_weight: WeightInfo,
    weight: float = 1.0,
    rank: int = 16,
) -> torch.Tensor:
    all_inputs = [base_tensor] + tensors
    rectify_embed_sizes(output_weight, all_inputs)
    base_tensor, tensors = all_inputs[0], all_inputs[1:]

    if base_tensor.ndim < 2:
        avg_delta = torch.stack([t - base_tensor for t in tensors]).mean(dim=0)
        return (base_tensor + avg_delta * weight).to(base_tensor.dtype)

    # Low-Rank Extraction
    lora_as, lora_bs = [], []
    for t in tensors:
        delta = (t - base_tensor).float()
        u, s, vh = torch.linalg.svd(delta, full_matrices=False)
        r = min(rank, s.size(0))
        lora_bs.append(u[:, :r])
        lora_as.append(torch.diag(s[:r]) @ vh[:r, :])
        del delta, u, s, vh

    # Reference Bases
    u_b, _, _ = torch.linalg.svd(torch.cat(lora_bs, dim=1), full_matrices=False)
    _, _, v_a_h = torch.linalg.svd(torch.cat(lora_as, dim=0), full_matrices=False)
    v_a = v_a_h.transpose(-2, -1)

    # Project & Merge
    common_r = min(u_b.shape[1], v_a.shape[1], rank * len(lora_as))
    ub_t, va_trunc = u_b[:, :common_r].t(), v_a[:, :common_r]

    core_sum = torch.zeros((common_r, common_r), device=base_tensor.device)
    for a, b in zip(lora_as, lora_bs):
        core_sum.add_(ub_t @ b @ a @ va_trunc)

    delta_w = u_b[:, :common_r] @ ((core_sum / len(lora_as)) * weight) @ va_trunc.t()
    return (base_tensor + delta_w).to(base_tensor.dtype)
