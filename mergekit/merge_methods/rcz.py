# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only

from typing import List
import torch

from mergekit.architecture import WeightInfo
from mergekit.merge_methods.easy_define import merge_method
from mergekit.merge_methods.rectify_embed import rectify_embed_sizes


@merge_method(
    name="rcz",
    pretty_name="RCZ (Relative Consensus Zone)",
)
def rcz_merge(
    tensors: List[torch.Tensor],
    base_tensor: torch.Tensor,
    output_weight: WeightInfo,
    max_rho: float = 0.25,  # Density ratio upper bound (ρ_max)
    tol: float = 1e-5,  # Perturbation threshold (ε)
    lambda_scale: float = 0.18,  # Scaling coefficient (λ)
    high_precision: bool = True, 
) -> torch.Tensor:
    if not tensors:
        return base_tensor

    all_tensors = [base_tensor] + tensors
    rectify_embed_sizes(output_weight, all_tensors)
    base_tensor = all_tensors[0]
    tensors = all_tensors[1:]

    working_dtype = torch.float32 if high_precision else base_tensor.dtype
    device = base_tensor.device

    usage_counts = torch.zeros_like(base_tensor, dtype=torch.uint8)
    for t in tensors:
        usage_counts.add_((t - base_tensor).abs() > tol)

    overlap_mask = usage_counts > 1
    unique_mask = usage_counts == 1

    norm_divisor = usage_counts.to(working_dtype).clamp(min=1.0)

    total_delta = torch.zeros_like(base_tensor, dtype=working_dtype)

    for t in tensors:
        t_work = t.to(working_dtype)
        base_work = base_tensor.to(working_dtype)
        delta = t_work.sub(base_work)

        active_mask = delta.abs() > tol

        shared_count = (active_mask & overlap_mask).sum().item()
        unique_count = (active_mask & unique_mask).sum().item()

        density_ratio = shared_count / max(unique_count, tol)
        boost = 1.0 + lambda_scale * min(max(density_ratio, 0.0), max_rho)

        multiplier = torch.where(
            unique_mask,
            torch.tensor(boost, dtype=working_dtype, device=device),
            1.0 / norm_divisor,
        )

        delta.mul_(multiplier)
        total_delta.add_(delta)

        del t_work, base_work, delta, active_mask, multiplier

    return base_tensor.to(working_dtype).add(total_delta).to(base_tensor.dtype)
