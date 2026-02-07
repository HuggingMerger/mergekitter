# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only

from typing import List, Optional

import torch

from mergekit.architecture import WeightInfo
from mergekit.merge_methods.easy_define import merge_method
from mergekit.merge_methods.rectify_embed import rectify_embed_sizes

# Remove registry.py entry and add to __init__.py
@merge_method(
    name="swcm",
    pretty_name="SWCM (Signal-Weighted Consensus Merge)",
)
def swcm_merge(
    tensors: List[torch.Tensor],
    output_weight: WeightInfo,
    base_tensor: Optional[torch.Tensor] = None,
    max_iter: int = 5,
    tol: float = 1e-4,
    density_strength: float = 1.0,
    regularization: float = 0.05,
) -> torch.Tensor:

    all_inputs = [base_tensor] + tensors if base_tensor is not None else tensors
    rectify_embed_sizes(output_weight, all_inputs)

    if base_tensor is not None:
        base_tensor = all_inputs[0]
        donor_tensors = all_inputs[1:]
    else:
        donor_tensors = all_inputs
        base_tensor = torch.stack(donor_tensors).mean(dim=0)

    device, dtype = base_tensor.device, base_tensor.dtype
    num_models = len(donor_tensors)

    # Cast max_iter to int
    max_iter_int = int(max_iter)

    deltas = torch.stack([(t - base_tensor).float() for t in donor_tensors])
    tensors.clear()

    norm_dims = tuple(range(1, deltas.ndim))
    d_norms = torch.linalg.norm(deltas, dim=norm_dims, keepdim=True)

    current_mean = deltas.mean(dim=0)
    epsilon = 1e-6

    for _ in range(max_iter_int):
        diffs = deltas - current_mean
        dists = torch.linalg.norm(diffs, dim=norm_dims, keepdim=True)

        layer_scale = torch.linalg.norm(current_mean) + epsilon

        signal = (d_norms / layer_scale).pow(density_strength)
        agreement = 1.0 / (1.0 + (dists / layer_scale))

        scores = signal * agreement
        total_score = scores.sum(dim=0).clamp(min=epsilon)

        # Weighted update across the model stack
        step = (diffs * scores).sum(dim=0) / total_score

        current_mean = current_mean * (1.0 - regularization) + step

        if torch.linalg.norm(step) < tol:
            break

    return (base_tensor + current_mean).to(dtype)
