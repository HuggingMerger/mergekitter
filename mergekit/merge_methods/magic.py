# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only

import logging
from typing import List

import torch

from mergekit.architecture import WeightInfo
from mergekit.merge_methods.easy_define import merge_method
from mergekit.merge_methods.rectify_embed import rectify_embed_sizes

LOG = logging.getLogger(__name__)


@merge_method(
    name="magic",
    pretty_name="MAGIC (Magnitude Calibration)",
    reference_url="https://arxiv.org/abs/2512.19320",
)
def magic_merge(
    tensors: List[torch.Tensor],
    base_tensor: torch.Tensor,
    output_weight: WeightInfo,
    weight: List[float] = 1.0,
    calibration_strength: float = 1.0,
    use_svc: bool = True,
    num_power_iter: int = 15,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """
    MAGIC (Magnitude Calibration)
    Adjusts the magnitude of the merged delta to match the weighted average
    of the input deltas' magnitudes. This compensates for the magnitude
    collapse that occurs during linear averaging.
    """
    all_inputs = [base_tensor] + tensors
    rectify_embed_sizes(output_weight, all_inputs)
    base_tensor, donors = all_inputs[0], all_inputs[1:]

    if not donors:
        return base_tensor

    device, dtype = base_tensor.device, base_tensor.dtype
    
    iters = int(num_power_iter)

    avg_delta = torch.zeros_like(base_tensor, dtype=torch.float32)
    sum_input_norms = 0.0
    total_w = 0.0

    for i, t in enumerate(donors):
        w_i = weight[i]
        delta = (t - base_tensor).float()

        # Measure magnitude (Spectral Norm/SVC vs Frobenius)
        if use_svc and delta.ndim >= 2:
            norm_i = _estimate_spectral_norm(delta, iters)
        else:
            norm_i = torch.linalg.norm(delta)

        sum_input_norms += norm_i.item() * w_i
        avg_delta.add_(delta, alpha=w_i)
        total_w += w_i

    donors.clear()

    avg_delta.div_(max(total_w, epsilon))
    target_norm = sum_input_norms / max(total_w, epsilon)

    if use_svc and avg_delta.ndim >= 2:
        merged_norm = _estimate_spectral_norm(avg_delta, iters)
    else:
        merged_norm = torch.linalg.norm(avg_delta)

    ratio = target_norm / max(merged_norm.item(), epsilon)
    actual_scale = 1.0 + (calibration_strength * (ratio - 1.0))

    if LOG.isEnabledFor(logging.DEBUG) and "vision" not in output_weight.name.lower():
        LOG.debug(
            f"MAGIC [{output_weight.name}] Target: {target_norm:.4f} | "
            f"Merged: {merged_norm.item():.4f} | Ratio: {ratio:.4f} | Scale: {actual_scale:.4f}"
        )

    return (base_tensor + (avg_delta * actual_scale)).to(dtype)


@torch.no_grad()
def _estimate_spectral_norm(tensor: torch.Tensor, iters: int) -> torch.Tensor:
    """
    Estimate the spectral norm (largest singular value) using Power Iteration.
    For tensors > 2D, they are flattened into matrices.
    """
    if tensor.ndim > 2:
        tensor = tensor.view(tensor.shape[0], -1)

    m, n = tensor.shape
    u = torch.randn((n, 1), device=tensor.device, dtype=tensor.dtype)
    u /= torch.linalg.norm(u).clamp(min=1e-12)

    for _ in range(iters):
        # Power iteration: v = Au; u = A^T v
        v = torch.matmul(tensor, u)
        v /= torch.linalg.norm(v).clamp(min=1e-12)
        u = torch.matmul(tensor.t(), v)
        u /= torch.linalg.norm(u).clamp(min=1e-12)

    return torch.linalg.norm(torch.matmul(tensor, u))
