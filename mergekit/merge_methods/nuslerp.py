# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: BUSL-1.1

from typing import List, Optional

import torch

from mergekit.architecture import WeightInfo
from mergekit.merge_methods.easy_define import merge_method
from mergekit.merge_methods.rectify_embed import rectify_embed_sizes


@merge_method(
    name="nuslerp",
    pretty_name="NuSLERP",
    reference_url="https://arxiv.org/abs/2412.19819",
)
def nuslerp_merge(
    tensors: List[torch.Tensor],
    output_weight: WeightInfo,
    base_tensor: Optional[torch.Tensor] = None,
    weight: List[float] = 0.5,
    nuslerp_row_wise: bool = False,
    nuslerp_flatten: bool = True,
    geodesic: bool = False,
    lambda_: Optional[float] = None,
) -> torch.Tensor:

    all_inputs = [base_tensor] + tensors if base_tensor is not None else tensors
    rectify_embed_sizes(output_weight, all_inputs)

    if base_tensor is not None:
        base_tensor = all_inputs[0]
        v0, v1 = all_inputs[1] - base_tensor, all_inputs[2] - base_tensor
    else:
        v0, v1 = all_inputs[0], all_inputs[1]

    t = lambda_ if lambda_ is not None else (weight[1] / max(sum(weight), 1e-6))
    orig_shape = v0.shape

    if nuslerp_flatten:
        v0, v1 = v0.flatten(), v1.flatten()
    elif nuslerp_row_wise:
        v0, v1 = v0.transpose(0, -1), v1.transpose(0, -1)

    res_unit = _vectorized_slerp(v0, v1, t)

    if geodesic:
        m0, m1 = torch.linalg.norm(v0.float()), torch.linalg.norm(v1.float())
        mag = (m0 ** (1 - t)) * (m1**t)
        res = res_unit * mag.to(v0.dtype)
    else:
        res = res_unit

    if not nuslerp_flatten and nuslerp_row_wise:
        res = res.transpose(0, -1)

    res = res.view(orig_shape)
    return (base_tensor + res) if base_tensor is not None else res


def _vectorized_slerp(v0, v1, t, eps=1e-8):
    v0_u = v0 / torch.linalg.norm(v0, dim=-1, keepdim=True).clamp(min=eps)
    v1_u = v1 / torch.linalg.norm(v1, dim=-1, keepdim=True).clamp(min=eps)
    cos_theta = torch.sum(v0_u * v1_u, dim=-1, keepdim=True).clamp(-1.0, 1.0)
    theta = torch.acos(cos_theta)
    sin_theta = torch.sin(theta)
    res = (
        torch.sin((1 - t) * theta) * v0_u + torch.sin(t * theta) * v1_u
    ) / sin_theta.clamp(min=eps)
    return torch.where(sin_theta.abs() < eps, torch.lerp(v0_u, v1_u, t), res)
