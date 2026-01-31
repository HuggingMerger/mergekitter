# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only

from typing import List

import torch

from mergekit.architecture import WeightInfo
from mergekit.merge_methods.easy_define import merge_method
from mergekit.merge_methods.rectify_embed import rectify_embed_sizes


@merge_method(
    name="delerp",
    pretty_name="DeLERP",
    reference_url="https://huggingface.co/blog/grimjim/delerp-merge-method",
)
def delerp_merge(
    tensors: List[torch.Tensor],
    base_tensor: torch.Tensor,
    output_weight: WeightInfo,
    t: float,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    # Rectify embeddings
    all_tensors = [base_tensor] + tensors
    rectify_embed_sizes(output_weight, all_tensors)
    base_tensor, v1 = all_tensors[0], all_tensors[1]

    v0_f = base_tensor.to(torch.float32)
    v1_f = v1.to(torch.float32)

    # Interpolate Direction
    mix = torch.lerp(v0_f, v1_f, t)
    norm_mix = torch.linalg.norm(mix)

    if norm_mix < epsilon:
        return mix.to(base_tensor.dtype)

    dir_mix = mix / norm_mix

    # Interpolate Magnitude
    norm_v0 = torch.linalg.norm(v0_f)
    norm_v1 = torch.linalg.norm(v1_f)
    target_mag = torch.maximum(norm_v0, norm_v1)

    return (dir_mix * target_mag).to(base_tensor.dtype)
