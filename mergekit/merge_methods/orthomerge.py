# Copyright (C) 2026 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only

import gc
import logging
from typing import List

import torch

from mergekit.architecture import WeightInfo
from mergekit.merge_methods.easy_define import merge_method
from mergekit.merge_methods.rectify_embed import rectify_embed_sizes

LOG = logging.getLogger(__name__)


@merge_method(
    name="ortho_merge",
    pretty_name="Orthogonal Model Merging (OrthoMerge)",
    reference_url="https://arxiv.org/abs/2602.05943",
)
def ortho_merge(
    tensors: List[torch.Tensor],
    base_tensor: torch.Tensor,
    output_weight: WeightInfo,
    weight: List[float] = 1.0,
    lambda_ortho: float = 0.5,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """
    OrthoMerge: Orthogonal-Residual Decoupling
    """
    all_inputs = [base_tensor] + tensors
    rectify_embed_sizes(output_weight, all_inputs)

    tensors.clear()

    base_tensor, donors = all_inputs[0], all_inputs[1:]
    del all_inputs

    if not donors:
        return base_tensor

    device, dtype = base_tensor.device, base_tensor.dtype

    if base_tensor.ndim < 2:
        num = torch.zeros_like(base_tensor, dtype=torch.float32)
        total_w = 0.0
        for i, t in enumerate(donors):
            w_i = weight[i] if isinstance(weight, list) else weight
            num.add_(t.float(), alpha=w_i)
            total_w += w_i
        donors.clear()
        return (num / max(total_w, 1e-8)).to(dtype)

    m_dim, n_dim = base_tensor.shape
    rotate_rows = m_dim <= n_dim
    rot_dim = m_dim if rotate_rows else n_dim

    w0_f32 = base_tensor.float()
    eye = torch.eye(rot_dim, device=device, dtype=torch.float32)

    sum_skew = torch.zeros((rot_dim, rot_dim), device=device, dtype=torch.float32)
    sum_residual = torch.zeros_like(w0_f32)
    total_weight = 0.0

    idx = 0
    while donors:
        wi_raw = donors.pop(0)
        w_i = weight[idx] if isinstance(weight, list) else weight
        wi_f32 = wi_raw.float()
        del wi_raw

        if rotate_rows:
            mat = torch.matmul(wi_f32, w0_f32.t())
        else:
            mat = torch.matmul(wi_f32.t(), w0_f32)

        u, _, vh = torch.linalg.svd(mat, full_matrices=False)
        r_matrix = torch.matmul(u, vh)
        del mat, u, vh

        stable_r = r_matrix + eye
        if epsilon > 0:
            stable_r.diagonal().add_(1e-10)

        skew_q = torch.linalg.solve(stable_r, r_matrix - eye)
        del stable_r, r_matrix

        if rotate_rows:
            rotated_base = torch.matmul(
                (eye + skew_q) @ torch.linalg.inv(eye - skew_q), w0_f32
            )
            residual = wi_f32 - rotated_base
        else:
            r_tmp = torch.linalg.solve(eye - skew_q, eye + skew_q)
            residual = wi_f32 - torch.matmul(w0_f32, r_tmp)
            del r_tmp

        sum_skew.add_(skew_q, alpha=w_i)
        sum_residual.add_(residual, alpha=w_i)
        total_weight += w_i

        del wi_f32, skew_q, residual
        if rot_dim > 4096:
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        idx += 1

    avg_skew = sum_skew / max(total_weight, 1e-8)
    avg_residual = sum_residual / max(total_weight, 1e-8)
    del sum_skew, sum_residual

    inv_part = eye - avg_skew
    inv_part.diagonal().add_(1e-10)
    r_merged = torch.linalg.solve(inv_part, eye + avg_skew)
    del avg_skew, inv_part

    if rotate_rows:
        rotated_w0 = torch.matmul(r_merged, w0_f32)
    else:
        rotated_w0 = torch.matmul(w0_f32, r_merged)
    del r_merged, w0_f32

    final_w = rotated_w0 + (lambda_ortho * avg_residual)

    if LOG.isEnabledFor(logging.DEBUG) and "vision" not in output_weight.name:
        LOG.debug(
            f"OrthoMerge [{output_weight.name}] Dim: {rot_dim} | "
            f"Rot-Norm: {torch.linalg.norm(rotated_w0):.2f} | "
            f"Res-Norm: {torch.linalg.norm(avg_residual):.2f}"
        )

    return final_w.to(dtype)
