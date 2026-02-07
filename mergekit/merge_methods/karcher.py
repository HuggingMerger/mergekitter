# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only

from typing import Any, Dict, List, Optional

import torch
from typing_extensions import override

from mergekit.architecture import WeightInfo
from mergekit.common import ImmutableMap, ModelReference
from mergekit.graph import Task
from mergekit.merge_methods.base import (
    ConfigParameterDef,
    MergeMethod,
    MergeTensorInput,
)
from mergekit.merge_methods.rectify_embed import rectify_embed_sizes


class KarcherTask(Task[torch.Tensor]):
    """
    Task for merging model weights using the Riemannian (Karcher) mean algorithm.
    Few memory tweaks to be able to include more input models.
    """

    gather_tensors: MergeTensorInput
    weight_info: WeightInfo
    max_iter: int
    tol: float

    def uses_accelerator(self) -> bool:
        return True

    def arguments(self) -> Dict[str, Task]:
        return {"tensors": self.gather_tensors}

    def execute(self, tensors: Dict[ModelReference, torch.Tensor]) -> torch.Tensor:
        if len(tensors) == 1:
            return list(tensors.values())[0]

        # outline keys to allow selective deletion from the dict later
        model_keys = list(tensors.keys())
        first_tensor = tensors[model_keys[0]]
        device = first_tensor.device
        dtype = first_tensor.dtype

        if self.weight_info.is_embed:
            tensor_list = [tensors[k] for k in model_keys]
            rectify_embed_sizes(self.weight_info, tensor_list)
            for i, k in enumerate(model_keys):
                tensors[k] = tensor_list[i]
            del tensor_list

        num_models = len(model_keys)
        alphas = [1.0 / num_models] * num_models

        norms = []
        units = []

        for k in model_keys:
            t = tensors[k].to(device)

            t_float = t.to(torch.float32)
            n_val = torch.linalg.norm(t_float).item()

            if n_val <= 1e-12:
                norms.append(0.0)
                units.append(torch.zeros_like(t))
            else:
                norms.append(n_val)
                units.append(t.div_(n_val))

            # cleanup
            del tensors[k]
            del t
            del t_float

        merged_unit = karcher_mean_loop(
            units, alphas, max_iter=self.max_iter, tol=self.tol, device=device
        )

        # s = sum(alpha * norm)
        s = sum(a * n for a, n in zip(alphas, norms))

        return merged_unit.mul_(s).to(dtype)

    def group_label(self) -> Optional[str]:
        return self.gather_tensors.group_label()


def karcher_mean_loop(units, alphas, max_iter, tol, device):
    """
    Solves for the Karcher mean on the hypersphere.
    """

    valid_data = []
    for u, a in zip(units, alphas):
        if torch.any(u):
            valid_data.append((u, a))

    if not valid_data:
        return units[0]

    valid_units, valid_alphas = zip(*valid_data)

    sum_alpha = sum(valid_alphas)
    normalized_alphas = [a / sum_alpha for a in valid_alphas]

    u_mean = torch.zeros_like(valid_units[0], dtype=torch.float32)
    for a, vec in zip(normalized_alphas, valid_units):
        u_mean.add_(vec, alpha=a)

    norm_u = torch.linalg.norm(u_mean).item()
    if norm_u < tol:
        u_mean = valid_units[0].to(torch.float32)
    else:
        u_mean.div_(norm_u)

    for _ in range(max_iter):
        T = torch.zeros_like(u_mean)

        # T = sum(alpha * weight * ui) - (sum(alpha * weight * dot)) * u

        u_subtraction_scalar = 0.0
        accumulation_happened = False

        u_flat = u_mean.flatten()

        for a, vec in zip(normalized_alphas, valid_units):
            vec_float = vec.to(torch.float32)

            dot = torch.dot(u_flat, vec_float.flatten()).clamp(-1.0, 1.0)

            if dot > 1.0 - tol:
                continue

            theta = torch.arccos(dot)
            sin_theta = torch.sin(theta)

            if sin_theta < tol:
                continue

            weight = theta / sin_theta

            factor = a * weight

            T.add_(vec_float, alpha=factor)

            # track the u component scalar
            u_subtraction_scalar += factor * dot

            accumulation_happened = True

        if not accumulation_happened:
            break

        T.add_(u_mean, alpha=-u_subtraction_scalar)

        norm_T = torch.linalg.norm(T)
        if norm_T.item() < tol:
            break

        # u_new = u * cos(norm_T) + (T / norm_T) * sin(norm_T)
        cos_norm_T = torch.cos(norm_T)
        sin_norm_T = torch.sin(norm_T)

        # u_mean = u_mean * cos + T * (sin / norm)
        u_mean.mul_(cos_norm_T).add_(T, alpha=(sin_norm_T / norm_T))

        u_norm_final = torch.linalg.norm(u_mean)
        if u_norm_final.item() > tol:
            u_mean.div_(u_norm_final)

    return u_mean


class KarcherMerge(MergeMethod):
    """
    Implementation of the Karcher mean merge method.
    """

    def name(self) -> str:
        return "karcher"

    @override
    def pretty_name(self) -> Optional[str]:
        return "Karcher Mean"

    @override
    def reference_url(self) -> Optional[str]:
        return "https://en.wikipedia.org/wiki/Karcher_mean"

    def parameters(self) -> List[ConfigParameterDef]:
        return [
            ConfigParameterDef(name="max_iter", required=False, default_value=10),
            ConfigParameterDef(name="tol", required=False, default_value=1e-5),
        ]

    def make_task(
        self,
        *,
        output_weight: WeightInfo,
        tensors: MergeTensorInput,
        parameters: ImmutableMap[str, Any],
        tensor_parameters: ImmutableMap[ModelReference, ImmutableMap[str, Any]],
        base_model: Optional[ModelReference],
        **_kwargs,
    ) -> Task:
        max_iter = parameters["max_iter"] if "max_iter" in parameters else 10
        tol = parameters["tol"] if "tol" in parameters else 1e-5

        return KarcherTask(
            gather_tensors=tensors,
            weight_info=output_weight,
            max_iter=max_iter,
            tol=tol,
        )
