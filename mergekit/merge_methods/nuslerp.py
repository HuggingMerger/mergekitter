# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: BUSL-1.1

from typing import Any, Dict, List, Optional

import torch
from torch._tensor import Tensor
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


class NuSlerpTask(Task[torch.Tensor]):
    """Task for performing NuSLERP or ChipAlign merges between two model tensors."""
    gather_tensors: MergeTensorInput
    tensor_parameters: ImmutableMap[ModelReference, ImmutableMap[str, Any]]
    weight_info: WeightInfo
    row_wise: bool
    flatten: bool
    base_model: Optional[ModelReference]
    geodesic: bool  # Whether to use ChipAlign-style geodesic interpolation
    lambda_val: Optional[float]  # Interpolation factor for geodesic mode

    def uses_accelerator(self) -> bool:
        return True

    def arguments(self) -> Dict[str, Task]:
        return {"tensors": self.gather_tensors}

    def execute(self, tensors: Dict[ModelReference, torch.Tensor]) -> Tensor:
        # Fast path for single-model case
        if len(tensors) == 1:
            return list(tensors.values())[0]

        # Handle base model if provided
        if self.base_model is not None:
            if len(tensors) != 3:
                raise RuntimeError(
                    "NuSlerp base model can not be one of the two models to merge"
                )
            base_tensor = tensors.pop(self.base_model)
        else:
            base_tensor = None

        # Extract tensors and weights
        keys = list(tensors.keys())
        tensors_list = [tensors[key] for key in keys]
        weights = [self.tensor_parameters[key]["weight"] for key in keys]

        # Verify exactly two models are provided
        if len(tensors_list) != 2:
            raise RuntimeError(
                "NuSlerp merge expects exactly two models (plus optional base model)"
            )

        # Calculate interpolation factor from weights
        if abs(sum(weights)) < 1e-6:
            t = 0.5  # Default when weights sum to zero
        else:
            t = weights[1] / sum(weights)

        # Handle embedding tensors with different sizes
        if base_tensor is not None:
            tensors_list.append(base_tensor)
        rectify_embed_sizes(self.weight_info, tensors_list)

        # ChipAlign geodesic interpolation path
        if self.geodesic:
            if base_tensor is not None:
                raise ValueError("ChipAlign-style geodesic interpolation does not support a base model.")
            
            # Allow 'lambda' from parameters or fall back to calculated 't'
            interp_factor = self.lambda_val if self.lambda_val is not None else t
            
            # Extract the instruction and domain-specific tensors
            instruction_tensor = tensors_list[0]
            domain_tensor = tensors_list[1]
            
            # Calculate norms for magnitude preservation
            instruction_norm = torch.norm(instruction_tensor)
            domain_norm = torch.norm(domain_tensor)
            
            # nuslerp function:
            # v0_u = normalize(v0), v1_u = normalize(v1)
            # res = slerp(v0_u, v1_u)
            
            merged_tensor_unit = nuslerp(
                interp_factor,
                instruction_tensor,
                domain_tensor,
                dim=0 if self.row_wise else -1,
                flatten=self.flatten
            )
            
            # Apply magnitude scaling using weighted geometric mean (ChipAlign paper)
            # magnitude = (||v0||^(1-t) * ||v1||^t)
            magnitude = (instruction_norm ** (1 - interp_factor)) * (domain_norm ** interp_factor)
            
            # Re-scale
            merged_tensor = merged_tensor_unit * magnitude
            
            return merged_tensor
        
        # Standard NuSlerp path
        if base_tensor is not None:
            base_tensor = tensors_list.pop()
            # For task vector mode (with base model)
            return base_tensor + nuslerp(
                t,
                tensors_list[0] - base_tensor,
                tensors_list[1] - base_tensor,
                dim=0 if self.row_wise else -1,
                flatten=self.flatten,
            )
        
        # Direct tensor mode (no base model)
        return nuslerp(
            t,
            tensors_list[0],
            tensors_list[1],
            dim=0 if self.row_wise else -1,
            flatten=self.flatten,
        )


class NuSlerpMerge(MergeMethod):
    """Merge method implementing both NuSLERP and ChipAlign geodesic interpolation."""
    def name(self) -> str:
        return "nuslerp"

    @override
    def pretty_name(self):
        return "NuSLERP"

    @override
    def reference_url(self):
        return "https://arxiv.org/abs/2412.19819" if self.is_chipalign() else None
    
    def is_chipalign(self) -> bool:
        try:
            return self._parameters and self._parameters.get("geodesic", False)
        except AttributeError:
            return False

    def parameters(self) -> List[ConfigParameterDef]:
        return [
            ConfigParameterDef(
                name="nuslerp_row_wise",
                required=False,
                default_value=False,
                description="SLERP row vectors instead of column vectors",
            ),
            ConfigParameterDef(
                name="nuslerp_flatten",
                required=False,
                default_value=True,
                description="Treat tensors as flattened vectors",
            ),
            ConfigParameterDef(
                name="geodesic",
                required=False,
                default_value=False,
                description="Enable ChipAlign-style geodesic interpolation with magnitude preservation",
            ),
            ConfigParameterDef(
                name="lambda",
                required=False,
                default_value=None,
                description="Interpolation factor (0.0-1.0) for geodesic mode",
            ),
        ]

    def tensor_parameters(self) -> List[ConfigParameterDef]:
        return [ConfigParameterDef(name="weight", required=True)]

    def make_task(
        self,
        *,
        output_weight: WeightInfo,
        tensors: MergeTensorInput,
        base_model: Optional[ModelReference],
        parameters: ImmutableMap[str, Any],
        tensor_parameters: ImmutableMap[ModelReference, ImmutableMap[str, Any]],
        **_kwargs,
    ) -> Task:
        # Store parameters for reference_url to detect ChipAlign mode
        self._parameters = parameters
        
        return NuSlerpTask(
            gather_tensors=tensors,
            tensor_parameters=tensor_parameters,
            weight_info=output_weight,
            row_wise=parameters["nuslerp_row_wise"],
            flatten=parameters["nuslerp_flatten"],
            base_model=base_model,
            geodesic=parameters["geodesic"],
            lambda_val=parameters["lambda"],
        )


def nuslerp(
    t: float,
    v0: torch.Tensor,
    v1: torch.Tensor,
    dim: int = -1,
    eps: float = 1e-8,
    flatten: bool = False,
):
    """Enhanced spherical linear interpolation (SLERP) with flexible tensor handling."""
    out_shape = v0.shape

    def _normalize(x: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
        """Normalize tensor along last dimension with numeric stability."""
        return x / torch.norm(x, dim=-1, keepdim=True).clamp(min=eps)

    # Handle tensor reshaping based on interpolation mode
    if flatten:
        # Treat entire tensor as a single vector
        v0 = v0.view(-1)
        v1 = v1.view(-1)
    elif dim != -1:
        # Perform interpolation along specified dimension
        v0 = v0.transpose(dim, -1)
        v1 = v1.transpose(dim, -1)

    # Normalize to unit vectors
    v0_u = _normalize(v0)
    v1_u = _normalize(v1)

    # Calculate angle between vectors
    cos_theta = torch.sum(v0_u * v1_u, dim=-1, keepdim=True)
    theta = torch.acos(cos_theta.clamp(-1, 1))
    sin_theta = torch.sin(theta)

    # Handle (nearly) colinear vectors to avoid numerical issues
    colinear = (sin_theta.abs() < eps).squeeze()

    # SLERP formula: (sin((1-t)*θ)/sin(θ))*v0 + (sin(t*θ)/sin(θ))*v1
    res = (torch.sin((1 - t) * theta) * v0_u + torch.sin(t * theta) * v1_u) / sin_theta
    
    # Fall back to linear interpolation for numerically colinear vectors
    res[colinear] = (1 - t) * v0_u[colinear] + t * v1_u[colinear]

    # Restore original tensor shape
    if dim != -1 and not flatten:
        res = res.transpose(dim, -1)
    return res.view(out_shape)
