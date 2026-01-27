# Copyright (C) 2026 Arcee AI
# SPDX-License-Identifier: BUSL-1.1

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


def magic_magnitude_calibration(
    merged_tv: torch.Tensor,
    original_tvs_or_norms: List[torch.Tensor] | torch.Tensor,
    scaling_factor: float = 1.0,
) -> torch.Tensor:
    """
    Implements Weight Space Calibration (WSC) from MAGIC (arXiv:2512.19320).
    Rescales the merged task vector to match the average magnitude of original task vectors.

    Args:
        merged_tv: The (merged) task vector to calibrate.
        original_tvs_or_norms: Either a list of original task vectors (deltas) OR
                               a precomputed tensor of their norms.
        scaling_factor: Multiplier for the target magnitude (rho in the paper).
    """
    if isinstance(original_tvs_or_norms, list):
        if not original_tvs_or_norms:
            return merged_tv
        # Compute norms of original vectors
        # Use float32 for stability
        orig_norms = torch.stack(
            [tv.to(torch.float32).norm() for tv in original_tvs_or_norms]
        )
    elif isinstance(original_tvs_or_norms, torch.Tensor):
        orig_norms = original_tvs_or_norms.to(torch.float32)
    else:
        raise ValueError("Invalid input for original_tvs_or_norms")

    target_norm = orig_norms.mean() * scaling_factor
    current_norm = merged_tv.to(torch.float32).norm()

    # Avoid division by zero
    if current_norm < 1e-6:
        return merged_tv

    gamma = target_norm / current_norm
    calibrated_tv = merged_tv * gamma

    return calibrated_tv.to(merged_tv.dtype)


class MagicTask(Task[torch.Tensor]):
    gather_tensors: MergeTensorInput
    base_model: ModelReference
    weight_info: WeightInfo
    tensor_parameters: ImmutableMap[ModelReference, ImmutableMap[str, Any]]
    scaling_factor: float

    def uses_accelerator(self) -> bool:
        return True

    def arguments(self) -> Dict[str, Task]:
        return {"tensors": self.gather_tensors}

    def execute(self, tensors: Dict[ModelReference, torch.Tensor]) -> torch.Tensor:
        if self.base_model not in tensors:
            raise RuntimeError("Base model tensor not found for MAGIC merge")

        base_tensor = tensors[self.base_model]
        base_device = base_tensor.device
        base_dtype = base_tensor.dtype

        # 1. Compute Task Vectors
        task_vectors = []
        model_keys = [k for k in tensors.keys() if k != self.base_model]
        
        # Prepare list for rectify to ensure sizes match
        all_tensors_for_rectify = [base_tensor] + [tensors[k] for k in model_keys]
        rectify_embed_sizes(self.weight_info, all_tensors_for_rectify)
        
        # Re-fetch potentially resized base
        base_tensor = all_tensors_for_rectify[0]

        for i, key in enumerate(model_keys):
            model_tensor = all_tensors_for_rectify[i+1].to(base_dtype).to(base_device)
            weight = self.tensor_parameters[key].get("weight", 1.0)
            
            delta = (model_tensor - base_tensor) * weight
            task_vectors.append(delta)

        if not task_vectors:
            return base_tensor

        # 2. Basic Merge (Sum/Task Arithmetic)
        # MAGIC paper primarily applies it to the sum of task vectors
        merged_delta = torch.sum(torch.stack(task_vectors), dim=0)

        # 3. MAGIC Calibration
        calibrated_delta = magic_magnitude_calibration(
            merged_delta, task_vectors, self.scaling_factor
        )

        return base_tensor + calibrated_delta

    def group_label(self) -> Optional[str]:
        return self.gather_tensors.group_label()


class MagicMerge(MergeMethod):
    def name(self) -> str:
        return "magic"

    @override
    def pretty_name(self) -> Optional[str]:
        return "MAGIC (Magnitude Calibration)"

    @override
    def reference_url(self) -> Optional[str]:
        return "https://arxiv.org/abs/2512.19320"

    def parameters(self) -> List[ConfigParameterDef]:
        return [
            ConfigParameterDef(name="scaling_factor", required=False, default_value=1.0)
        ]

    def tensor_parameters(self) -> List[ConfigParameterDef]:
        return [ConfigParameterDef(name="weight", required=False, default_value=1.0)]

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
        return MagicTask(
            gather_tensors=tensors,
            base_model=base_model,
            weight_info=output_weight,
            tensor_parameters=tensor_parameters,
            scaling_factor=parameters["scaling_factor"],
        )
