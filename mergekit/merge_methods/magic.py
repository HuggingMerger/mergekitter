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
    input_norms: torch.Tensor,
    scaling_factor: float = 1.0,
) -> torch.Tensor:
    """
    Implements Weight Space Calibration (WSC) from MAGIC (arXiv:2512.19320).
    Rescales the merged task vector to match the average magnitude of original task vectors.
    """
    # calc target norm rho * (1/N * sum(||tau_i||))
    target_norm = input_norms.mean() * scaling_factor
    
    # calc current norm of the merged vector
    current_norm = merged_tv.to(torch.float32).norm()

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

        # grab other models
        model_keys = [k for k in tensors.keys() if k != self.base_model]
        if not model_keys:
            return base_tensor

        if self.weight_info.is_embed:
            # prep list
            all_tensors_ordered = [base_tensor] + [tensors[k] for k in model_keys]
            rectify_embed_sizes(self.weight_info, all_tensors_ordered)
            
            # refetch resized base
            base_tensor = all_tensors_ordered[0]
            for i, k in enumerate(model_keys):
                tensors[k] = all_tensors_ordered[i+1]
        
        # gather summed tv
        merged_delta = torch.zeros_like(base_tensor, dtype=base_dtype, device=base_device)
        
        # store norms of individual tvs for magic calc
        norms_list = []

        for key in model_keys:
            # dtype
            model_tensor = tensors[key].to(base_dtype).to(base_device)
            
            params = self.tensor_parameters[key]
            weight = params["weight"] if "weight" in params else 1.0
            
            # calc tv (tau_i)
            # delta = (fine_tuned - base) * weight
            delta = model_tensor.sub_(base_tensor).mul_(weight)

            # capture norm
            norms_list.append(delta.to(torch.float32).norm())

            # add to merged delta sum
            merged_delta.add_(delta)

            # cleanup
            del delta
            del model_tensor
            del tensors[key]

        if not norms_list:
            return base_tensor

        # Stack norms 
        input_norms = torch.stack(norms_list)

        calibrated_delta = magic_magnitude_calibration(
            merged_delta, input_norms, self.scaling_factor
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
