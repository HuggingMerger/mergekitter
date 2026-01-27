import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, cast

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

LOG = logging.getLogger(__name__)

TensorMap = Dict[ModelReference, torch.Tensor]


class SWCMTask(Task[torch.Tensor]):
    """
    Implements the SWCM merging task with optimized memory handling
    and vectorized pre-computation.
    """

    gather_tensors: MergeTensorInput
    weight_info: WeightInfo
    base_model: Optional[ModelReference]
    max_iter: int
    tol: float
    density_strength: float
    regularization: float

    @override
    def uses_accelerator(self) -> bool:
        return True

    @override
    def arguments(self) -> Dict[str, Task]:
        return {"tensors": self.gather_tensors}

    @torch.inference_mode()
    @override
    def execute(self, tensors: TensorMap) -> torch.Tensor:
        """
        Executes the signal-weighted consensus merge.
        
        Optimizations:
        1. Pre-computes static norms outside the convergence loop.
        2. Uses in-place operations where possible to conserve VRAM.
        3. Enforces FP32 precision for stability during accumulation.
        """
        # Filter valid donor models
        donor_models: List[ModelReference] = [
            m for m in tensors.keys() 
            if m is not None and m != self.base_model
        ]

        if not donor_models:
            if self.base_model and self.base_model in tensors:
                return tensors[self.base_model]
            raise ValueError(f"No donor tensors found for {self.weight_info.name}")

        # Determine target dtype and initialize reference
        first_donor_t = tensors[donor_models[0]]
        target_dtype = first_donor_t.dtype
        device = first_donor_t.device
        
        # Construct Reference Tensor (Mean of Base + Donors, or just Donors)
        ref_t = torch.zeros_like(first_donor_t, dtype=torch.float32)
        count = 0

        if self.base_model and self.base_model in tensors:
            base_t = tensors[self.base_model].to(dtype=torch.float32)
            ref_t.add_(base_t)
            count = 1 
            pass 
        else:
            # Calculate mean of donors as reference
            for m in donor_models:
                t = tensors[m].to(dtype=torch.float32)
                # Ensure embedding sizes match before addition
                rectify_embed_sizes(self.weight_info, [first_donor_t, t])
                ref_t.add_(t)
            
            ref_t.div_(len(donor_models))

        # Compute Deltas and Pre-calculate Static Signal Norms
        deltas: List[torch.Tensor] = []
        delta_norms: List[float] = []
        
        # Initialize mu_delta 
        mu_delta = torch.zeros_like(ref_t)

        for m in donor_models:
            t_f32 = tensors[m].to(dtype=torch.float32)
            rectify_embed_sizes(self.weight_info, [ref_t, t_f32])
            
            # Calculate delta: T - Ref
            delta = t_f32.sub_(ref_t) 
            
            # Pre-compute norm for signal strength
            # norm() is slightly expensive, do it once.
            d_norm = torch.norm(delta).item()
            
            deltas.append(delta)
            delta_norms.append(d_norm)
            mu_delta.add_(delta)

            # Explicit cleanup hint
            del t_f32

        num_donors = len(donor_models)
        mu_delta.div_(num_donors)

        # Iterative Consensus Solver
        current_mean = mu_delta.clone()
        
        # Pre-allocate buffer for update vector to avoid malloc inside loop
        update_vec = torch.empty_like(current_mean)

        epsilon = 1e-6
        min_weight = 1e-9

        for _ in range(self.max_iter):
            update_vec.zero_()
            total_weight = 0.0
            
            # The layer scale changes as current_mean changes
            layer_scale = torch.norm(current_mean).item() + epsilon

            # Accumulate weighted updates
            for i in range(num_donors):
                delta = deltas[i]
                d_norm = delta_norms[i]

                # diff = delta - current_mean
                diff = delta - current_mean 
                dist = torch.norm(diff).item()

                signal = d_norm / layer_scale
                agreement = 1.0 / (1.0 + (dist / layer_scale))
                
                score = (signal ** self.density_strength) * agreement

                # update_vec += diff * score
                update_vec.add_(diff, alpha=score)
                total_weight += score

            if total_weight < min_weight:
                break

            # step = update_vec / total_weight
            step = update_vec.div_(total_weight)

            # Apply regularization: current_mean = current_mean * (1 - reg) + step
            current_mean.mul_(1.0 - self.regularization).add_(step)

            # Check convergence
            if torch.norm(step).item() < self.tol:
                break

        # result = ref_t + current_mean
        ref_t.add_(current_mean)

        return ref_t.to(dtype=target_dtype)


class SWCMMerge(MergeMethod):
    @override
    def name(self) -> str:
        return "SWCM"

    @override
    def pretty_name(self) -> str:
        return "SWCM (Signal‑Weighted Consensus Merge)"

    @override
    def parameters(self) -> List[ConfigParameterDef]:
        return [
            ConfigParameterDef(name="max_iter", default_value=5),
            ConfigParameterDef(name="tol", default_value=1e-4),
            ConfigParameterDef(name="density_strength", default_value=1.0),
            ConfigParameterDef(name="regularization", default_value=0.05),
        ]

    @override
    def make_task(
        self,
        *,
        output_weight: WeightInfo,
        tensors: MergeTensorInput,
        parameters: ImmutableMap[str, Any],
        tensor_parameters: ImmutableMap[ModelReference, ImmutableMap[str, Any]],
        base_model: Optional[ModelReference],
    ) -> Task:
        return SWCMTask(
            gather_tensors=tensors,
            weight_info=output_weight,
            base_model=base_model,
            max_iter=parameters["max_iter"] if "max_iter" in parameters else 5,
            tol=parameters["tol"] if "tol" in parameters else 1e-4,
            density_strength=parameters["density_strength"]
            if "density_strength" in parameters
            else 1.0,
            regularization=parameters["regularization"]
            if "regularization" in parameters
            else 0.05,
        )
