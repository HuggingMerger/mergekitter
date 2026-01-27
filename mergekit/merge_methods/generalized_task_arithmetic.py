## File: ./MergekitGemini/mergekit/merge_methods/generalized_task_arithmetic.py

# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: BUSL-1.1

import logging
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import torch
from pydantic import BaseModel
from typing_extensions import Literal, override

from mergekit.architecture import WeightInfo
from mergekit.common import ImmutableMap, ModelReference
from mergekit.graph import Task
from mergekit.merge_methods.base import (
    ConfigParameterDef,
    MergeMethod,
    MergeTensorInput,
)
from mergekit.sparsify import RescaleNorm, SparsificationMethod, sparsify
from mergekit.subspace_helpers import iso_c, compute_and_sum_svd_mem_reduction, subspace_boosting
from mergekit.merge_methods.magic import magic_magnitude_calibration  # Import MAGIC

class ConsensusMethod(str, Enum):
    count = "count"
    sum = "sum"


class GeneralizedTaskArithmeticMerge(MergeMethod, BaseModel, frozen=True):
    consensus_method: Optional[ConsensusMethod]
    sparsification_method: Optional[SparsificationMethod]
    default_normalize: bool
    default_rescale: bool
    default_swapping: bool = False
    method_name: str
    method_pretty_name: Optional[str]
    method_reference_url: Optional[str]

    def name(self) -> str:
        return self.method_name

    @override
    def pretty_name(self) -> Optional[str]:
        return self.method_pretty_name

    @override
    def reference_url(self) -> Optional[str]:
        return self.method_reference_url

    def parameters(self) -> List[ConfigParameterDef]:
        return [
            ConfigParameterDef(name="int8_mask", required=False, default_value=False),
            ConfigParameterDef(
                name="normalize", required=False, default_value=self.default_normalize
            ),
            ConfigParameterDef(
                name="rescale", required=False, default_value=self.default_rescale
            ),
            # Swapping Parameters
            ConfigParameterDef(name="swapping", required=False, default_value=self.default_swapping),
            ConfigParameterDef(name="lambda", required=False, default_value=1.0),
            # Subspace Boosting Parameters
            ConfigParameterDef(name="svd_thresh", required=False, default_value=0.01),
            ConfigParameterDef(name="cumsum", required=False, default_value=True),
            # MAGIC Parameters
            ConfigParameterDef(name="use_magic", required=False, default_value=False),
            ConfigParameterDef(name="magic_scaling_factor", required=False, default_value=1.0),
        ]

    def tensor_parameters(self) -> List[ConfigParameterDef]:
        res = [
            ConfigParameterDef(name="weight", required=True),
            ConfigParameterDef(name="density", required=False, default_value=1.0),
            ConfigParameterDef(name="diagonal_offset", required=False),
            ConfigParameterDef(name="invert_offset", required=False, default_value=False),
            ConfigParameterDef(name="random_mask", required=False, default_value=0.0),
            ConfigParameterDef(name="random_mask_seed", required=False, default_value=None),
        ]
        if self.sparsification_method == SparsificationMethod.magnitude_outliers:
            res.append(ConfigParameterDef(name="gamma", default_value=0.01))
        if self.sparsification_method == SparsificationMethod.della_magprune:
            res.append(ConfigParameterDef(name="epsilon", default_value=0.15))
        return res

    def make_task(
        self,
        output_weight: WeightInfo,
        tensors: MergeTensorInput,
        base_model: Optional[ModelReference],
        parameters: ImmutableMap[str, Any],
        tensor_parameters: ImmutableMap[ModelReference, ImmutableMap[str, Any]],
    ) -> Task:
        return GTATask(
            method=self,
            tensors=tensors,
            base_model=base_model,
            tensor_parameters=tensor_parameters,
            int8_mask=parameters["int8_mask"],
            normalize=parameters["normalize"],
            lambda_=parameters["lambda"],
            rescale_norm=RescaleNorm.l1 if parameters["rescale"] else None,
            swapping=parameters["swapping"],
            weight_info=output_weight,
            svd_thresh=parameters["svd_thresh"],
            cumsum=parameters["cumsum"],
            use_magic=parameters["use_magic"],
            magic_scaling_factor=parameters["magic_scaling_factor"],
        )


class GTATask(Task[torch.Tensor]):
    method: GeneralizedTaskArithmeticMerge
    tensors: MergeTensorInput
    base_model: ModelReference
    weight_info: WeightInfo
    tensor_parameters: ImmutableMap[ModelReference, Any]
    int8_mask: bool
    normalize: bool
    lambda_: float
    rescale_norm: Optional[RescaleNorm]
    swapping: bool
    svd_thresh: float
    cumsum: bool
    use_magic: bool
    magic_scaling_factor: float

    def uses_accelerator(self) -> bool:
        return True

    def arguments(self) -> Dict[str, Task]:
        return {"tensors": self.tensors}

    def execute(
        self,
        tensors: Dict[ModelReference, torch.Tensor],
        **_kwargs,
    ) -> torch.Tensor:
        # collect task vectors
        tvs, base = get_task_vectors(
            self.weight_info,
            self.base_model,
            tensors,
            tensor_parameters=self.tensor_parameters.data,
            swapping=self.swapping,
        )
        if not tvs:
            return base

        # MAGIC 2512.19320
        original_norms = None
        if self.use_magic:
            original_norms = torch.stack(
                [tv["delta"].to(torch.float32).norm() for tv in tvs]
            )

        # sparsify
        if self.method.sparsification_method:
            for tv_info in tvs:
                kwargs = {}
                if "gamma" in tv_info:
                    kwargs["gamma"] = tv_info["gamma"]

                if "epsilon" in tv_info:
                    kwargs["epsilon"] = tv_info["epsilon"]

                tv_info["delta"] = sparsify(
                    tv_info["delta"],
                    density=tv_info["density"],
                    method=self.method.sparsification_method,
                    rescale_norm=self.rescale_norm,
                    **kwargs,
                )
        deltas = torch.stack([tv["delta"] for tv in tvs], dim=0)

        weights = torch.tensor(
            [tv["weight"] for tv in tvs], dtype=deltas.dtype, device=deltas.device
        )
        while len(deltas.shape) > len(weights.shape):
            weights.unsqueeze_(-1)

        weighted_deltas = deltas * weights

        # get sign consensus and mix deltas
        if self.method.consensus_method:
            mask_dtype = torch.int8 if self.int8_mask else base.dtype
            mask = get_mask(
                weighted_deltas,
                method=self.method.consensus_method,
                mask_dtype=mask_dtype,
            )
            mixed_delta = (weighted_deltas * mask).sum(dim=0)
            divisor = (weights * mask).sum(dim=0)
            divisor[divisor == 0] = 1
        else:
            mixed_delta = weighted_deltas.sum(dim=0)
            divisor = weights.sum(dim=0)
            divisor[divisor.abs() < 1e-8] = 1

        param_key = self.weight_info.name
        subspace_input = [tv["delta"] for tv in tvs]

        if self.method.name() == "iso_c":
            mixed_delta = iso_c(subspace_input, param_key, deltas.device)
        elif self.method.name() == "tsvm":
            mixed_delta = compute_and_sum_svd_mem_reduction(subspace_input, param_key, deltas.device)
        elif self.method.name() in ["task_arithmetic_sb", "ties_sb"]:
            mixed_delta = subspace_boosting(param_key, mixed_delta, svd_thresh=self.svd_thresh, cumsum=self.cumsum)

        if self.normalize:
            mixed_delta /= divisor

        # Apply MAGIC calibration
        if self.use_magic and original_norms is not None:
             mixed_delta = magic_magnitude_calibration(
                 mixed_delta,
                 original_norms,
                 scaling_factor=self.magic_scaling_factor
             )

        if self.lambda_ != 1:
            mixed_delta *= self.lambda_

        return (base + mixed_delta).to(base.dtype)

    def group_label(self) -> Optional[str]:
        return self.tensors.group_label()


def swapping_method(base, x, parameters):
    """
    Applies task swapping logic to input tensor x based on base model.
    """
    def swap_values(shape, n, base, x):
        if x.dim() == 2:
           rows, cols = shape
           rows_range = torch.arange(rows).view(-1, 1)
           cols_range = torch.arange(cols).view(1, -1)
           mask = ((rows_range + cols_range) % n == 0).to(base.device).bool()
           x = torch.where(mask, x, base)
        else:
           rows_range = torch.arange(shape[0])
           mask = ((rows_range) % n == 0).to(base.device).bool()
           x = torch.where(mask, x, base)
        return x

    def rand_mask(base, x, percent, seed=None):
        oldseed = torch.seed()
        if seed is not None:
            torch.manual_seed(seed)
        random = torch.rand(base.shape, device=base.device)
        mask = (random <= percent).bool()
        del random
        torch.manual_seed(oldseed)
        x = torch.where(mask, x, base)
        return x

    bt = base.dtype
    if x.device.type == "cpu":
        x = x.to(torch.float32)
        base = base.to(torch.float32)

    diagonal_offset = parameters.get('diagonal_offset')
    random_mask = parameters.get('random_mask', 0.0)
    random_mask_seed = parameters.get('random_mask_seed')
    random_mask_seed = int(random_mask_seed) if random_mask_seed is not None else None

    # Logic branch: Random Mask or Diagonal Offset
    if random_mask != 0.0:
       if random_mask is None or random_mask >= 1.0 or random_mask <= 0.0:
           raise ValueError("The random_mask parameter must be a number strictly between 0 and 1.")
       if random_mask_seed is not None and (not isinstance(random_mask_seed, int)):
           raise TypeError("The random_mask_seed parameter must be None or an integer.")

       x = rand_mask(base, x, random_mask, random_mask_seed)

    elif diagonal_offset is not None:
        if (diagonal_offset % 1 != 0) or (diagonal_offset < 2):
            raise ValueError("The diagonal_offset must be an integer greater than or equal to 2.")

        diagonal_offset = int(diagonal_offset)
        if parameters.get('invert_offset') == False:
            x = swap_values(x.shape, diagonal_offset, base, x)
        else:
            x = swap_values(x.shape, diagonal_offset, x, base)

    return x.to(bt)


def get_task_vectors(
    weight_info: WeightInfo,
    base_model: ModelReference,
    tensors: ImmutableMap[ModelReference, torch.Tensor],
    tensor_parameters: ImmutableMap[ModelReference, ImmutableMap[str, Any]],
    swapping: bool,
) -> Tuple[List[Dict[str, Any]], torch.Tensor]:
    keys = list(tensors.keys())
    base = tensors[base_model]

    parameter_name = weight_info.name

    res = []
    for model in keys:
        if model == base_model:
            continue

        x = tensors[model].to(base.dtype)

        # Swapping logic applied before size check or delta calculation
        if swapping:
            if model in tensor_parameters:
                params = dict(tensor_parameters[model].items())
                x = swapping_method(base, x, params)

        if x.shape != base.shape:
            if weight_info.is_embed:
                x = x[: base.shape[0], : base.shape[1]]
                logging.warning(f"Using submatrix of {model}:{parameter_name}")
            else:
                logging.warning(
                    f"skipping {model}:{parameter_name} due to size mismatch"
                )
                continue

        delta = x - base
        del x
        del tensors[model]

        d = {}
        d["model"] = model
        d["delta"] = delta
        for p in tensor_parameters[model]:
            d[p] = tensor_parameters[model][p]
        res.append(d)
    return res, base


def get_mask(
    delta: torch.Tensor,
    method: Literal["sum", "count"] = "sum",
    mask_dtype: Optional[torch.dtype] = None,
):
    """Returns a mask determining which delta vectors should be merged
    into the final model."""
    if mask_dtype is None:
        mask_dtype = delta.dtype

    sign = delta.sign().to(mask_dtype)

    if method == "sum":
        sign_weight = delta.sum(dim=0)
        majority_sign = (sign_weight >= 0).to(mask_dtype) * 2 - 1
        del sign_weight
    elif method == "count":
        majority_sign = (sign.sum(dim=0) >= 0).to(mask_dtype) * 2 - 1
    else:
        raise RuntimeError(f'Unimplemented mask method "{method}"')

    return sign == majority_sign
