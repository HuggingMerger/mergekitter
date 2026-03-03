import logging
  import torch
  from typing import List
  from mergekit.architecture import WeightInfo
  from mergekit.merge_methods.easy_define import merge_method
  
  LOG = logging.getLogger(__name__)
  
  def _compute_distribution_aware_importance(t: torch.Tensor, b: torch.Tensor, eps: float) -> torch.Tensor:
      """
      My attempt to implement https://arxiv.org/abs/2602.11717
      """
      delta = t - b
      mag = delta.abs().clamp(min=eps)
      return mag / (b.abs() + eps)
  
  @merge_method(
      name="scf_rkl",
      pretty_name="Sparse Complementary Fusion (RKL)",
      reference_url="https://arxiv.org/abs/2602.11717",
  )
  def scf_rkl_merge(
      tensors: List[torch.Tensor],
      base_tensor: torch.Tensor,
      output_weight: WeightInfo,
      sparsity_ratio: float = 0.9,
      epsilon: float = 1e-6,
      use_soft_fusion: bool = False,
  ) -> torch.Tensor:
      name = output_weight.name.lower()
      is_sensitive = any(x in name for x in ["norm", "embed_tokens", "lm_head", "bias"])
      current_sparsity = 0.0 if is_sensitive else sparsity_ratio
  
      if not tensors:
          return base_tensor
  
      importances = []
      deltas = []
      
      for t in tensors:
          delta = t - base_tensor
          imp = _compute_distribution_aware_importance(t, base_tensor, epsilon)
          deltas.append(delta)
          importances.append(imp)
  
      # If sparsity > 0, we mask.
      mask = None
      if current_sparsity > 0:
          all_imps = torch.cat([i.view(-1) for i in importances])
          # Use roughly top-k%
          k = int(all_imps.numel() * (1 - current_sparsity))
          if k > 0:
              top_val = torch.topk(all_imps, k, sorted=False)[0].min()
              threshold = top_val
          else:
              threshold = float('inf')
      else:
          threshold = float('-inf')
  
      merged_delta = torch.zeros_like(base_tensor)
      
      if use_soft_fusion:
          total_weight = torch.zeros_like(base_tensor)
          
          for delta, imp in zip(deltas, importances):
              valid_mask = imp >= threshold
  
              weight = torch.exp(imp - imp.max()) * valid_mask.float()
              
              merged_delta += delta * weight
              total_weight += weight
              
          merged_delta /= (total_weight + epsilon)
          merged_delta.nan_to_num_(0.0)
          
      else:
          # Hard Selection (Max Importance Winner)
          max_imp_grid = torch.full_like(base_tensor, float('-inf'))
          
          for delta, imp in zip(deltas, importances):
              valid_mask = imp >= threshold
              
              update_mask = valid_mask & (imp > max_imp_grid)
              
              merged_delta = torch.where(update_mask, delta, merged_delta)
              max_imp_grid = torch.where(update_mask, imp, max_imp_grid)
  
      return base_tensor + merged_delta
