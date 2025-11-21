"""
Author: Zaryab Rahman
Date: 21-11-2025
"""

"""base pruning utilities and common functions."""

from typing import Optional, Dict
import torch
from torch import nn


def apply_mask(module: nn.Module, mask: torch.Tensor, param_name: str = 'weight') -> None:
    """apply binary mask to module parameter."""
    if hasattr(module, param_name):
        param = getattr(module, param_name)
        param.data.mul_(mask)


def compute_sparsity(tensor: torch.Tensor) -> float:
    """return fraction of zero weights."""
    return float((tensor == 0).sum()) / tensor.numel()


def prune_l1_unstructured(module: nn.Module, amount: float = 0.2, param_name: str = 'weight') -> torch.Tensor:
    """prune smallest magnitude weights by L1 norm (unstructured). returns mask."""
    if not hasattr(module, param_name):
        raise ValueError(f'module has no parameter {param_name}')
    weight = getattr(module, param_name).data
    k = int(weight.numel() * amount)
    if k == 0:
        return torch.ones_like(weight)
    flattened = weight.abs().view(-1)
    threshold, _ = torch.kthvalue(flattened, k)
    mask = (weight.abs() > threshold).float()
    apply_mask(module, mask, param_name)
    return mask


def global_prune(model: nn.Module, amount: float = 0.2, param_name: str = 'weight') -> Dict[str, torch.Tensor]:
    """global unstructured pruning across all conv/linear layers."""
    tensors = []
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)) and hasattr(m, param_name):
            tensors.append(m.weight.data.abs().flatten())
    all_weights = torch.cat(tensors)
    k = int(all_weights.numel() * amount)
    if k == 0:
        return {m: torch.ones_like(m.weight) for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))}
    threshold, _ = torch.kthvalue(all_weights, k)
    masks = {}
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)) and hasattr(m, param_name):
            mask = (m.weight.data.abs() > threshold).float()
            apply_mask(m, mask, param_name)
            masks[m] = mask
    return masks
