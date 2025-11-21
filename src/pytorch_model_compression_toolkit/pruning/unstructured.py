"""unstructured pruning routines"""

from typing import Dict
import torch
from torch import nn

from pytorch_model_compression_toolkit.pruning.base import apply_mask, prune_l1_unstructured, compute_sparsity


class UnstructuredPruner:
    """unstructured pruning manager for conv and linear layers"""

    def __init__(self, model: nn.Module, amount: float = 0.2, param_name: str = 'weight') -> None:
        self.model = model
        self.amount = amount
        self.param_name = param_name
        self.masks: Dict[nn.Module, torch.Tensor] = {}

    def step(self) -> None:
        """prune all eligible layers for current step"""
        for m in self.model.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)) and hasattr(m, self.param_name):
                mask = prune_l1_unstructured(m, self.amount, self.param_name)
                self.masks[m] = mask

    def compute_model_sparsity(self) -> float:
        total, zeros = 0, 0
        for m in self.model.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)) and hasattr(m, self.param_name):
                w = getattr(m, self.param_name)
                total += w.numel()
                zeros += (w == 0).sum().item()
        return zeros / total if total > 0 else 0.0
