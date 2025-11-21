"""structured pruning routines"""

from typing import Dict
import torch
from torch import nn

from pytorch_model_compression_toolkit.pruning.base import apply_mask, compute_sparsity


class StructuredPruner:
    """structured pruning by channel/filters"""

    def __init__(self, model: nn.Module, amount: float = 0.2, dim: int = 0, param_name: str = 'weight') -> None:
        self.model = model
        self.amount = amount
        self.param_name = param_name
        self.dim = dim  # dimension to prune along (0=out_channels)
        self.masks: Dict[nn.Module, torch.Tensor] = {}

    def step(self) -> None:
        for m in self.model.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)) and hasattr(m, self.param_name):
                w = getattr(m, self.param_name).data
                norm = w.norm(p=2, dim=self.dim, keepdim=False)
                k = int(norm.numel() * self.amount)
                if k == 0:
                    mask = torch.ones_like(norm)
                else:
                    threshold, _ = torch.kthvalue(norm, k)
                    mask = (norm > threshold).float()
                # expand mask to weight shape
                shape = [1]*w.dim()
                shape[self.dim] = -1
                mask_full = mask.view(*shape).expand_as(w)
                apply_mask(m, mask_full, self.param_name)
                self.masks[m] = mask_full

    def compute_model_sparsity(self) -> float:
        total, zeros = 0, 0
        for m in self.model.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)) and hasattr(m, self.param_name):
                w = getattr(m, self.param_name)
                total += w.numel()
                zeros += (w == 0).sum().item()
        return zeros / total if total > 0 else 0.0
