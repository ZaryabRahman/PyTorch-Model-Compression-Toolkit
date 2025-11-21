"""pruning metrics and performance evaluation"""

from typing import Dict
from torch import nn

from pytorch_model_compression_toolkit.pruning.base import compute_sparsity


def model_sparsity(model: nn.Module, param_name: str = 'weight') -> float:
    total, zeros = 0, 0
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)) and hasattr(m, param_name):
            w = getattr(m, param_name)
            total += w.numel()
            zeros += (w == 0).sum().item()
    return zeros / total if total > 0 else 0.0


def print_sparsity_summary(model: nn.Module) -> None:
    print('--- pruning sparsity summary ---')
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)) and hasattr(m, 'weight'):
            w = m.weight
            sparsity = compute_sparsity(w)
            print(f'{m.__class__.__name__}: {sparsity*100:.2f}% zeros')
    total_sparsity = model_sparsity(model)
    print(f'Total model sparsity: {total_sparsity*100:.2f}%')
