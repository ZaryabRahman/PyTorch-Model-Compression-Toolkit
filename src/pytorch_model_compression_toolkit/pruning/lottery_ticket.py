"""Lottery Ticket Hypothesis utilities"""

from typing import Dict
import torch
from torch import nn

from pytorch_model_compression_toolkit.pruning.unstructured import UnstructuredPruner


class LotteryTicket:
    """find sparse subnetworks that train well from scratch"""

    def __init__(self, model: nn.Module, amount: float = 0.2):
        self.model = model
        self.amount = amount
        self.initial_state = {k: v.clone() for k, v in model.state_dict().items()}
        self.pruner = UnstructuredPruner(model, amount=amount)

    def prune_step(self) -> None:
        self.pruner.step()

    def reset_weights(self) -> None:
        self.model.load_state_dict(self.initial_state)

    def compute_sparsity(self) -> float:
        return self.pruner.compute_model_sparsity()
