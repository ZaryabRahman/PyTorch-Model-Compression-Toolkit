import torch
from torch import nn
from pytorch_model_compression_toolkit.pruning.unstructured import UnstructuredPruner
from pytorch_model_compression_toolkit.pruning.structured import StructuredPruner
from pytorch_model_compression_toolkit.pruning.metrics import model_sparsity

def build_model():
    return nn.Sequential(
        nn.Conv2d(3, 16, 3, 1, 1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 3, 1, 1),
        nn.Linear(32*32*32, 10)
    )

def test_unstructured_pruning():
    model = build_model()
    pruner = UnstructuredPruner(model, amount=0.5)
    pruner.step()
    sparsity = pruner.compute_model_sparsity()
    assert 0 < sparsity <= 0.6

def test_structured_pruning():
    model = build_model()
    pruner = StructuredPruner(model, amount=0.5, dim=0)
    pruner.step()
    sparsity = pruner.compute_model_sparsity()
    assert 0 < sparsity <= 0.6

def test_global_sparsity_metric():
    model = build_model()
    pruner = UnstructuredPruner(model, amount=0.3)
    pruner.step()
    s = model_sparsity(model)
    assert 0 < s <= 0.35
