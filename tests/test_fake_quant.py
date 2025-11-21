import torch
from pytorch_model_compression_toolkit.quantization.fake_quant import FakeQuant, MinMaxObserver


def test_observer_updates():
    obs = MinMaxObserver(num_bits=8, symmetric=False)
    x = torch.tensor([-1.0, 0.5, 2.0])
    obs(x)
    assert obs.min_val <= -1.0 and obs.max_val >= 2.0


def test_fake_quant_forward():
    fq = FakeQuant(num_bits=8, symmetric=False)
    x = torch.randn(4, 4)
    out = fq(x)
    assert out.shape == x.shape


def test_qparams_change():
    fq = FakeQuant(num_bits=8, symmetric=False)
    x1 = torch.randn(16)
    x2 = torch.randn(16) * 5
    fq(x1)
    qmin1, qmax1 = fq.qmin, fq.qmax
    fq(x2)
    qmin2, qmax2 = fq.qmin, fq.qmax
    assert qmin1 == qmin2 and qmax1 == qmax2

