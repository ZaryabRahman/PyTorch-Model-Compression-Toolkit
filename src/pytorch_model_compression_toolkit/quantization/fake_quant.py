"""
Author: Zaryab Rahman
Data: 21-11-2025,   <<hostel room>> songgggggg i love it 
"""

"""fake quantization utilities: observers and fake-quant modules."""

from typing import Optional, Tuple
import math
import torch
from torch import nn
from torch.autograd import Function


def _clamp(x: torch.Tensor, min_val: float, max_val: float) -> torch.Tensor:
    return torch.max(torch.min(x, torch.tensor(max_val, device=x.device)), torch.tensor(min_val, device=x.device))


def _calc_zero_point(scale: float, min_val: float, max_val: float, qmin: int, qmax: int, signed: bool) -> int:
    if signed:
        zp = 0
    else:
        initial_zero_point = qmin - round(min_val / scale)
        zp = int(_clamp(torch.tensor(initial_zero_point), qmin, qmax).item())
    return zp


def calculate_qparams(min_val: torch.Tensor, max_val: torch.Tensor, num_bits: int = 8, symmetric: bool = False, per_channel: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """calculate scale and zero-point for quantization.

    min_val and max_val may be tensors for per-channel mode.
    returns (scale, zero_point) both as tensors.
    """
    qmin = 0
    qmax = (1 << num_bits) - 1
    signed = False
    if symmetric:
        max_abs = torch.max(min_val.abs(), max_val.abs())
        eps = 1e-8
        scale = max_abs / ((qmax - qmin) / 2.0 - eps)
        scale = torch.clamp(scale, min=eps)
        zero_point = torch.zeros_like(scale, dtype=torch.long)
        return scale, zero_point

    min_val = min_val.to(torch.float32)
    max_val = max_val.to(torch.float32)
    min_val = torch.min(min_val, torch.tensor(0.0, device=min_val.device))
    max_val = torch.max(max_val, torch.tensor(0.0, device=max_val.device))
    scale = (max_val - min_val) / float(qmax - qmin)
    scale = torch.clamp(scale, min=1e-8)
    zero_point = (qmin - (min_val / scale)).round().to(torch.long)
    zero_point = torch.clamp(zero_point, qmin, qmax)
    return scale, zero_point


class ObserverBase(nn.Module):
    """base observer tracking statistics required for quantization."""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def get_qparams(self) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class MinMaxObserver(ObserverBase):
    """min-max observer; records min and max values seen."""

    def __init__(self, per_channel: bool = False, channel_axis: int = 0):
        super().__init__()
        self.per_channel = per_channel
        self.channel_axis = channel_axis
        self.register_buffer("min_val", torch.tensor(float("inf")))
        self.register_buffer("max_val", torch.tensor(float("-inf")))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.per_channel:
            dims = list(range(x.dim()))
            dims.pop(self.channel_axis)
            cur_min = x.amin(dim=dims)
            cur_max = x.amax(dim=dims)
        else:
            cur_min = x.min()
            cur_max = x.max()
        self.min_val = torch.min(self.min_val, cur_min)
        self.max_val = torch.max(self.max_val, cur_max)
        return x

    def get_qparams(self, num_bits: int = 8, symmetric: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        return calculate_qparams(self.min_val, self.max_val, num_bits=num_bits, symmetric=symmetric, per_channel=self.per_channel)


class MovingAverageMinMaxObserver(ObserverBase):
    """moving-average min-max observer with momentum."""

    def __init__(self, momentum: float = 0.1, per_channel: bool = False, channel_axis: int = 0):
        super().__init__()
        self.momentum = float(momentum)
        self.per_channel = per_channel
        self.channel_axis = channel_axis
        self.register_buffer("min_val", torch.tensor(float("inf")))
        self.register_buffer("max_val", torch.tensor(float("-inf")))
        self.initialized = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.per_channel:
            dims = list(range(x.dim()))
            dims.pop(self.channel_axis)
            cur_min = x.amin(dim=dims)
            cur_max = x.amax(dim=dims)
        else:
            cur_min = x.min()
            cur_max = x.max()
        if not self.initialized:
            self.min_val = cur_min.detach().clone()
            self.max_val = cur_max.detach().clone()
            self.initialized = True
        else:
            self.min_val = (1 - self.momentum) * self.min_val + self.momentum * cur_min.detach()
            self.max_val = (1 - self.momentum) * self.max_val + self.momentum * cur_max.detach()
        return x

    def get_qparams(self, num_bits: int = 8, symmetric: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        return calculate_qparams(self.min_val, self.max_val, num_bits=num_bits, symmetric=symmetric, per_channel=self.per_channel)


class _FakeQuantizeFn(Function):
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, scale: torch.Tensor, zero_point: Optional[torch.Tensor], num_bits: int, axis: Optional[int], signed: bool):
        ctx.save_for_backward(scale, zero_point if zero_point is not None else torch.tensor(0, device=inputs.device))
        qmin = 0
        qmax = (1 << num_bits) - 1
        if axis is None:
            scale_t = scale
            zero_t = zero_point if zero_point is not None else torch.tensor(0, device=inputs.device)
            inv_scale = 1.0 / scale_t
            transformed = torch.round(torch.clamp(torch.round(inputs * inv_scale) + zero_t, qmin, qmax))
            dequant = (transformed - zero_t) * scale_t
            return dequant
        else:
            shape = [1] * inputs.dim()
            shape[axis] = -1
            scale_shaped = scale.view(*shape)
            if zero_point is not None:
                zp_shaped = zero_point.view(*shape).to(inputs.device)
            else:
                zp_shaped = torch.zeros_like(scale_shaped, dtype=torch.long)
            inv_scale = 1.0 / scale_shaped
            transformed = torch.round(torch.clamp(torch.round(inputs * inv_scale) + zp_shaped, qmin, qmax))
            dequant = (transformed - zp_shaped) * scale_shaped
            return dequant

    @staticmethod
    def backward(ctx, grad_output):
        scale, zero_point = ctx.saved_tensors
        grad_input = grad_output.clone()
        return grad_input, None, None, None, None, None


class FakeQuantize(nn.Module):
    """module that applies fake quantization with observer integration."""

    def __init__(self, observer: ObserverBase, num_bits: int = 8, per_channel: bool = False, channel_axis: int = 0, symmetric: bool = False):
        super().__init__()
        self.observer = observer
        self.num_bits = num_bits
        self.per_channel = per_channel
        self.channel_axis = channel_axis if per_channel else None
        self.symmetric = symmetric
        self.enabled = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.enabled:
            self.observer(x.detach())
            return x
        scale, zero_point = self.observer.get_qparams(num_bits=self.num_bits, symmetric=self.symmetric)
        if not self.per_channel:
            scale = scale.to(x.device)
            zp = zero_point.to(x.device) if isinstance(zero_point, torch.Tensor) else torch.tensor(0, device=x.device)
            return _FakeQuantizeFn.apply(x, scale, zp, self.num_bits, None, not self.symmetric)
        else:
            scale = scale.to(x.device)
            zp = zero_point.to(x.device) if isinstance(zero_point, torch.Tensor) else None
            return _FakeQuantizeFn.apply(x, scale, zp, self.num_bits, self.channel_axis, not self.symmetric)


def prepare_qat(module: nn.Module, observer_ctor=MinMaxObserver, num_bits: int = 8, per_channel: bool = False, channel_axis: int = 0, symmetric: bool = False) -> nn.Module:
    """insert fake-quant modules in place of modules that should be quantized."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Conv2d) or isinstance(child, nn.Linear):
            weight_obs = observer_ctor(per_channel=per_channel, channel_axis=channel_axis) if per_channel else observer_ctor(per_channel=False)
            fq_w = FakeQuantize(weight_obs, num_bits=num_bits, per_channel=per_channel, channel_axis=channel_axis, symmetric=symmetric)
            setattr(module, name + "_fakequant_weight", fq_w)
            setattr(module, name, child)
        else:
            prepare_qat(child, observer_ctor=observer_ctor, num_bits=num_bits, per_channel=per_channel, channel_axis=channel_axis, symmetric=symmetric)
    return module


def convert_qat(module: nn.Module) -> nn.Module:
    """convert modules with fake-quant attached to quantized aware equivalents (weights folded)."""
    for name, child in list(module.named_children()):
        if name.endswith("_fakequant_weight") and isinstance(child, FakeQuantize):
            parent_name = name[: -len("_fakequant_weight")]
            parent = getattr(module, parent_name)
            with torch.no_grad():
                w = parent.weight
                scale, zero_point = child.observer.get_qparams(num_bits=child.num_bits, symmetric=child.symmetric)
                if child.per_channel:
                    shape = [1] * w.dim()
                    shape[child.channel_axis] = -1
                    scale_shaped = scale.view(*shape).to(w.device)
                    if isinstance(zero_point, torch.Tensor):
                        zp_shaped = zero_point.view(*shape).to(w.device)
                    else:
                        zp_shaped = torch.zeros_like(scale_shaped, dtype=torch.long)
                    q = torch.round(torch.clamp(torch.round(w / scale_shaped) + zp_shaped, 0, (1 << child.num_bits) - 1))
                    q = (q - zp_shaped) * scale_shaped
                else:
                    s = scale.to(w.device)
                    zp = zero_point.to(w.device) if isinstance(zero_point, torch.Tensor) else torch.tensor(0, device=w.device)
                    q = torch.round(torch.clamp(torch.round(w / s) + zp, 0, (1 << child.num_bits) - 1))
                    q = (q - zp) * s
                parent.weight.data.copy_(q)
            delattr(module, name)
        else:
            convert_qat(child)
    return module


if __name__ == "__main__":
    x = torch.randn(4, 3, 32, 32)
    obs = MovingAverageMinMaxObserver(momentum=0.1, per_channel=False)
    fq = FakeQuantize(obs, num_bits=8, per_channel=False, symmetric=False)
    fq.train()
    for _ in range(10):
        _ = fq(x)
    fq.eval()
    y = fq(x)
    print("input mean", x.mean().item())
    print("fake-quant mean", y.mean().item())
