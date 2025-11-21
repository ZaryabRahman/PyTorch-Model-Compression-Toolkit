"""
Author: Zaryab Rahman
Data: 21-11-2025, <<hostel room> with haroon aziz
"""


"""qat training utilities and trainer harness."""

from typing import Optional, Dict, Any, Callable
import time
import os
import math
import json

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from pytorch_model_compression_toolkit.quantization.fake_quant import prepare_qat, convert_qat, ObserverBase


def _to_device(batch, device):
    if isinstance(batch, (list, tuple)):
        return [b.to(device) if hasattr(b, 'to') else b for b in batch]
    if isinstance(batch, dict):
        return {k: v.to(device) if hasattr(v, 'to') else v for k, v in batch.items()}
    return batch.to(device)


class QATTrainer:
    """trainer for quantization-aware training supporting amp, checkpointing and hooks."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        device: Optional[torch.device] = None,
        scaler: Optional[GradScaler] = None,
        max_epochs: int = 100,
        lr_scheduler: Optional[Any] = None,
        grad_accum_steps: int = 1,
        clip_grad_norm: Optional[float] = None,
        log_interval: int = 50,
        save_dir: Optional[str] = None,
        callbacks: Optional[Dict[str, Callable]] = None,
    ) -> None:
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scaler = scaler if scaler is not None else GradScaler()
        self.max_epochs = max_epochs
        self.lr_scheduler = lr_scheduler
        self.grad_accum_steps = max(1, grad_accum_steps)
        self.clip_grad_norm = clip_grad_norm
        self.log_interval = log_interval
        self.save_dir = save_dir
        self.callbacks = callbacks or {}
        self.start_epoch = 0

    def _save_checkpoint(self, epoch: int, extra: Optional[Dict[str, Any]] = None) -> None:
        if not self.save_dir:
            return
        os.makedirs(self.save_dir, exist_ok=True)
        path = os.path.join(self.save_dir, f"checkpoint_epoch_{epoch}.pt")
        payload = {
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scaler_state': self.scaler.state_dict() if self.scaler is not None else None,
        }
        if extra:
            payload.update(extra)
        torch.save(payload, path)

    def _load_checkpoint(self, path: str) -> int:
        ck = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ck['model_state'])
        self.optimizer.load_state_dict(ck['optimizer_state'])
        if self.scaler is not None and ck.get('scaler_state') is not None:
            self.scaler.load_state_dict(ck['scaler_state'])
        return int(ck.get('epoch', 0))

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: Optional[int] = None,
        start_epoch: int = 0,
        checkpoint_path: Optional[str] = None,
    ) -> None:
        epochs = epochs or self.max_epochs
        if checkpoint_path:
            self.start_epoch = self._load_checkpoint(checkpoint_path)
        else:
            self.start_epoch = start_epoch

        self.model.train()
        for epoch in range(self.start_epoch, epochs):
            t0 = time.time()
            running_loss = 0.0
            num_samples = 0
            self.model.train()
            for batch_idx, batch in enumerate(train_loader):
                batch = _to_device(batch, self.device)
                inputs, targets = batch if not isinstance(batch, dict) else (batch['input'], batch['target'])
                with autocast():
                    outputs = self.model(inputs)
                    loss = self.loss_fn(outputs, targets)
                    loss = loss / self.grad_accum_steps
                self.scaler.scale(loss).backward()
                if (batch_idx + 1) % self.grad_accum_steps == 0:
                    if self.clip_grad_norm is not None:
                        self.scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()
                running_loss += loss.item() * self.grad_accum_steps
                num_samples += 1
                if batch_idx % self.log_interval == 0:
                    if 'on_batch' in self.callbacks:
                        try:
                            self.callbacks['on_batch'](epoch, batch_idx, loss.item())
                        except Exception:
                            pass
            epoch_time = time.time() - t0
            avg_loss = running_loss / max(1, num_samples)
            if 'on_epoch' in self.callbacks:
                try:
                    self.callbacks['on_epoch'](epoch, avg_loss, epoch_time)
                except Exception:
                    pass
            if val_loader is not None:
                val_loss = self.evaluate(val_loader)
            else:
                val_loss = None
            if self.save_dir:
                self._save_checkpoint(epoch, extra={'val_loss': val_loss})
        return

    def evaluate(self, loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        n = 0
        with torch.no_grad():
            for batch in loader:
                batch = _to_device(batch, self.device)
                inputs, targets = batch if not isinstance(batch, dict) else (batch['input'], batch['target'])
                with autocast():
                    outputs = self.model(inputs)
                    loss = self.loss_fn(outputs, targets)
                total_loss += loss.item()
                n += 1
        self.model.train()
        return total_loss / max(1, n)


def prepare_qat_model(model: nn.Module, observer_ctor: Callable[..., ObserverBase] = None, **kwargs) -> nn.Module:
    """attach observers and fake-quant modules to model for qat."""
    if observer_ctor is None:
        observer_ctor = ObserverBase
    return prepare_qat(model, observer_ctor=observer_ctor, **kwargs)


def convert_qat_model(model: nn.Module) -> nn.Module:
    """convert a qat model by folding quantized weights into parent modules."""
    return convert_qat(model)


def enable_qat(model: nn.Module) -> None:
    """enable observers during training."""
    for m in model.modules():
        if hasattr(m, 'enabled'):
            try:
                setattr(m, 'enabled', True)
            except Exception:
                pass


def disable_qat(model: nn.Module) -> None:
    """disable observers for final evaluation."""
    for m in model.modules():
        if hasattr(m, 'enabled'):
            try:
                setattr(m, 'enabled', False)
            except Exception:
                pass


def freeze_batchnorm_stats(model: nn.Module) -> None:
    """set batchnorm layers to eval mode without freezing parameters."""
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()


def example_qat_workflow(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, optimizer: Optimizer, loss_fn: Callable, save_dir: str, device: Optional[torch.device] = None) -> None:
    """example workflow: prepare, train with qat, convert and evaluate."""
    model = prepare_qat_model(model)
    trainer = QATTrainer(model=model, optimizer=optimizer, loss_fn=loss_fn, device=device, save_dir=save_dir)
    enable_qat(model)
    trainer.train(train_loader, val_loader, epochs=20)
    disable_qat(model)
    model = convert_qat_model(model)
    return model
