"""
Training / evaluation loop.

Trains each model for 100 epochs with Adam + cross-entropy loss
and reports *validation accuracy averaged over the last 20 epochs* because
"the value at this time has tended to be stable".  This module implements
that exact protocol.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset


def pick_device(prefer: str = "auto") -> torch.device:
    """Return a sensible torch device.

    ``prefer`` may be ``"cpu"``, ``"cuda"``, ``"mps"`` or ``"auto"``.
    """

    prefer = prefer.lower()
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if prefer == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if prefer == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
    return torch.device("cpu")


@dataclass
class TrainConfig:
    """Hyper-parameters for :func:`train_and_evaluate`."""

    epochs: int = 100
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 0.0
    num_workers: int = 0
    last_k_epochs: int = 20  # matches paper's averaging window
    device: str = "auto"
    log_every: int = 0  # 0 -> silent


@dataclass
class TrainResult:
    """Output of :func:`train_and_evaluate`."""

    train_losses: list[float] = field(default_factory=list)
    train_accs: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    val_accs: list[float] = field(default_factory=list)

    @property
    def final_val_accuracy(self) -> float:
        """Average validation accuracy over the last ``last_k_epochs`` epochs."""
        if not self.val_accs:
            return float("nan")
        return float(sum(self.val_accs[-self._last_k:]) / min(self._last_k, len(self.val_accs)))

    _last_k: int = 20


def _run_epoch(
    model: nn.Module,
    loader: Iterable,
    criterion: nn.Module,
    optimizer: Optional[optim.Optimizer],
    device: torch.device,
) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    for batch in loader:
        x, y = batch
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(is_train):
            logits = model(x)
            loss = criterion(logits, y)
            if is_train:
                loss.backward()
                optimizer.step()
        total_loss += float(loss.item()) * x.size(0)
        total_correct += int((logits.argmax(1) == y).sum().item())
        total_seen += x.size(0)
    if total_seen == 0:
        return 0.0, 0.0
    return total_loss / total_seen, total_correct / total_seen


def train_and_evaluate(
    model: nn.Module,
    train_dataset: Dataset,
    val_dataset: Dataset,
    config: TrainConfig | None = None,
) -> TrainResult:
    """Train ``model`` and return training history + averaged final accuracy."""

    cfg = config or TrainConfig()
    device = pick_device(cfg.device)
    model.to(device)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        drop_last=False,
    )

    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.CrossEntropyLoss()

    result = TrainResult()
    result._last_k = cfg.last_k_epochs

    for epoch in range(1, cfg.epochs + 1):
        tr_loss, tr_acc = _run_epoch(model, train_loader, criterion, optimizer, device)
        vl_loss, vl_acc = _run_epoch(model, val_loader, criterion, None, device)
        result.train_losses.append(tr_loss)
        result.train_accs.append(tr_acc)
        result.val_losses.append(vl_loss)
        result.val_accs.append(vl_acc)
        if cfg.log_every and epoch % cfg.log_every == 0:
            print(
                f"[epoch {epoch:3d}/{cfg.epochs}] "
                f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
                f"val loss {vl_loss:.4f} acc {vl_acc:.4f}"
            )

    return result


__all__ = ["TrainConfig", "TrainResult", "train_and_evaluate", "pick_device"]
