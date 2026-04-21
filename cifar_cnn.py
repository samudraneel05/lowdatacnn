"""
Low-data architecture study on CIFAR-10.

This is the CIFAR-10 companion to ``pokemon_cnn.py``.  It runs exactly the
same L9(3^4) Taguchi orthogonal experimental design over four LeNet-5
architectural factors (number of conv layers, kernel size, dropout,
channel widths) plus AlexNet and ResNet-50 baselines - but on a
deliberately small subset of CIFAR-10: all 10 classes, ``N`` training
images per class (default 100) and ``M`` validation images per class
(default 50).  The script:

1. Downloads CIFAR-10 once via ``torchvision`` (``--data-dir`` is used
   as the download cache so it is re-usable across sessions).
2. Selects a class-balanced, deterministic subset.
3. Trains each of the 11 models (9 LeNet OED variants + AlexNet +
   ResNet-50) for ``--epochs`` epochs with Adam + cross-entropy.
4. Writes the paper-style outputs to ``--output-dir``:

   * ``table2_results.csv``           - per-run validation accuracy
   * ``table3_range_analysis.csv``    - K / K_bar / R / ranking / best combo
   * ``figure4_range_values.png``     - R per factor
   * ``figure5_num_conv_layers.png``  - factor A line plot
   * ``figure6_filter_size.png``      - factor B line plot
   * ``figure7_dropout.png``          - factor C line plot
   * ``figure8_num_filters.png``      - factor D line plot
   * ``figure_model_comparison.png``  - LeNet-5 best OED vs AlexNet vs ResNet
   * ``results.json``                 - everything above, machine-readable

Quick smoke test (~2 minutes on CPU)::

    python cifar_cnn.py --epochs 3 --train-per-class 30 --val-per-class 10 \\
                        --skip-baselines --device cpu

Full run on Kaggle GPU::

    python cifar_cnn.py \\
        --data-dir /kaggle/working/cifar_data \\
        --output-dir /kaggle/working/cifar_results \\
        --epochs 100
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models as tv_models
from torchvision import transforms
from torchvision.datasets import CIFAR10


# ---------------------------------------------------------------------------
# 1. Constants: OED factor levels, L9 array, CIFAR-10 class names
# ---------------------------------------------------------------------------

CIFAR10_CLASSES: tuple[str, ...] = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)

FACTOR_NAMES: tuple[str, str, str, str] = ("A", "B", "C", "D")
FACTOR_DESCRIPTIONS: dict[str, str] = {
    "A": "Number of convolutional layers",
    "B": "Size of filter",
    "C": "Percentage of dropouts",
    "D": "Number of filters",
}

LEVELS: dict[str, tuple] = {
    "A": (2, 3, 4),
    "B": (5, 7, 11),
    "C": (0.0, 0.2, 0.5),
    "D": ((6, 16, 16), (24, 64, 64), (96, 256, 256)),
}

# L9(3^4) orthogonal array
# so tables and figures are directly comparable across the two studies.
L9_ARRAY: tuple[tuple[int, int, int, int], ...] = (
    (1, 1, 1, 1),
    (1, 2, 3, 2),
    (1, 3, 2, 3),
    (2, 1, 3, 3),
    (2, 2, 2, 1),
    (2, 3, 1, 2),
    (3, 1, 2, 2),
    (3, 2, 1, 3),
    (3, 3, 3, 1),
)


# ---------------------------------------------------------------------------
# 2. CIFAR-10 subset dataset
# ---------------------------------------------------------------------------

def _select_balanced_indices(targets: list[int], per_class: int, num_classes: int,
                             rng: np.random.Generator) -> list[int]:
    targets_arr = np.asarray(targets)
    picks: list[int] = []
    for cls in range(num_classes):
        positions = np.where(targets_arr == cls)[0]
        if len(positions) < per_class:
            raise RuntimeError(
                f"Class {cls} has only {len(positions)} samples, "
                f"cannot pick {per_class} examples."
            )
        positions = positions.copy()
        rng.shuffle(positions)
        picks.extend(positions[:per_class].tolist())
    return picks


class CIFARSubset(Dataset):
    """A deterministic, class-balanced subset of CIFAR-10.

    Wraps the raw ``torchvision.datasets.CIFAR10`` dataset (which returns
    PIL images when ``transform=None``) and applies our augmentation
    pipeline on top.
    """

    def __init__(self, raw: CIFAR10, indices: Sequence[int], transform) -> None:
        self.raw = raw
        self.indices = list(indices)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        img, label = self.raw[self.indices[i]]
        if self.transform is not None:
            img = self.transform(img)
        return img, int(label)


def build_cifar_splits(
    data_dir: Path,
    image_size: int,
    train_per_class: int,
    val_per_class: int,
    num_classes: int,
    seed: int,
):
    """Download (if missing) and split CIFAR-10 into balanced subsets."""
    data_dir.mkdir(parents=True, exist_ok=True)
    raw_train = CIFAR10(root=str(data_dir), train=True, download=True, transform=None)
    raw_test = CIFAR10(root=str(data_dir), train=False, download=True, transform=None)

    rng = np.random.default_rng(seed)
    train_idx = _select_balanced_indices(list(raw_train.targets), train_per_class, num_classes, rng)
    val_idx = _select_balanced_indices(list(raw_test.targets), val_per_class, num_classes, rng)

    train_ds = CIFARSubset(raw_train, train_idx, build_transforms(image_size, train=True))
    val_ds = CIFARSubset(raw_test, val_idx, build_transforms(image_size, train=False))
    return train_ds, val_ds, CIFAR10_CLASSES[:num_classes]


def build_transforms(image_size: int, train: bool) -> transforms.Compose:
    """CIFAR-standard augmentation: pad+crop+flip (training), then resize."""
    steps: list = []
    if train:
        steps += [
            transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
            transforms.RandomHorizontalFlip(p=0.5),
        ]
    if image_size != 32:
        steps.append(transforms.Resize((image_size, image_size)))
    steps.append(transforms.ToTensor())
    return transforms.Compose(steps)


# ---------------------------------------------------------------------------
# 3. Models: configurable LeNet-5, AlexNet, ResNet-50
# ---------------------------------------------------------------------------

@dataclass
class LeNetConfig:
    num_conv_layers: int = 2
    filter_size: int = 5
    dropout: float = 0.0
    filters: tuple[int, int, int] = (6, 16, 16)
    num_classes: int = 10
    input_channels: int = 3
    input_size: int = 32
    fc_sizes: tuple[int, int] = (120, 84)


def _expand_filters(filters: tuple[int, int, int], n: int) -> list[int]:
    base = list(filters) + [filters[-1]] * max(0, n - len(filters))
    return base[:n]


class LeNet5(nn.Module):
    """LeNet-5 with the four OED factors exposed as ``LeNetConfig``.

    Each conv block: Conv2d(padding = kernel//2) -> ReLU -> MaxPool2d(2)
    (while spatial >= 2) -> Dropout2d(p) if p > 0.
    Two fully connected layers of widths 120 and 84 precede the classifier.
    """

    def __init__(self, cfg: LeNetConfig) -> None:
        super().__init__()
        self.cfg = cfg
        padding = cfg.filter_size // 2
        filters = _expand_filters(cfg.filters, cfg.num_conv_layers)

        layers: list[nn.Module] = []
        in_ch = cfg.input_channels
        spatial = cfg.input_size
        for out_ch in filters:
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=cfg.filter_size, padding=padding))
            layers.append(nn.ReLU(inplace=True))
            if spatial >= 2:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                spatial //= 2
            if cfg.dropout > 0:
                layers.append(nn.Dropout2d(p=cfg.dropout))
            in_ch = out_ch
        self.features = nn.Sequential(*layers)

        with torch.no_grad():
            dummy = torch.zeros(1, cfg.input_channels, cfg.input_size, cfg.input_size)
            flat = self.features(dummy).flatten(1).shape[1]

        clf: list[nn.Module] = [nn.Flatten()]
        prev = flat
        for h in cfg.fc_sizes:
            clf += [nn.Linear(prev, h), nn.ReLU(inplace=True)]
            if cfg.dropout > 0:
                clf.append(nn.Dropout(p=cfg.dropout))
            prev = h
        clf.append(nn.Linear(prev, cfg.num_classes))
        self.classifier = nn.Sequential(*clf)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class AlexNet(nn.Module):
    """Canonical AlexNet (5 conv + 3 FC) adapted to configurable input size."""

    def __init__(self, num_classes: int = 10, input_channels: int = 3,
                 input_size: int = 96, dropout: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(256, 384, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
        )
        with torch.no_grad():
            flat = self.features(torch.zeros(1, input_channels, input_size, input_size)).flatten(1).shape[1]
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, 4096), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(4096, 1024), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def resnet50_head(num_classes: int = 10) -> nn.Module:
    """torchvision ResNet-50 with an N-class linear head (random init)."""
    model = tv_models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# ---------------------------------------------------------------------------
# 4. Training loop
# ---------------------------------------------------------------------------

def pick_device(prefer: str = "auto") -> torch.device:
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


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _run_epoch(model, loader, criterion, optimizer, device):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_correct = 0
    total = 0
    for x, y in loader:
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
        total += x.size(0)
    if total == 0:
        return 0.0, 0.0
    return total_loss / total, total_correct / total


@dataclass
class TrainResult:
    train_losses: list[float] = field(default_factory=list)
    train_accs: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    val_accs: list[float] = field(default_factory=list)

    def final_val_accuracy(self, last_k: int) -> float:
        if not self.val_accs:
            return float("nan")
        k = min(last_k, len(self.val_accs))
        return float(sum(self.val_accs[-k:]) / k)


def train_and_evaluate(model, train_ds, val_ds, epochs, batch_size, lr,
                       device, num_workers=0, log_every=10) -> TrainResult:
    model.to(device)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    result = TrainResult()
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = _run_epoch(model, train_loader, criterion, optimizer, device)
        vl_loss, vl_acc = _run_epoch(model, val_loader, criterion, None, device)
        result.train_losses.append(tr_loss); result.train_accs.append(tr_acc)
        result.val_losses.append(vl_loss); result.val_accs.append(vl_acc)
        if log_every and (epoch == 1 or epoch == epochs or epoch % log_every == 0):
            print(
                f"  [epoch {epoch:3d}/{epochs}] "
                f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
                f"val loss {vl_loss:.4f} acc {vl_acc:.4f}"
            )
    return result


# ---------------------------------------------------------------------------
# 5. OED range analysis
# ---------------------------------------------------------------------------

@dataclass
class OEDRun:
    run_id: int
    levels: tuple[int, int, int, int]
    num_conv_layers: int
    filter_size: int
    dropout: float
    filters: tuple[int, int, int]

    @property
    def label(self) -> str:
        return "".join(f"{f}{lv}" for f, lv in zip(FACTOR_NAMES, self.levels))


def oed_runs() -> list[OEDRun]:
    out = []
    for i, levels in enumerate(L9_ARRAY, start=1):
        a, b, c, d = levels
        out.append(OEDRun(
            run_id=i, levels=levels,
            num_conv_layers=LEVELS["A"][a - 1],
            filter_size=LEVELS["B"][b - 1],
            dropout=LEVELS["C"][c - 1],
            filters=LEVELS["D"][d - 1],
        ))
    return out


@dataclass
class FactorSummary:
    factor: str
    description: str
    sums: tuple[float, float, float]
    means: tuple[float, float, float]
    range_value: float
    best_level: int


@dataclass
class OEDAnalysis:
    results: tuple[float, ...]
    factor_summaries: dict[str, FactorSummary]
    ranking: tuple[str, ...]
    best_combination: tuple[int, int, int, int]


def range_analysis(results: Sequence[float]) -> OEDAnalysis:
    if len(results) != 9:
        raise ValueError("range_analysis expects exactly 9 results")
    summaries: dict[str, FactorSummary] = {}
    for f_idx, factor in enumerate(FACTOR_NAMES):
        sums = [0.0, 0.0, 0.0]
        counts = [0, 0, 0]
        for run_idx, row in enumerate(L9_ARRAY):
            level = row[f_idx]
            sums[level - 1] += results[run_idx]
            counts[level - 1] += 1
        means = [s / c if c else 0.0 for s, c in zip(sums, counts)]
        r_value = max(means) - min(means)
        best_level = max(range(3), key=lambda i: means[i]) + 1
        summaries[factor] = FactorSummary(
            factor=factor,
            description=FACTOR_DESCRIPTIONS[factor],
            sums=tuple(sums),
            means=tuple(means),
            range_value=r_value,
            best_level=best_level,
        )
    ranking = tuple(sorted(FACTOR_NAMES, key=lambda f: summaries[f].range_value, reverse=True))
    best_combo = tuple(summaries[f].best_level for f in FACTOR_NAMES)
    return OEDAnalysis(
        results=tuple(results),
        factor_summaries=summaries,
        ranking=ranking,
        best_combination=best_combo,  # type: ignore[arg-type]
    )


def format_analysis_text(analysis: OEDAnalysis) -> str:
    lines = []
    header = f"{'Index':<8}" + "".join(f"{f:<12}" for f in FACTOR_NAMES)
    lines.append(header)
    lines.append("-" * len(header))
    for i in range(3):
        row = f"K{i + 1:<7}"
        for f in FACTOR_NAMES:
            row += f"{analysis.factor_summaries[f].sums[i]:<12.4f}"
        lines.append(row)
    for i in range(3):
        row = f"Kbar{i + 1:<4}"
        for f in FACTOR_NAMES:
            row += f"{analysis.factor_summaries[f].means[i]:<12.4f}"
        lines.append(row)
    row = f"{'R':<8}"
    for f in FACTOR_NAMES:
        row += f"{analysis.factor_summaries[f].range_value:<12.4f}"
    lines.append(row)
    lines.append(f"Ranking: {' > '.join(analysis.ranking)}")
    best = "".join(f"{f}{lv}" for f, lv in zip(FACTOR_NAMES, analysis.best_combination))
    lines.append(f"Best combination (by K-bar): {best}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 6. Plots
# ---------------------------------------------------------------------------

def _savefig(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _level_label(factor: str, idx: int) -> str:
    v = LEVELS[factor][idx]
    if factor == "B":
        return f"{v}x{v}"
    if factor == "C":
        return f"{v:.1f}"
    if factor == "D":
        a, b, c = v
        return f"{a}-{b}-({c})"
    return str(v)


def plot_range_values(analysis: OEDAnalysis, path: Path) -> None:
    labels = [FACTOR_DESCRIPTIONS[f] for f in FACTOR_NAMES]
    values = [analysis.factor_summaries[f].range_value for f in FACTOR_NAMES]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, values, color="#4C72B0")
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{v:.4f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Range value R")
    ax.set_title("Figure 4. Range values of different factors")
    ax.set_ylim(0, max(values) * 1.25 if values else 1)
    plt.xticks(rotation=12, ha="right")
    fig.tight_layout()
    _savefig(fig, path)


def plot_factor_mean(analysis: OEDAnalysis, factor: str, title: str, path: Path) -> None:
    summary = analysis.factor_summaries[factor]
    x_labels = [_level_label(factor, i) for i in range(3)]
    y = list(summary.means)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(x_labels, y, marker="o", color="#4C72B0")
    for xi, yi in zip(x_labels, y):
        ax.annotate(f"{yi:.4f}", (xi, yi), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=9)
    ax.set_xlabel(FACTOR_DESCRIPTIONS[factor])
    ax.set_ylabel("Validation accuracy (mean)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _savefig(fig, path)


def plot_model_comparison(names: Sequence[str], accs: Sequence[float], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4))
    bars = ax.bar(list(names), list(accs), color="#55A868")
    for bar, v in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{v:.4f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Validation accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("Model comparison on CIFAR-10 subset")
    plt.xticks(rotation=8, ha="right")
    fig.tight_layout()
    _savefig(fig, path)


def write_all_plots(analysis: OEDAnalysis, baselines_for_cmp: dict[str, float], out_dir: Path) -> None:
    plot_range_values(analysis, out_dir / "figure4_range_values.png")
    plot_factor_mean(analysis, "A",
                     "Figure 5. Number of convolutional layers vs. validation accuracy",
                     out_dir / "figure5_num_conv_layers.png")
    plot_factor_mean(analysis, "B",
                     "Figure 6. Size of filter vs. validation accuracy",
                     out_dir / "figure6_filter_size.png")
    plot_factor_mean(analysis, "C",
                     "Figure 7. Percentage of dropouts vs. validation accuracy",
                     out_dir / "figure7_dropout.png")
    plot_factor_mean(analysis, "D",
                     "Figure 8. Number of filters vs. validation accuracy",
                     out_dir / "figure8_num_filters.png")

    best_lenet = max(analysis.results)
    names = ["LeNet-5 (best OED)"]
    accs = [best_lenet]
    for k, v in baselines_for_cmp.items():
        names.append(k)
        accs.append(v)
    plot_model_comparison(names, accs, out_dir / "figure_model_comparison.png")


# ---------------------------------------------------------------------------
# 7. CSV / JSON writers
# ---------------------------------------------------------------------------

def write_table2(runs: list[OEDRun], run_results: list[dict], baselines: dict[str, float], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "label", "num_conv_layers", "filter_size", "dropout",
                    "filters", "val_accuracy"])
        for run, res in zip(runs, run_results):
            w.writerow([
                run.run_id, run.label, run.num_conv_layers, run.filter_size, run.dropout,
                "-".join(str(x) for x in run.filters),
                f"{res['val_accuracy']:.4f}",
            ])
        for name, acc in baselines.items():
            w.writerow([name, name, "-", "-", "-", "-", f"{acc:.4f}"])


def write_table3(analysis: OEDAnalysis, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index"] + list(FACTOR_NAMES))
        for i in range(3):
            w.writerow([f"K{i + 1}"] + [f"{analysis.factor_summaries[fn].sums[i]:.4f}" for fn in FACTOR_NAMES])
        for i in range(3):
            w.writerow([f"Kbar{i + 1}"] + [f"{analysis.factor_summaries[fn].means[i]:.4f}" for fn in FACTOR_NAMES])
        w.writerow(["R"] + [f"{analysis.factor_summaries[fn].range_value:.4f}" for fn in FACTOR_NAMES])
        w.writerow(["ranking"] + [" > ".join(analysis.ranking), "", "", ""])
        best = "".join(f"{f}{lv}" for f, lv in zip(FACTOR_NAMES, analysis.best_combination))
        w.writerow(["best_combination_by_Kbar"] + [best, "", "", ""])


def save_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    def default(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if hasattr(o, "tolist"):
            return o.tolist()
        if hasattr(o, "__dict__"):
            return o.__dict__
        raise TypeError(f"Cannot serialise {type(o).__name__}")
    with path.open("w") as f:
        json.dump(payload, f, indent=2, default=default)


# ---------------------------------------------------------------------------
# 8. CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Low-data L9(3^4) OED architecture study on a CIFAR-10 subset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir", default="/kaggle/working/cifar_data",
                   help="Cache directory for the CIFAR-10 download.")
    p.add_argument("--output-dir", default="/kaggle/working/cifar_results")
    p.add_argument("--train-per-class", type=int, default=100,
                   help="Number of training images sampled per class.")
    p.add_argument("--val-per-class", type=int, default=50,
                   help="Number of validation images sampled per class "
                        "(drawn from the CIFAR-10 test split).")
    p.add_argument("--num-classes", type=int, default=10,
                   help="Number of CIFAR-10 classes to include (first N of the "
                        "canonical 10).")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--last-k", type=int, default=20,
                   help="Average the final K validation accuracies.")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--lenet-image-size", type=int, default=32,
                   help="CIFAR's native 32x32 works well for LeNet.")
    p.add_argument("--alexnet-image-size", type=int, default=96,
                   help="CIFAR is upscaled for AlexNet (its receptive field needs it).")
    p.add_argument("--resnet-image-size", type=int, default=96,
                   help="CIFAR is upscaled for ResNet-50.")
    p.add_argument("--skip-baselines", action="store_true",
                   help="Only run the 9 LeNet OED experiments.")
    p.add_argument("--only-oed-run", type=int, default=None,
                   help="Run just row N (1..9) of the L9 array.")
    return p.parse_args(argv)


def _train_one_lenet(run: OEDRun, cfg: argparse.Namespace, device: torch.device,
                     num_classes: int, printed_sizes: list[bool]) -> dict:
    _set_seed(cfg.seed + run.run_id)
    train_ds, val_ds, classes = build_cifar_splits(
        Path(cfg.data_dir), cfg.lenet_image_size,
        cfg.train_per_class, cfg.val_per_class, num_classes, cfg.seed,
    )
    if not printed_sizes[0]:
        print(f"[info] Train size: {len(train_ds)}  Val size: {len(val_ds)}  "
              f"(classes: {classes})")
        printed_sizes[0] = True
    lenet_cfg = LeNetConfig(
        num_conv_layers=run.num_conv_layers,
        filter_size=run.filter_size,
        dropout=run.dropout,
        filters=run.filters,
        num_classes=num_classes,
        input_channels=3,
        input_size=cfg.lenet_image_size,
    )
    model = LeNet5(lenet_cfg)
    t0 = time.perf_counter()
    tr = train_and_evaluate(
        model, train_ds, val_ds,
        epochs=cfg.epochs, batch_size=cfg.batch_size, lr=cfg.lr,
        device=device, num_workers=cfg.num_workers,
        log_every=max(1, cfg.epochs // 5),
    )
    dt = time.perf_counter() - t0
    acc = tr.final_val_accuracy(cfg.last_k)
    print(f"[run {run.run_id}] {run.label}  val_acc (avg last {cfg.last_k}) = {acc:.4f}   [{dt:.1f}s]")
    return {
        "run_id": run.run_id,
        "label": run.label,
        "levels": list(run.levels),
        "num_conv_layers": run.num_conv_layers,
        "filter_size": run.filter_size,
        "dropout": run.dropout,
        "filters": list(run.filters),
        "val_accuracy": acc,
        "val_acc_history": tr.val_accs,
        "train_acc_history": tr.train_accs,
        "duration_sec": dt,
    }


def _train_alexnet(cfg: argparse.Namespace, device: torch.device, num_classes: int) -> dict:
    _set_seed(cfg.seed + 1001)
    train_ds, val_ds, _ = build_cifar_splits(
        Path(cfg.data_dir), cfg.alexnet_image_size,
        cfg.train_per_class, cfg.val_per_class, num_classes, cfg.seed,
    )
    model = AlexNet(num_classes=num_classes, input_channels=3, input_size=cfg.alexnet_image_size)
    t0 = time.perf_counter()
    tr = train_and_evaluate(
        model, train_ds, val_ds,
        epochs=cfg.epochs, batch_size=cfg.batch_size, lr=cfg.lr,
        device=device, num_workers=cfg.num_workers, log_every=max(1, cfg.epochs // 5),
    )
    dt = time.perf_counter() - t0
    acc = tr.final_val_accuracy(cfg.last_k)
    print(f"[AlexNet] val_acc = {acc:.4f}   [{dt:.1f}s]")
    return {"val_accuracy": acc, "val_acc_history": tr.val_accs, "duration_sec": dt}


def _train_resnet(cfg: argparse.Namespace, device: torch.device, num_classes: int) -> dict:
    _set_seed(cfg.seed + 2002)
    train_ds, val_ds, _ = build_cifar_splits(
        Path(cfg.data_dir), cfg.resnet_image_size,
        cfg.train_per_class, cfg.val_per_class, num_classes, cfg.seed,
    )
    model = resnet50_head(num_classes=num_classes)
    t0 = time.perf_counter()
    tr = train_and_evaluate(
        model, train_ds, val_ds,
        epochs=cfg.epochs, batch_size=cfg.batch_size, lr=cfg.lr,
        device=device, num_workers=cfg.num_workers, log_every=max(1, cfg.epochs // 5),
    )
    dt = time.perf_counter() - t0
    acc = tr.final_val_accuracy(cfg.last_k)
    print(f"[ResNet-50] val_acc = {acc:.4f}   [{dt:.1f}s]")
    return {"val_accuracy": acc, "val_acc_history": tr.val_accs, "duration_sec": dt}


def main(argv: Optional[Sequence[str]] = None) -> int:
    cfg = _parse_args(argv)
    if not (1 <= cfg.num_classes <= len(CIFAR10_CLASSES)):
        raise SystemExit(f"--num-classes must be in 1..{len(CIFAR10_CLASSES)}")
    out_dir = Path(cfg.output_dir)

    device = pick_device(cfg.device)
    print(f"[info] device = {device}  | epochs = {cfg.epochs}  "
          f"| batch_size = {cfg.batch_size}  | lr = {cfg.lr}")
    print(f"[info] CIFAR-10 subset: {cfg.num_classes} classes x "
          f"{cfg.train_per_class} train / {cfg.val_per_class} val per class")

    runs = oed_runs()
    selected_runs = (
        [r for r in runs if r.run_id == cfg.only_oed_run]
        if cfg.only_oed_run is not None
        else runs
    )
    if cfg.only_oed_run is not None and not selected_runs:
        raise SystemExit(f"--only-oed-run must be in 1..9, got {cfg.only_oed_run}")

    printed_sizes = [False]
    run_results: list[dict] = []
    for run in selected_runs:
        print(f"\n=== OED run {run.run_id}: {run.label} "
              f"(A={run.num_conv_layers} conv, B={run.filter_size}x{run.filter_size} filter, "
              f"C={run.dropout} dropout, D={run.filters}) ===")
        run_results.append(_train_one_lenet(run, cfg, device, cfg.num_classes, printed_sizes))

    baselines: dict[str, float] = {}
    baselines_full: dict[str, dict] = {}
    if not cfg.skip_baselines and cfg.only_oed_run is None:
        print("\n=== AlexNet baseline ===")
        alex = _train_alexnet(cfg, device, cfg.num_classes)
        baselines["AlexNet"] = alex["val_accuracy"]
        baselines_full["AlexNet"] = alex
        print("\n=== ResNet-50 baseline ===")
        rn = _train_resnet(cfg, device, cfg.num_classes)
        baselines["ResNet"] = rn["val_accuracy"]
        baselines_full["ResNet"] = rn

    analysis: Optional[OEDAnalysis] = None
    if cfg.only_oed_run is None:
        analysis = range_analysis([r["val_accuracy"] for r in run_results])
        print("\n" + format_analysis_text(analysis))
        write_table3(analysis, out_dir / "table3_range_analysis.csv")
        write_all_plots(analysis, baselines, out_dir)

    write_table2(runs if cfg.only_oed_run is None else selected_runs,
                 run_results, baselines, out_dir / "table2_results.csv")

    save_json({
        "config": vars(cfg),
        "classes": list(CIFAR10_CLASSES[:cfg.num_classes]),
        "oed_runs": run_results,
        "baselines": baselines_full,
        "analysis": None if analysis is None else {
            "ranking": analysis.ranking,
            "best_combination_by_Kbar": analysis.best_combination,
            "factor_summaries": {
                k: {"sums": v.sums, "means": v.means, "range_value": v.range_value,
                    "best_level": v.best_level}
                for k, v in analysis.factor_summaries.items()
            },
        },
    }, out_dir / "results.json")

    print(f"\n[done] Results written to {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
