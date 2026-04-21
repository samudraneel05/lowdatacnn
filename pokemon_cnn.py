"""
Classifies Pokemon by primary type (fire / water / grass) using three CNN
architectures - a configurable LeNet-5, AlexNet and ResNet-50 - and runs
experiments

Dataset:
    Folder-per-type layout (the Riley Wong
    `pokemon-images-dataset-by-type <https://github.com/rileynwong/pokemon-images-dataset-by-type>`_
    repo).  Only the ``fire/``, ``water/`` and ``grass/`` sub-directories are
    used.

Outputs (written to ``--output-dir``):
    * ``table2_results.csv``          - (9 OED runs + baselines)
    * ``table3_range_analysis.csv``   - (range / intuitive analysis)
    * ``figure4_range_values.png``    
    * ``figure5_num_conv_layers.png`` 
    * ``figure6_filter_size.png``   
    * ``figure7_dropout.png``
    * ``figure8_num_filters.png``
    * ``figure_model_comparison.png`` - LeNet-5 vs AlexNet vs ResNet-50
    * ``results.json``                - everything above in machine-readable form

Usage - see ``instructions.md`` for the Kaggle recipe.  Quick examples::

    # Full paper recreation (100 epochs x 9 OED runs + AlexNet + ResNet)
    python pokemon_cnn.py \\
        --data-dir /kaggle/working/pokemon-images-dataset-by-type \\
        --output-dir /kaggle/working/results \\
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
from PIL import Image
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models as tv_models
from torchvision import transforms


# ---------------------------------------------------------------------------
# 1. Constants: paper classes, OED factor levels, L9 array
# ---------------------------------------------------------------------------

PAPER_CLASSES: tuple[str, ...] = ("fire", "water", "grass")

FACTOR_NAMES: tuple[str, str, str, str] = ("A", "B", "C", "D")
FACTOR_DESCRIPTIONS: dict[str, str] = {
    "A": "Number of convolutional layers",
    "B": "Size of filter",
    "C": "Percentage of dropouts",
    "D": "Number of filters",
}

# Factor levels (rows of Table I of the paper).
LEVELS: dict[str, tuple] = {
    "A": (2, 3, 4),
    "B": (5, 7, 11),
    "C": (0.0, 0.2, 0.5),
    "D": ((6, 16, 16), (24, 64, 64), (96, 256, 256)),
}

# L9(3^4) orthogonal array
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

PAPER_OED_RESULTS: tuple[float, ...] = (
    0.7696, 0.5661, 0.4571, 0.6286, 0.5589, 0.5946, 0.6625, 0.7232, 0.4357,
)
PAPER_BASELINES: dict[str, float] = {"AlexNet": 0.5714, "ResNet": 0.6573}


# ---------------------------------------------------------------------------
# 2. Dataset (folder-per-type layout; filters to fire/water/grass)
# ---------------------------------------------------------------------------

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def _find_class_dir(root: Path, cls: str) -> Optional[Path]:
    """Find a sub-directory named ``cls`` up to two levels deep under ``root``."""
    for p in [root / cls] + list(root.glob(f"*/{cls}")) + list(root.glob(f"*/*/{cls}")):
        if p.is_dir():
            return p
    return None


class PokemonByTypeDataset(Dataset):
    """Pokemon images sorted into directories named after their primary type.

    Matches the layout of https://github.com/rileynwong/pokemon-images-dataset-by-type
    (``<root>/fire/*.png``, ``<root>/water/*.png``, ``<root>/grass/*.png``).
    Only the three classes are kept.
    """

    def __init__(
        self,
        root: str | Path,
        classes: Sequence[str] = PAPER_CLASSES,
        transform=None,
    ) -> None:
        self.root = Path(root)
        self.classes = tuple(c.lower() for c in classes)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.transform = transform

        samples: list[tuple[Path, int]] = []
        missing: list[str] = []
        for cls in self.classes:
            cdir = _find_class_dir(self.root, cls)
            if cdir is None:
                missing.append(cls)
                continue
            for p in sorted(cdir.rglob("*")):
                if p.is_file() and p.suffix.lower() in _IMAGE_EXTS:
                    samples.append((p, self.class_to_idx[cls]))
        if missing:
            raise FileNotFoundError(
                f"Could not find class sub-directories {missing} under {self.root}. "
                "Expected layout: <root>/fire/, <root>/water/, <root>/grass/."
            )
        if not samples:
            raise RuntimeError(f"No images found under {self.root}")
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[index]
        with Image.open(path) as img:
            img.load()
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def build_transforms(image_size: int, channels: int, train: bool) -> transforms.Compose:
    """Augmentation: flip / rotation / zoom / rescale for training."""
    mode = "RGBA" if channels == 4 else "RGB"
    to_mode = transforms.Lambda(lambda im: im.convert(mode))
    if train:
        return transforms.Compose([
            to_mode,
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=20),
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            transforms.ToTensor(),
        ])
    return transforms.Compose([
        to_mode,
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])


def _split_dataset(base: PokemonByTypeDataset, image_size: int, channels: int, val_fraction: float, seed: int):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(base)).tolist()
    n_val = max(1, int(round(len(base) * val_fraction)))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    class _Split(Dataset):
        def __init__(self, indices: list[int], train: bool):
            self.indices = indices
            self.tf = build_transforms(image_size, channels, train=train)

        def __len__(self) -> int:
            return len(self.indices)

        def __getitem__(self, i: int):
            path, label = base.samples[self.indices[i]]
            with Image.open(path) as img:
                img.load()
            return self.tf(img), label

    return _Split(train_idx, True), _Split(val_idx, False)


def build_splits(data_dir: Path, image_size: int, channels: int, val_fraction: float, seed: int):
    base = PokemonByTypeDataset(data_dir, classes=PAPER_CLASSES, transform=None)
    train, val = _split_dataset(base, image_size, channels, val_fraction, seed)
    return train, val, base


# ---------------------------------------------------------------------------
# 3. Models: configurable LeNet-5, AlexNet, ResNet-50
# ---------------------------------------------------------------------------

@dataclass
class LeNetConfig:
    """Four OED factors expressed as a LeNet-5 configuration."""
    num_conv_layers: int = 2
    filter_size: int = 5
    dropout: float = 0.0
    filters: tuple[int, int, int] = (6, 16, 16)
    num_classes: int = 3
    input_channels: int = 3
    input_size: int = 64
    fc_sizes: tuple[int, int] = (120, 84)


def _expand_filters(filters: tuple[int, int, int], n: int) -> list[int]:
    base = list(filters) + [filters[-1]] * max(0, n - len(filters))
    return base[:n]


class LeNet5(nn.Module):
    """LeNet-5 with the four factors exposed as ``LeNetConfig``.

    At ``LeNetConfig()`` defaults this reproduces the classic LeNet-5.
    Extra convolutional blocks (factor A levels 2 and 3) are appended and each
    conv block is followed by 2x2 max-pooling (until the spatial dim drops below
    2).  Dropout is inserted after each conv block and each FC layer when
    ``dropout > 0``.
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
    """AlexNet

    5 conv + 3 max-pool + 3 FC (last = 3-class softmax head), ReLU activations,
    dropout after the first two FC layers, input 120x120x3 (or 4 for RGBA).
    """

    def __init__(self, num_classes: int = 3, input_channels: int = 3, input_size: int = 120, dropout: float = 0.5):
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


def resnet50_head(num_classes: int = 3) -> nn.Module:
    """torchvision ResNet-50 with a 3-class linear head (trained from scratch)."""
    model = tv_models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# ---------------------------------------------------------------------------
# 4. Training loop (Adam + cross-entropy, avg of last K val accuracies)
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


def train_and_evaluate(model, train_ds, val_ds, epochs, batch_size, lr, device, num_workers=0, log_every=10) -> TrainResult:
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
# 5. OED range analysis (reproduces paper Table III)
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
# 6. Plots: Figures 4-8 + model comparison
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
    labels = [f"{FACTOR_DESCRIPTIONS[f]}" for f in FACTOR_NAMES]
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
    ax.set_title("Model comparison on small Pokemon dataset")
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
# 8. CLI / main
# ---------------------------------------------------------------------------

def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Reproduce tables and figures of Yuan & Zuo 2022 (MLISE).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir", default="/kaggle/working/pokemon-images-dataset-by-type",
                   help="Root of the Riley Wong dataset (contains fire/ water/ grass/ folders).")
    p.add_argument("--output-dir", default="/kaggle/working/results")
    p.add_argument("--epochs", type=int, default=100, help="Epochs per run (paper uses 100).")
    p.add_argument("--last-k", type=int, default=20,
                   help="Average the final K validation accuracies (paper uses 20).")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--lenet-image-size", type=int, default=64)
    p.add_argument("--alexnet-image-size", type=int, default=120)
    p.add_argument("--resnet-image-size", type=int, default=96,
                   help="Smaller than ImageNet 224 since we train from scratch on ~200 images.")
    p.add_argument("--skip-baselines", action="store_true",
                   help="Only run the 9 LeNet OED experiments (skip AlexNet / ResNet-50).")
    p.add_argument("--only-oed-run", type=int, default=None,
                   help="Run just row N (1..9) of the L9 array. Useful for splitting sessions.")
    p.add_argument("--demo", action="store_true",
                   help="No training - regenerate Table III and Figures 4-8 from the paper's "
                        "published Table II accuracies. Runs in ~2 seconds.")
    return p.parse_args(argv)


def _train_one_lenet(run: OEDRun, data_dir: Path, cfg: argparse.Namespace, device: torch.device):
    _set_seed(cfg.seed + run.run_id)
    train_ds, val_ds, base = build_splits(
        data_dir, cfg.lenet_image_size, channels=3, val_fraction=cfg.val_fraction, seed=cfg.seed,
    )
    if run.run_id == 1:
        print(f"[info] Train size: {len(train_ds)}  Val size: {len(val_ds)}  "
              f"(classes: {base.classes})")
    lenet_cfg = LeNetConfig(
        num_conv_layers=run.num_conv_layers,
        filter_size=run.filter_size,
        dropout=run.dropout,
        filters=run.filters,
        num_classes=3,
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


def _train_alexnet(data_dir: Path, cfg: argparse.Namespace, device: torch.device) -> dict:
    _set_seed(cfg.seed + 1001)
    train_ds, val_ds, _ = build_splits(
        data_dir, cfg.alexnet_image_size, channels=3, val_fraction=cfg.val_fraction, seed=cfg.seed,
    )
    model = AlexNet(num_classes=3, input_channels=3, input_size=cfg.alexnet_image_size)
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


def _train_resnet(data_dir: Path, cfg: argparse.Namespace, device: torch.device) -> dict:
    _set_seed(cfg.seed + 2002)
    train_ds, val_ds, _ = build_splits(
        data_dir, cfg.resnet_image_size, channels=3, val_fraction=cfg.val_fraction, seed=cfg.seed,
    )
    model = resnet50_head(num_classes=3)
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


def run_demo(out_dir: Path) -> None:
    print("=== DEMO MODE: regenerating paper outputs from published Table II values ===")
    analysis = range_analysis(PAPER_OED_RESULTS)
    print(format_analysis_text(analysis))
    out_dir.mkdir(parents=True, exist_ok=True)

    # Table II and III from the paper.
    runs = oed_runs()
    pseudo_results = [
        {"val_accuracy": acc} for acc in PAPER_OED_RESULTS
    ]
    write_table2(runs, pseudo_results, PAPER_BASELINES, out_dir / "table2_results.csv")
    write_table3(analysis, out_dir / "table3_range_analysis.csv")
    write_all_plots(analysis, PAPER_BASELINES, out_dir)

    save_json({
        "source": "paper Table II / Table III (no training performed)",
        "classes": list(PAPER_CLASSES),
        "oed_runs": [
            {**runs[i].__dict__, "val_accuracy": PAPER_OED_RESULTS[i]}
            for i in range(9)
        ],
        "baselines": PAPER_BASELINES,
        "analysis": {
            "ranking": analysis.ranking,
            "best_combination_by_Kbar": analysis.best_combination,
            "factor_summaries": {
                k: {"sums": v.sums, "means": v.means, "range_value": v.range_value,
                    "best_level": v.best_level}
                for k, v in analysis.factor_summaries.items()
            },
        },
    }, out_dir / "results.json")
    print(f"\n[done] Demo outputs written to {out_dir.resolve()}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    cfg = _parse_args(argv)
    out_dir = Path(cfg.output_dir)

    if cfg.demo:
        run_demo(out_dir)
        return 0

    data_dir = Path(cfg.data_dir)
    if not data_dir.exists():
        print(
            f"[error] --data-dir {data_dir} does not exist.\n"
            "        Clone https://github.com/rileynwong/pokemon-images-dataset-by-type\n"
            "        or pass --demo to regenerate the analysis from the paper's\n"
            "        published numbers.",
            file=sys.stderr,
        )
        return 2

    device = pick_device(cfg.device)
    print(f"[info] device = {device}  | epochs = {cfg.epochs}  "
          f"| batch_size = {cfg.batch_size}  | lr = {cfg.lr}")

    runs = oed_runs()
    selected_runs = (
        [r for r in runs if r.run_id == cfg.only_oed_run]
        if cfg.only_oed_run is not None
        else runs
    )
    if cfg.only_oed_run is not None and not selected_runs:
        raise SystemExit(f"--only-oed-run must be in 1..9, got {cfg.only_oed_run}")

    run_results: list[dict] = []
    for run in selected_runs:
        print(f"\n=== OED run {run.run_id}: {run.label} "
              f"(A={run.num_conv_layers} conv, B={run.filter_size}x{run.filter_size} filter, "
              f"C={run.dropout} dropout, D={run.filters}) ===")
        run_results.append(_train_one_lenet(run, data_dir, cfg, device))

    baselines: dict[str, float] = {}
    baselines_full: dict[str, dict] = {}
    if not cfg.skip_baselines and cfg.only_oed_run is None:
        print("\n=== AlexNet baseline ===")
        alex = _train_alexnet(data_dir, cfg, device)
        baselines["AlexNet"] = alex["val_accuracy"]
        baselines_full["AlexNet"] = alex
        print("\n=== ResNet-50 baseline ===")
        rn = _train_resnet(data_dir, cfg, device)
        baselines["ResNet"] = rn["val_accuracy"]
        baselines_full["ResNet"] = rn

    # Analysis only makes sense if we have all 9 runs.
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
        "classes": list(PAPER_CLASSES),
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
