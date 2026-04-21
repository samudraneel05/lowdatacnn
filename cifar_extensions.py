"""
Three extensions to the baseline ``cifar_cnn.py`` study, picked to answer
two concrete research questions:

    Q1. *Which knobs actually matter?*
        Beyond architectural hyper-parameters, how do **augmentation
        strength** and **training-time regularisation** compare as sources
        of accuracy?  We run three L_9 studies (architecture, augmentation,
        regularisation) and plot the resulting factor-range values
        side-by-side.

    Q2. *How much data does each architecture need to catch up?*
        We train four architectures - a small LeNet-5, AlexNet, ResNet-50
        from scratch, and ResNet-50 pre-trained on ImageNet - at several
        training-set sizes and plot accuracy vs. data on a log axis.

Experiments implemented
-----------------------
* ``--experiment sample-efficiency`` -- accuracy vs. training-set size per
  architecture, including pre-trained ResNet-50.
* ``--experiment augmentation-oed``  -- an L_9(3^4) OED that treats
  augmentation strength as the design.
* ``--experiment regularisation-oed`` -- an L_9(3^4) OED that treats
  dropout / weight-decay / label-smoothing / learning-rate as the design.
* ``--experiment all``               -- all three, then a combined
  "what-matters" comparison plot.

Outputs (written under ``--output-dir``)
---------------------------------------
* ``sample_efficiency.csv``                and ``figure_sample_efficiency.png``
* ``aug_table2.csv``   ``aug_table3.csv``  and ``figure_aug_*.png``
* ``reg_table2.csv``   ``reg_table3.csv``  and ``figure_reg_*.png``
* ``figure_what_matters.png``              -- side-by-side R-values across
  the three OED studies (architecture R values loaded from
  ``--arch-results-json`` if available).
* ``extensions_results.json``              -- everything, machine-readable.

Quick CPU smoke test (~3 minutes)::

    python cifar_extensions.py --experiment all \\
        --epochs 3 --train-per-class 30 --val-per-class 20 \\
        --sample-sizes 30 60 --device cpu --skip-pretrained

Full Kaggle-GPU run::

    python cifar_extensions.py --experiment all \\
        --data-dir /kaggle/working/cifar_data \\
        --output-dir /kaggle/working/cifar_extensions_results \\
        --arch-results-json /kaggle/working/cifar_results/results.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms

# ---- re-use everything from the baseline CIFAR script ---------------------
from cifar_cnn import (
    CIFAR10_CLASSES,
    CIFARSubset,
    FACTOR_DESCRIPTIONS,
    FACTOR_NAMES,
    FactorSummary,
    L9_ARRAY,
    LEVELS,
    LeNet5,
    LeNetConfig,
    OEDAnalysis,
    OEDRun,
    AlexNet,
    TrainResult,
    _level_label,
    _run_epoch,
    _savefig,
    _set_seed,
    build_cifar_splits,
    format_analysis_text,
    oed_runs,
    pick_device,
    range_analysis,
    resnet50_head,
    save_json,
    write_table2,
    write_table3,
)
from torchvision.datasets import CIFAR10


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

# "Fixed" LeNet used when the experiment varies something other than the
# architecture (augmentation-OED, regularisation-OED).  Chosen to be in the
# upper half of the baseline study's L_9 runs without being an extremum.
FIXED_LENET_CFG = LeNetConfig(
    num_conv_layers=3, filter_size=5, dropout=0.0,
    filters=(24, 64, 64),
    num_classes=10, input_channels=3, input_size=32,
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _tiny_cifar_raw(data_dir: Path):
    """Load (and cache-download) the raw CIFAR-10 train/test splits once."""
    data_dir.mkdir(parents=True, exist_ok=True)
    raw_train = CIFAR10(root=str(data_dir), train=True, download=True, transform=None)
    raw_test = CIFAR10(root=str(data_dir), train=False, download=True, transform=None)
    return raw_train, raw_test


def _balanced_indices(targets, per_class: int, num_classes: int, rng: np.random.Generator) -> list[int]:
    targets_arr = np.asarray(targets)
    picks: list[int] = []
    for cls in range(num_classes):
        positions = np.where(targets_arr == cls)[0]
        if len(positions) < per_class:
            raise RuntimeError(f"Class {cls} has {len(positions)} samples < {per_class}")
        positions = positions.copy()
        rng.shuffle(positions)
        picks.extend(positions[:per_class].tolist())
    return picks


def _loaders(train_ds, val_ds, batch_size: int, num_workers: int) -> tuple[DataLoader, DataLoader]:
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    )


def train_with(
    model: nn.Module,
    train_ds,
    val_ds,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    num_workers: int = 0,
    weight_decay: float = 0.0,
    label_smoothing: float = 0.0,
    log_every: int = 0,
) -> TrainResult:
    """Training loop with hooks for weight-decay and label-smoothing."""
    model.to(device)
    train_loader, val_loader = _loaders(train_ds, val_ds, batch_size, num_workers)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    result = TrainResult()
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = _run_epoch(model, train_loader, criterion, optimizer, device)
        vl_loss, vl_acc = _run_epoch(model, val_loader, criterion, None, device)
        result.train_losses.append(tr_loss); result.train_accs.append(tr_acc)
        result.val_losses.append(vl_loss); result.val_accs.append(vl_acc)
        if log_every and (epoch == 1 or epoch == epochs or epoch % log_every == 0):
            print(f"    [epoch {epoch:3d}/{epochs}] train {tr_acc:.3f} | val {vl_acc:.3f}")
    return result


# ---------------------------------------------------------------------------
# Pre-trained ResNet-50 head
# ---------------------------------------------------------------------------

def resnet50_pretrained(num_classes: int) -> nn.Module:
    """ResNet-50 initialised from ImageNet-1K V2 weights with a new head."""
    from torchvision.models import ResNet50_Weights, resnet50
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# ---------------------------------------------------------------------------
# EXPERIMENT A -- Sample efficiency (incl. pre-trained ResNet)
# ---------------------------------------------------------------------------

def _build_default_transforms(image_size: int, train: bool, *, imagenet_norm: bool = False):
    """Same augmentation as cifar_cnn, plus optional ImageNet normalisation."""
    steps: list = []
    if train:
        steps += [
            transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
            transforms.RandomHorizontalFlip(p=0.5),
        ]
    if image_size != 32:
        steps.append(transforms.Resize((image_size, image_size)))
    steps.append(transforms.ToTensor())
    if imagenet_norm:
        steps.append(transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD))
    return transforms.Compose(steps)


@dataclass
class ModelSpec:
    name: str
    build: Callable[[int], nn.Module]
    image_size: int
    imagenet_norm: bool = False
    # Optional override for epoch count (pre-trained nets converge faster).
    epochs_override: Optional[int] = None


def _default_model_specs(num_classes: int, skip_pretrained: bool) -> list[ModelSpec]:
    specs: list[ModelSpec] = [
        ModelSpec(
            name="LeNet-5",
            build=lambda n: LeNet5(LeNetConfig(
                num_conv_layers=3, filter_size=5, dropout=0.0, filters=(24, 64, 64),
                num_classes=n, input_channels=3, input_size=32,
            )),
            image_size=32,
        ),
        ModelSpec(
            name="AlexNet",
            build=lambda n: AlexNet(num_classes=n, input_channels=3, input_size=96),
            image_size=96,
        ),
        ModelSpec(
            name="ResNet-50 (scratch)",
            build=lambda n: resnet50_head(num_classes=n),
            image_size=96,
        ),
    ]
    if not skip_pretrained:
        specs.append(ModelSpec(
            name="ResNet-50 (ImageNet pretrained)",
            build=lambda n: resnet50_pretrained(num_classes=n),
            image_size=96,
            imagenet_norm=True,
            epochs_override=None,  # same epoch budget; converges faster anyway
        ))
    return specs


def run_sample_efficiency(
    *,
    data_dir: Path,
    out_dir: Path,
    sample_sizes: Sequence[int],
    val_per_class: int,
    num_classes: int,
    epochs: int,
    batch_size: int,
    lr: float,
    last_k: int,
    device: torch.device,
    num_workers: int,
    seed: int,
    skip_pretrained: bool,
) -> dict:
    """For each (size, model), train on a class-balanced subset and record val accuracy."""
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = _default_model_specs(num_classes, skip_pretrained)

    raw_train, raw_test = _tiny_cifar_raw(data_dir)

    # Pre-compute the validation split once (shared across N).
    val_rng = np.random.default_rng(seed ^ 0xBEEF)
    val_idx = _balanced_indices(raw_test.targets, val_per_class, num_classes, val_rng)

    records: list[dict] = []
    for n_per_class in sample_sizes:
        train_rng = np.random.default_rng(seed ^ (0xABCDEF + n_per_class))
        train_idx = _balanced_indices(raw_train.targets, n_per_class, num_classes, train_rng)
        for spec in specs:
            # Deterministic per-model offset so the seed varies predictably.
            name_offset = sum(ord(c) for c in spec.name) % 10_000
            _set_seed(seed + n_per_class + name_offset)
            train_tf = _build_default_transforms(spec.image_size, train=True, imagenet_norm=spec.imagenet_norm)
            val_tf = _build_default_transforms(spec.image_size, train=False, imagenet_norm=spec.imagenet_norm)
            train_ds = CIFARSubset(raw_train, train_idx, train_tf)
            val_ds = CIFARSubset(raw_test, val_idx, val_tf)
            model = spec.build(num_classes)
            ep = spec.epochs_override or epochs
            print(f"\n[sample-efficiency] n={n_per_class:>4}  model={spec.name}  "
                  f"(train={len(train_ds)}, val={len(val_ds)}, ep={ep})")
            t0 = time.perf_counter()
            tr = train_with(
                model, train_ds, val_ds,
                epochs=ep, batch_size=batch_size, lr=lr,
                device=device, num_workers=num_workers,
                log_every=max(1, ep // 3),
            )
            dt = time.perf_counter() - t0
            acc = tr.final_val_accuracy(last_k)
            print(f"    -> val_acc (avg last {last_k}) = {acc:.4f}  [{dt:.1f}s]")
            records.append({
                "model": spec.name,
                "train_per_class": n_per_class,
                "total_train": len(train_ds),
                "val_accuracy": acc,
                "val_acc_history": tr.val_accs,
                "duration_sec": dt,
            })

    # Write CSV.
    csv_path = out_dir / "sample_efficiency.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "train_per_class", "total_train", "val_accuracy"])
        for r in records:
            w.writerow([r["model"], r["train_per_class"], r["total_train"], f"{r['val_accuracy']:.4f}"])
    _plot_sample_efficiency(records, out_dir / "figure_sample_efficiency.png")
    return {"records": records}


def _plot_sample_efficiency(records: Sequence[dict], path: Path) -> None:
    by_model: dict[str, list[tuple[int, float]]] = {}
    for r in records:
        by_model.setdefault(r["model"], []).append((r["train_per_class"], r["val_accuracy"]))
    for k in by_model:
        by_model[k].sort(key=lambda p: p[0])

    colors = {"LeNet-5": "#4C72B0", "AlexNet": "#DD8452",
              "ResNet-50 (scratch)": "#55A868",
              "ResNet-50 (ImageNet pretrained)": "#C44E52"}

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, points in by_model.items():
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        ax.plot(xs, ys, marker="o", label=name, color=colors.get(name))
        for x, y in points:
            ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                        xytext=(0, 6), ha="center", fontsize=7)
    ax.set_xscale("log")
    ax.set_xlabel("Training images per class  (log scale)")
    ax.set_ylabel("Validation accuracy (mean of last epochs)")
    ax.set_title("Sample-efficiency: accuracy vs. training-set size per architecture")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower right", fontsize=9, frameon=True)
    fig.tight_layout()
    _savefig(fig, path)


# ---------------------------------------------------------------------------
# EXPERIMENT B -- Augmentation-strength OED
# ---------------------------------------------------------------------------

AUG_FACTOR_DESCRIPTIONS: dict[str, str] = {
    "A": "RandomCrop padding",
    "B": "HFlip probability",
    "C": "Rotation degrees",
    "D": "ColorJitter strength",
}

AUG_LEVELS: dict[str, tuple] = {
    "A": (0, 4, 8),
    "B": (0.0, 0.3, 0.5),
    "C": (0, 10, 25),
    "D": (0.0, 0.2, 0.4),
}


@dataclass
class AugRun:
    run_id: int
    levels: tuple[int, int, int, int]
    crop_padding: int
    hflip_p: float
    rotation_deg: int
    color_jitter: float

    @property
    def label(self) -> str:
        return "".join(f"{f}{lv}" for f, lv in zip(FACTOR_NAMES, self.levels))


def aug_runs() -> list[AugRun]:
    out = []
    for i, levels in enumerate(L9_ARRAY, start=1):
        a, b, c, d = levels
        out.append(AugRun(
            run_id=i, levels=levels,
            crop_padding=AUG_LEVELS["A"][a - 1],
            hflip_p=AUG_LEVELS["B"][b - 1],
            rotation_deg=AUG_LEVELS["C"][c - 1],
            color_jitter=AUG_LEVELS["D"][d - 1],
        ))
    return out


def _build_aug_transforms(image_size: int, aug: AugRun, train: bool):
    steps: list = []
    if train:
        if aug.crop_padding > 0:
            steps.append(transforms.RandomCrop(32, padding=aug.crop_padding, padding_mode="reflect"))
        if aug.hflip_p > 0:
            steps.append(transforms.RandomHorizontalFlip(p=aug.hflip_p))
        if aug.rotation_deg > 0:
            steps.append(transforms.RandomRotation(degrees=aug.rotation_deg))
        if aug.color_jitter > 0:
            cj = aug.color_jitter
            steps.append(transforms.ColorJitter(brightness=cj, contrast=cj, saturation=cj))
    if image_size != 32:
        steps.append(transforms.Resize((image_size, image_size)))
    steps.append(transforms.ToTensor())
    return transforms.Compose(steps)


def run_augmentation_oed(
    *,
    data_dir: Path,
    out_dir: Path,
    train_per_class: int,
    val_per_class: int,
    num_classes: int,
    epochs: int,
    batch_size: int,
    lr: float,
    last_k: int,
    device: torch.device,
    num_workers: int,
    seed: int,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_train, raw_test = _tiny_cifar_raw(data_dir)

    rng = np.random.default_rng(seed ^ 0xA06)
    train_idx = _balanced_indices(raw_train.targets, train_per_class, num_classes, rng)
    val_idx = _balanced_indices(raw_test.targets, val_per_class, num_classes, rng)

    lenet_cfg = LeNetConfig(
        num_conv_layers=FIXED_LENET_CFG.num_conv_layers,
        filter_size=FIXED_LENET_CFG.filter_size,
        dropout=FIXED_LENET_CFG.dropout,
        filters=FIXED_LENET_CFG.filters,
        num_classes=num_classes, input_channels=3, input_size=32,
    )

    records: list[dict] = []
    for run in aug_runs():
        _set_seed(seed + run.run_id)
        train_tf = _build_aug_transforms(32, run, train=True)
        val_tf = _build_aug_transforms(32, run, train=False)
        train_ds = CIFARSubset(raw_train, train_idx, train_tf)
        val_ds = CIFARSubset(raw_test, val_idx, val_tf)
        model = LeNet5(lenet_cfg)
        print(f"\n[aug-OED run {run.run_id}] {run.label}  "
              f"(pad={run.crop_padding}, flip={run.hflip_p}, rot={run.rotation_deg}, jit={run.color_jitter})")
        t0 = time.perf_counter()
        tr = train_with(
            model, train_ds, val_ds,
            epochs=epochs, batch_size=batch_size, lr=lr,
            device=device, num_workers=num_workers,
            log_every=max(1, epochs // 5),
        )
        dt = time.perf_counter() - t0
        acc = tr.final_val_accuracy(last_k)
        print(f"    -> val_acc = {acc:.4f}  [{dt:.1f}s]")
        records.append({
            "run_id": run.run_id, "label": run.label,
            "crop_padding": run.crop_padding, "hflip_p": run.hflip_p,
            "rotation_deg": run.rotation_deg, "color_jitter": run.color_jitter,
            "val_accuracy": acc, "val_acc_history": tr.val_accs, "duration_sec": dt,
        })

    accuracies = [r["val_accuracy"] for r in records]
    analysis = range_analysis(accuracies)
    print("\n" + _format_oed_with(AUG_FACTOR_DESCRIPTIONS, AUG_LEVELS, analysis))

    _write_oed_tables(
        records=records, analysis=analysis, levels_dict=AUG_LEVELS,
        factor_descriptions=AUG_FACTOR_DESCRIPTIONS,
        table2_path=out_dir / "aug_table2.csv",
        table3_path=out_dir / "aug_table3.csv",
        factor_columns=[
            ("crop_padding", "A"), ("hflip_p", "B"),
            ("rotation_deg", "C"), ("color_jitter", "D"),
        ],
    )
    _plot_oed_figures(
        analysis, out_dir, prefix="figure_aug_",
        factor_descriptions=AUG_FACTOR_DESCRIPTIONS, levels_dict=AUG_LEVELS,
        range_title="Range values - augmentation factors",
    )
    return {"records": records, "analysis": analysis}


# ---------------------------------------------------------------------------
# EXPERIMENT C -- Regularisation / training-hyperparameter OED
# ---------------------------------------------------------------------------

REG_FACTOR_DESCRIPTIONS: dict[str, str] = {
    "A": "Dropout rate",
    "B": "Weight decay",
    "C": "Label smoothing",
    "D": "Learning rate",
}

REG_LEVELS: dict[str, tuple] = {
    "A": (0.0, 0.2, 0.5),
    "B": (0.0, 1e-4, 1e-3),
    "C": (0.0, 0.05, 0.1),
    "D": (3e-4, 1e-3, 3e-3),
}


@dataclass
class RegRun:
    run_id: int
    levels: tuple[int, int, int, int]
    dropout: float
    weight_decay: float
    label_smoothing: float
    lr: float

    @property
    def label(self) -> str:
        return "".join(f"{f}{lv}" for f, lv in zip(FACTOR_NAMES, self.levels))


def reg_runs() -> list[RegRun]:
    out = []
    for i, levels in enumerate(L9_ARRAY, start=1):
        a, b, c, d = levels
        out.append(RegRun(
            run_id=i, levels=levels,
            dropout=REG_LEVELS["A"][a - 1],
            weight_decay=REG_LEVELS["B"][b - 1],
            label_smoothing=REG_LEVELS["C"][c - 1],
            lr=REG_LEVELS["D"][d - 1],
        ))
    return out


def run_regularisation_oed(
    *,
    data_dir: Path,
    out_dir: Path,
    train_per_class: int,
    val_per_class: int,
    num_classes: int,
    epochs: int,
    batch_size: int,
    last_k: int,
    device: torch.device,
    num_workers: int,
    seed: int,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    train_ds, val_ds, _ = build_cifar_splits(
        data_dir, image_size=32,
        train_per_class=train_per_class, val_per_class=val_per_class,
        num_classes=num_classes, seed=seed,
    )

    records: list[dict] = []
    for run in reg_runs():
        _set_seed(seed + run.run_id + 500)
        cfg = LeNetConfig(
            num_conv_layers=FIXED_LENET_CFG.num_conv_layers,
            filter_size=FIXED_LENET_CFG.filter_size,
            dropout=run.dropout,
            filters=FIXED_LENET_CFG.filters,
            num_classes=num_classes, input_channels=3, input_size=32,
        )
        model = LeNet5(cfg)
        print(f"\n[reg-OED run {run.run_id}] {run.label}  "
              f"(drop={run.dropout}, wd={run.weight_decay}, "
              f"ls={run.label_smoothing}, lr={run.lr})")
        t0 = time.perf_counter()
        tr = train_with(
            model, train_ds, val_ds,
            epochs=epochs, batch_size=batch_size, lr=run.lr,
            weight_decay=run.weight_decay, label_smoothing=run.label_smoothing,
            device=device, num_workers=num_workers,
            log_every=max(1, epochs // 5),
        )
        dt = time.perf_counter() - t0
        acc = tr.final_val_accuracy(last_k)
        print(f"    -> val_acc = {acc:.4f}  [{dt:.1f}s]")
        records.append({
            "run_id": run.run_id, "label": run.label,
            "dropout": run.dropout, "weight_decay": run.weight_decay,
            "label_smoothing": run.label_smoothing, "lr": run.lr,
            "val_accuracy": acc, "val_acc_history": tr.val_accs, "duration_sec": dt,
        })

    accuracies = [r["val_accuracy"] for r in records]
    analysis = range_analysis(accuracies)
    print("\n" + _format_oed_with(REG_FACTOR_DESCRIPTIONS, REG_LEVELS, analysis))

    _write_oed_tables(
        records=records, analysis=analysis, levels_dict=REG_LEVELS,
        factor_descriptions=REG_FACTOR_DESCRIPTIONS,
        table2_path=out_dir / "reg_table2.csv",
        table3_path=out_dir / "reg_table3.csv",
        factor_columns=[
            ("dropout", "A"), ("weight_decay", "B"),
            ("label_smoothing", "C"), ("lr", "D"),
        ],
    )
    _plot_oed_figures(
        analysis, out_dir, prefix="figure_reg_",
        factor_descriptions=REG_FACTOR_DESCRIPTIONS, levels_dict=REG_LEVELS,
        range_title="Range values - regularisation factors",
    )
    return {"records": records, "analysis": analysis}


# ---------------------------------------------------------------------------
# Generic OED plotting / table helpers (work for any factor dictionary)
# ---------------------------------------------------------------------------

def _format_oed_with(factor_desc: dict[str, str], levels_dict: dict[str, tuple],
                     analysis: OEDAnalysis) -> str:
    """Like ``format_analysis_text`` but with custom factor labels."""
    lines: list[str] = []
    header = f"{'Index':<8}" + "".join(f"{f:<12}" for f in FACTOR_NAMES)
    lines.append(header); lines.append("-" * len(header))
    for i in range(3):
        lines.append(f"K{i + 1:<7}" + "".join(
            f"{analysis.factor_summaries[f].sums[i]:<12.4f}" for f in FACTOR_NAMES))
    for i in range(3):
        lines.append(f"Kbar{i + 1:<4}" + "".join(
            f"{analysis.factor_summaries[f].means[i]:<12.4f}" for f in FACTOR_NAMES))
    lines.append(f"{'R':<8}" + "".join(
        f"{analysis.factor_summaries[f].range_value:<12.4f}" for f in FACTOR_NAMES))
    lines.append(f"Ranking: {' > '.join(analysis.ranking)}")
    best = "".join(
        f"{f}{lv} ({levels_dict[f][lv - 1]})"
        for f, lv in zip(FACTOR_NAMES, analysis.best_combination)
    )
    lines.append(f"Best combination (by K-bar): {best}")
    lines.append("Factor legend: " + ", ".join(
        f"{f}={factor_desc[f]}" for f in FACTOR_NAMES))
    return "\n".join(lines)


def _write_oed_tables(
    *, records, analysis, levels_dict, factor_descriptions,
    table2_path: Path, table3_path: Path,
    factor_columns: list[tuple[str, str]],
) -> None:
    table2_path.parent.mkdir(parents=True, exist_ok=True)
    with table2_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "label"] + [c[0] for c in factor_columns] + ["val_accuracy"])
        for r in records:
            w.writerow([r["run_id"], r["label"]] +
                       [r[c[0]] for c in factor_columns] +
                       [f"{r['val_accuracy']:.4f}"])
    with table3_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index"] + list(FACTOR_NAMES))
        for i in range(3):
            w.writerow([f"K{i + 1}"] + [
                f"{analysis.factor_summaries[fn].sums[i]:.4f}" for fn in FACTOR_NAMES])
        for i in range(3):
            w.writerow([f"Kbar{i + 1}"] + [
                f"{analysis.factor_summaries[fn].means[i]:.4f}" for fn in FACTOR_NAMES])
        w.writerow(["R"] + [
            f"{analysis.factor_summaries[fn].range_value:.4f}" for fn in FACTOR_NAMES])
        w.writerow(["ranking"] + [" > ".join(analysis.ranking), "", "", ""])
        best = "".join(f"{f}{lv}" for f, lv in zip(FACTOR_NAMES, analysis.best_combination))
        w.writerow(["best_combination_by_Kbar"] + [best, "", "", ""])
        w.writerow(["legend"] + [", ".join(
            f"{f}={factor_descriptions[f]}" for f in FACTOR_NAMES), "", "", ""])


def _plot_oed_figures(
    analysis: OEDAnalysis,
    out_dir: Path,
    *,
    prefix: str,
    factor_descriptions: dict[str, str],
    levels_dict: dict[str, tuple],
    range_title: str,
) -> None:
    # Range bar chart.
    labels = [factor_descriptions[f] for f in FACTOR_NAMES]
    values = [analysis.factor_summaries[f].range_value for f in FACTOR_NAMES]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, values, color="#4C72B0")
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{v:.4f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Range value R")
    ax.set_title(range_title)
    ax.set_ylim(0, max(values) * 1.25 if values else 1)
    plt.xticks(rotation=12, ha="right")
    fig.tight_layout()
    _savefig(fig, out_dir / f"{prefix}range_values.png")

    # Per-factor line plots.
    for factor in FACTOR_NAMES:
        summary = analysis.factor_summaries[factor]
        lvls = levels_dict[factor]

        def _fmt(v):
            if isinstance(v, float):
                return f"{v:g}"
            return str(v)

        x_labels = [_fmt(lvls[i]) for i in range(3)]
        y = list(summary.means)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(x_labels, y, marker="o", color="#4C72B0")
        for xi, yi in zip(x_labels, y):
            ax.annotate(f"{yi:.4f}", (xi, yi), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=9)
        ax.set_xlabel(factor_descriptions[factor])
        ax.set_ylabel("Validation accuracy (mean)")
        ax.set_title(f"{factor}: {factor_descriptions[factor]}")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        _savefig(fig, out_dir / f"{prefix}factor_{factor}.png")


# ---------------------------------------------------------------------------
# Combined "what actually matters" summary plot
# ---------------------------------------------------------------------------

def _load_arch_range_values(arch_results_json: Optional[Path]) -> Optional[dict[str, float]]:
    if arch_results_json is None or not arch_results_json.exists():
        return None
    try:
        data = json.loads(arch_results_json.read_text())
        analysis = data.get("analysis") or {}
        summaries = analysis.get("factor_summaries") or {}
        return {f: float(summaries[f]["range_value"]) for f in FACTOR_NAMES}
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


def _plot_what_matters(
    *,
    arch_r: Optional[dict[str, float]],
    aug_analysis: Optional[OEDAnalysis],
    reg_analysis: Optional[OEDAnalysis],
    path: Path,
) -> None:
    """Side-by-side R values across the three studies.

    Architecture factors: A=conv layers, B=kernel, C=dropout, D=filters.
    Augmentation:         A=crop,        B=flip,   C=rotation, D=jitter.
    Regularisation:       A=dropout,     B=wd,     C=ls,       D=lr.
    We plot each study's R values as a cluster; factor letters differ in
    meaning but the four bars per study are sorted by rank.
    """
    studies: list[tuple[str, list[float], list[str]]] = []
    if arch_r is not None:
        studies.append((
            "Architecture",
            [arch_r[f] for f in FACTOR_NAMES],
            [f"{f}: {FACTOR_DESCRIPTIONS[f]}" for f in FACTOR_NAMES],
        ))
    if aug_analysis is not None:
        studies.append((
            "Augmentation",
            [aug_analysis.factor_summaries[f].range_value for f in FACTOR_NAMES],
            [f"{f}: {AUG_FACTOR_DESCRIPTIONS[f]}" for f in FACTOR_NAMES],
        ))
    if reg_analysis is not None:
        studies.append((
            "Regularisation",
            [reg_analysis.factor_summaries[f].range_value for f in FACTOR_NAMES],
            [f"{f}: {REG_FACTOR_DESCRIPTIONS[f]}" for f in FACTOR_NAMES],
        ))
    if not studies:
        print("[warn] Nothing to plot in figure_what_matters (no study data).")
        return

    n_studies = len(studies)
    fig, axes = plt.subplots(1, n_studies, figsize=(5 * n_studies, 4.5), sharey=True)
    if n_studies == 1:
        axes = [axes]
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    for ax, (title, values, labels) in zip(axes, studies):
        bars = ax.bar(labels, values, color=colors[:len(values)])
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_title(title)
        ax.set_ylabel("Range value R")
        ax.tick_params(axis="x", rotation=25, labelsize=8)
    global_max = max(max(s[1]) for s in studies)
    for ax in axes:
        ax.set_ylim(0, global_max * 1.25)
        for lbl in ax.get_xticklabels():
            lbl.set_horizontalalignment("right")
    fig.suptitle("What actually matters?  R values across studies")
    fig.tight_layout()
    _savefig(fig, path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extensions to the baseline CIFAR-10 OED study.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--experiment",
                   choices=["sample-efficiency", "augmentation-oed",
                            "regularisation-oed", "all"],
                   default="all")
    p.add_argument("--data-dir", default="/kaggle/working/cifar_data",
                   help="CIFAR-10 download cache.")
    p.add_argument("--output-dir", default="/kaggle/working/cifar_extensions_results")
    p.add_argument("--arch-results-json", default="/kaggle/working/cifar_results/results.json",
                   help="Path to the cifar_cnn.py results.json; its architecture R "
                        "values are overlaid on the combined summary plot.")

    # Dataset.
    p.add_argument("--num-classes", type=int, default=10)
    p.add_argument("--train-per-class", type=int, default=100,
                   help="Used by the two OED studies (aug, reg).")
    p.add_argument("--val-per-class", type=int, default=50)

    # Sample-efficiency study.
    p.add_argument("--sample-sizes", type=int, nargs="+",
                   default=[25, 50, 100, 200, 400],
                   help="Training-set sizes (per class) swept in --experiment sample-efficiency.")
    p.add_argument("--skip-pretrained", action="store_true",
                   help="Skip the ImageNet-pretrained ResNet-50 (saves one download).")

    # Training.
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--last-k", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--num-workers", type=int, default=2)

    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    cfg = _parse_args(argv)
    out_dir = Path(cfg.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(cfg.device)
    print(f"[info] device={device}  epochs={cfg.epochs}  "
          f"train_per_class={cfg.train_per_class}  val_per_class={cfg.val_per_class}")

    all_results: dict = {"config": vars(cfg)}

    if cfg.experiment in ("sample-efficiency", "all"):
        print("\n########## EXPERIMENT A: SAMPLE EFFICIENCY ##########")
        res = run_sample_efficiency(
            data_dir=Path(cfg.data_dir), out_dir=out_dir,
            sample_sizes=cfg.sample_sizes, val_per_class=cfg.val_per_class,
            num_classes=cfg.num_classes, epochs=cfg.epochs,
            batch_size=cfg.batch_size, lr=cfg.lr, last_k=cfg.last_k,
            device=device, num_workers=cfg.num_workers, seed=cfg.seed,
            skip_pretrained=cfg.skip_pretrained,
        )
        all_results["sample_efficiency"] = res["records"]

    aug_analysis: Optional[OEDAnalysis] = None
    if cfg.experiment in ("augmentation-oed", "all"):
        print("\n########## EXPERIMENT B: AUGMENTATION-STRENGTH OED ##########")
        res = run_augmentation_oed(
            data_dir=Path(cfg.data_dir), out_dir=out_dir,
            train_per_class=cfg.train_per_class, val_per_class=cfg.val_per_class,
            num_classes=cfg.num_classes, epochs=cfg.epochs,
            batch_size=cfg.batch_size, lr=cfg.lr, last_k=cfg.last_k,
            device=device, num_workers=cfg.num_workers, seed=cfg.seed,
        )
        aug_analysis = res["analysis"]
        all_results["augmentation_oed"] = {
            "runs": res["records"],
            "analysis": _serialise_analysis(aug_analysis),
        }

    reg_analysis: Optional[OEDAnalysis] = None
    if cfg.experiment in ("regularisation-oed", "all"):
        print("\n########## EXPERIMENT C: REGULARISATION OED ##########")
        res = run_regularisation_oed(
            data_dir=Path(cfg.data_dir), out_dir=out_dir,
            train_per_class=cfg.train_per_class, val_per_class=cfg.val_per_class,
            num_classes=cfg.num_classes, epochs=cfg.epochs,
            batch_size=cfg.batch_size, last_k=cfg.last_k,
            device=device, num_workers=cfg.num_workers, seed=cfg.seed,
        )
        reg_analysis = res["analysis"]
        all_results["regularisation_oed"] = {
            "runs": res["records"],
            "analysis": _serialise_analysis(reg_analysis),
        }

    # Combined summary plot.
    if cfg.experiment == "all" or (aug_analysis is not None or reg_analysis is not None):
        arch_r = _load_arch_range_values(Path(cfg.arch_results_json) if cfg.arch_results_json else None)
        if arch_r is None:
            print(f"[info] Architecture R values not loaded from {cfg.arch_results_json!r}; "
                  "combined plot will omit that panel.")
        _plot_what_matters(
            arch_r=arch_r, aug_analysis=aug_analysis, reg_analysis=reg_analysis,
            path=out_dir / "figure_what_matters.png",
        )
        all_results["arch_range_values"] = arch_r

    save_json(all_results, out_dir / "extensions_results.json")
    print(f"\n[done] Results written to {out_dir.resolve()}")
    return 0


def _serialise_analysis(analysis: OEDAnalysis) -> dict:
    return {
        "ranking": analysis.ranking,
        "best_combination_by_Kbar": analysis.best_combination,
        "factor_summaries": {
            k: {"sums": v.sums, "means": v.means, "range_value": v.range_value,
                "best_level": v.best_level}
            for k, v in analysis.factor_summaries.items()
        },
    }


if __name__ == "__main__":
    raise SystemExit(main())
