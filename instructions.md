# Running the experiments from a Kaggle notebook

This repo contains **three scripts**, each producing the paper-style tables
and figures of a different study:

| Script | What it does | Approx. Kaggle-T4 time (full run) |
|---|---|---|
| `pokemon_cnn.py`       | Baseline L_9 OED study on 3 Pokémon types (~245 images)       | 20-40 min |
| `cifar_cnn.py`         | Same OED, rerun on a 10-class CIFAR-10 subset (100/class)     | 30-60 min |
| `cifar_extensions.py`  | 3 follow-up studies: sample efficiency (incl. pretrained ResNet), augmentation-OED, regularisation-OED, plus a combined "what-matters" plot | 1-2 h |

You can run them **independently**; the extensions script will load the
architecture R values from `cifar_cnn.py`'s `results.json` automatically
(falling back gracefully if it is absent).

> On your local machine the same commands work — drop the `/kaggle/working/`
> prefix and clone the repos wherever you like.

---

## 0. Create the notebook (shared setup)

1. Go to **kaggle.com -> Code -> New Notebook**.
2. **Settings -> Accelerator -> GPU T4 x2** (or any GPU). Leave **Internet** on.
3. You do *not* need to attach a Kaggle dataset — everything is cloned or
   downloaded into `/kaggle/working`.

### Cell 0 — clone the code and (for Part A only) the Pokémon dataset

```bash
!git clone --depth 1 https://github.com/samudraneel05/lowdatacnn.git /kaggle/working/lowdatacnn
# The Pokémon-by-type images:
!git clone --depth 1 https://github.com/rileynwong/pokemon-images-dataset-by-type.git /kaggle/working/pokemon-images-dataset-by-type
# Install pinned deps (Kaggle usually has them already):
!pip install -q -r /kaggle/working/lowdatacnn/requirements.txt
```

---

# Part A — Pokémon baseline (`pokemon_cnn.py`)

Trains **11 models** — 9 LeNet-5 L_9 OED variants + AlexNet + ResNet-50 — for
100 epochs on the fire / water / grass subset (~245 images, 3 classes).

```bash
!python /kaggle/working/lowdatacnn/pokemon_cnn.py \
    --data-dir /kaggle/working/pokemon-images-dataset-by-type \
    --output-dir /kaggle/working/results \
    --epochs 100
```

Outputs (all under `/kaggle/working/results/`):
`table2_results.csv`, `table3_range_analysis.csv`,
`figure4_range_values.png`, `figure5_num_conv_layers.png`,
`figure6_filter_size.png`, `figure7_dropout.png`,
`figure8_num_filters.png`, `figure_model_comparison.png`,
`results.json`.

---

# Part B — CIFAR-10 low-data rerun (`cifar_cnn.py`)

Runs the **same** L_9 OED architecture study on a deliberately small
CIFAR-10 subset: all 10 classes, 100 training images + 50 validation images
per class by default. CIFAR-10 is auto-downloaded via `torchvision` on
first run and cached in `--data-dir`.

```bash
!python /kaggle/working/lowdatacnn/cifar_cnn.py \
    --data-dir /kaggle/working/cifar_data \
    --output-dir /kaggle/working/cifar_results \
    --train-per-class 100 --val-per-class 50 \
    --epochs 100
```

Outputs (under `/kaggle/working/cifar_results/`) mirror Part A exactly:
`table2_results.csv`, `table3_range_analysis.csv`, `figure4_*`,
`figure5_*`–`figure8_*`, `figure_model_comparison.png`, `results.json`.

Comparing the two studies side-by-side tells you whether the architectural
factor ranking (paper finding: **B > C > A > D**) is dataset-specific or a
general low-data phenomenon.

---

# Part C — Follow-up studies (`cifar_extensions.py`)

Three extensions, each picked to answer a concrete research question.

| Experiment              | Answers                                              | CLI flag                                  |
|---|---|---|
| Sample efficiency       | *How much data does each architecture need?*         | `--experiment sample-efficiency`          |
| Augmentation-OED        | *How much of "what matters" is augmentation?*        | `--experiment augmentation-oed`           |
| Regularisation-OED      | *How much is dropout / weight decay / LS / LR?*      | `--experiment regularisation-oed`         |
| All three               | Combined "what-matters" plot at the end              | `--experiment all` **(default)**          |

### Cell — run everything

```bash
!python /kaggle/working/lowdatacnn/cifar_extensions.py \
    --experiment all \
    --data-dir /kaggle/working/cifar_data \
    --output-dir /kaggle/working/cifar_extensions_results \
    --arch-results-json /kaggle/working/cifar_results/results.json \
    --epochs 60 \
    --train-per-class 100 --val-per-class 50 \
    --sample-sizes 25 50 100 200 400
```

Key extra outputs:

- **`figure_sample_efficiency.png`** — accuracy vs. training-size on a log
  axis, one line per architecture (LeNet-5 / AlexNet / ResNet-50 scratch /
  ResNet-50 **ImageNet pretrained**). Directly answers *"how much data does
  each architecture need to catch up?"*.
- **`figure_what_matters.png`** — three side-by-side panels of range
  values R: architectural knobs (loaded from `cifar_cnn.py`'s JSON) vs.
  augmentation knobs vs. regularisation knobs. Directly answers *"which
  parameters are actually important?"*.
- Per-study paper-style outputs:
  `aug_table2.csv`, `aug_table3.csv`, `figure_aug_range_values.png`,
  `figure_aug_factor_{A,B,C,D}.png`;
  `reg_table2.csv`, `reg_table3.csv`, `figure_reg_range_values.png`,
  `figure_reg_factor_{A,B,C,D}.png`.
- `sample_efficiency.csv`, `extensions_results.json`.

### Smoke test (optional, ~3 min on CPU)

Before committing to a 1-2 h run, confirm the whole pipeline end-to-end:

```bash
!python /kaggle/working/lowdatacnn/cifar_extensions.py \
    --experiment all \
    --epochs 3 \
    --train-per-class 30 --val-per-class 20 \
    --sample-sizes 30 60 \
    --skip-pretrained --device cpu
```

---

# Displaying results inline

Put this in the final cell to see every figure and table without leaving
the notebook:

```python
from pathlib import Path
import pandas as pd
from IPython.display import Image, display, Markdown

for folder in ["results", "cifar_results", "cifar_extensions_results"]:
    out = Path(f"/kaggle/working/{folder}")
    if not out.exists():
        continue
    display(Markdown(f"## `{folder}/`"))
    for png in sorted(out.glob("*.png")):
        display(Markdown(f"### {png.name}"))
        display(Image(str(png)))
    for csv in sorted(out.glob("*.csv")):
        display(Markdown(f"### {csv.name}"))
        try:
            display(pd.read_csv(csv))
        except Exception as e:
            display(Markdown(f"_(could not parse: {e})_"))
```

---

# Useful shortcuts

| Goal | Script | Flag |
|---|---|---|
| Fewer epochs | any | `--epochs 20` |
| Skip AlexNet / ResNet baselines | `pokemon_cnn.py`, `cifar_cnn.py` | `--skip-baselines` |
| Run just one L_9 row | `pokemon_cnn.py`, `cifar_cnn.py` | `--only-oed-run 1` |
| Skip the ImageNet pretrained ResNet (save download) | `cifar_extensions.py` | `--skip-pretrained` |
| Run only one follow-up study | `cifar_extensions.py` | `--experiment sample-efficiency` (or `augmentation-oed`, or `regularisation-oed`) |
| Force CPU | any | `--device cpu` |
| Change random seed | any | `--seed 7` |
| Regenerate Pokémon Tables III + Figs 4-8 from paper numbers, no training | `pokemon_cnn.py` | `--demo` |

Every script supports `--help`.

---

# Saving the results

Kaggle wipes `/kaggle/working` when your session ends **unless** you commit
the notebook:

1. Click **Save Version** → **Save & Run All (Commit)**, or
2. download each output folder as a zip from the **Output** tab.

---

# Troubleshooting

- **`FileNotFoundError: Could not find class sub-directories ['fire', 'water', 'grass']`**
  The Pokémon dataset clone in Cell 0 failed. Re-run it and confirm
  `ls /kaggle/working/pokemon-images-dataset-by-type` shows the type folders.

- **First CIFAR run is slow**
  The initial `torchvision` CIFAR-10 download (~170 MB) is one-time and
  caches under `--data-dir`. Subsequent runs reuse it.

- **ResNet-50 pretrained download fails**
  Torchvision pulls weights from `https://download.pytorch.org`. Kaggle's
  default image allows this, but if you've disabled Internet in Settings
  you'll need to re-enable it. Use `--skip-pretrained` to bypass entirely.

- **`--demo` works on Pokémon but the real run stalls around chance**
  That's the paper's core finding: over-parameterised architectures lose
  in the small-data regime. See the range analysis — the *smallest*
  LeNet-5 configurations dominate.

- **Training is slow on CPU**
  Make sure you enabled the GPU accelerator (step 0.2). Drop
  `--batch-size` to 16 if you're memory-constrained.
