# Running `pokemon_cnn.py` from a Kaggle notebook

This walks through producing the tables and figures on Kaggle.

> If you are on your local machine instead of Kaggle, the exact same commands work - just drop the `/kaggle/working/` prefix and clone the two repos wherever you like.

---

## 1. Create the notebook

1. Go to **kaggle.com -> Code -> New Notebook**.
2. In the right-hand sidebar: **Settings -> Accelerator -> GPU T4 x2** (or any GPU). Leave **Internet** on (default).
3. You do *not* need to attach a Kaggle dataset - we clone everything fresh into `/kaggle/working`.

---

## 2. Cell 1: clone the code and the dataset

Paste this into the first cell and run it.

```bash
!git clone --depth 1 https://github.com/samudraneel05/lowdatacnn.git /kaggle/working/lowdatacnn
!git clone --depth 1 https://github.com/rileynwong/pokemon-images-dataset-by-type.git /kaggle/working/pokemon-images-dataset-by-type

# Sanity check: the three paper classes should each have ~50-100 images.
!ls /kaggle/working/pokemon-images-dataset-by-type | head
!echo "fire:  $(ls /kaggle/working/pokemon-images-dataset-by-type/fire  | wc -l)"
!echo "water: $(ls /kaggle/working/pokemon-images-dataset-by-type/water | wc -l)"
!echo "grass: $(ls /kaggle/working/pokemon-images-dataset-by-type/grass | wc -l)"
```

---

## 3. Cell 2: install deps

Kaggle already ships recent `torch`, `torchvision`, `numpy`, `matplotlib` and `pillow`, so this is usually a no-op - but it guarantees version pins.

```bash
!pip install -q -r /kaggle/working/lowdatacnn/requirements.txt
```

---

## 4. Cell 3: full paper recreation (100 epochs)

This trains **11 models** (9 LeNet-5 OED variants + AlexNet + ResNet-50) for 100 epochs each on the ~200-image fire/water/grass subset, exactly as in the paper. Expect ~20-40 min on a T4, longer on a P100, much longer on CPU.

```bash
!python /kaggle/working/lowdatacnn/pokemon_cnn.py \
    --data-dir /kaggle/working/pokemon-images-dataset-by-type \
    --output-dir /kaggle/working/results \
    --epochs 100
```

The script prints:

- A one-line summary per epoch (only at 20 %, 40 %, ... intervals).
- The final validation accuracy (averaged over the last 20 epochs, matching the paper) for each run.
- The full range-analysis table (paper Table III) at the end.

---

## 5. Cell 4: display the figures inline

```python
from pathlib import Path
from IPython.display import Image, display, Markdown

out = Path("/kaggle/working/results")
for png in sorted(out.glob("*.png")):
    display(Markdown(f"### {png.name}"))
    display(Image(str(png)))

# Print the CSV tables too.
import pandas as pd
display(Markdown("### Table II"))
display(pd.read_csv(out / "table2_results.csv"))
display(Markdown("### Table III"))
display(pd.read_csv(out / "table3_range_analysis.csv"))
```

---

## Useful shortcuts

| Goal | Command |
|------|---------|
| Quick-iterate (fewer epochs) | `--epochs 20` |
| Skip AlexNet / ResNet | `--skip-baselines` |
| Run just one OED row (e.g. distribute across sessions) | `--only-oed-run 1` |
| Force CPU (debug) | `--device cpu` |
| Change random seed | `--seed 7` |
| Change LeNet input size | `--lenet-image-size 96` |

Run `python pokemon_cnn.py --help` for the full list.

---

## Saving the results

Kaggle wipes `/kaggle/working` when your session ends **unless** you commit the notebook. Two options:

1. Click **Save Version** (top-right) -> **Save & Run All (Commit)** - everything under `/kaggle/working` is snapshotted and available as the notebook's output.
2. Or download `/kaggle/working/results/` as a zip from the **Output** tab.

---

## Troubleshooting

- **`FileNotFoundError: Could not find class sub-directories ['fire', 'water', 'grass']`**
  The dataset clone in Cell 1 failed or landed elsewhere. Re-run Cell 1 and confirm the three `ls ... | wc -l` lines print non-zero counts.
- **Training is slow / OOM on CPU**
  Make sure you enabled the GPU accelerator (step 1.2). Drop `--batch-size` to 16 if you are memory-constrained.
- **`--demo` works but the real run stays at ~0.33 val accuracy**
  This *is* the paper's finding: "the deeper the model, the lower the accuracy" on a dataset this small (see Section IV). Run 1 (A1B1C1D1 = original LeNet-5) is typically the best. Keep `--epochs 100` and `--last-k 20`; the accuracy averages stabilise in the last 20 epochs.
