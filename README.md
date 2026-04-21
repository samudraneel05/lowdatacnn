# lowdatacnn

Low-data CNN architecture study: how should convolutional networks be made
when the training set has only a few hundred images?

Three self-contained scripts, each producing paper-style tables and figures:

- **`pokemon_cnn.py`** — baseline L_9(3^4) Taguchi orthogonal-array study
  on 3-class Pokémon type recognition (~245 images), plus AlexNet and
  ResNet-50 from-scratch baselines.
- **`cifar_cnn.py`** — same L_9 study, rerun on a deliberately small
  CIFAR-10 subset (10 classes, 100 train / 50 val per class by default).
- **`cifar_extensions.py`** — three follow-up studies:
  sample-efficiency curves (including an ImageNet-pretrained ResNet-50),
  an L_9 study of **augmentation strength**, an L_9 study of
  **regularisation / training hyper-parameters**, and a combined
  "what-actually-matters" comparison that puts architecture, augmentation 
  and regularisation R-values side by side.

How to run: see [`instructions.md`](instructions.md) for the Kaggle
notebook recipe.