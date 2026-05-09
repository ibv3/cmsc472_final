# CTMC Cell Classification

A deep learning pipeline for classifying cell microscopy images using ResNet18 and PyTorch.

Expected runtime: 2-3 hours

---

## Features

- ResNet18 transfer learning
- Data augmentation
- Hyperparameter experiments
- AUROC / F1 evaluation
- Confusion matrices and ROC curves
- Density-group performance analysis

---

## Installation

Clone the repository:

```bash
git clone https://github.com/ibv3/cmsc472_final.git
cd cmsc472_final
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Dataset

Download the data from https://web.archive.org/web/20260304154036/https://motchallenge.net/data/CTMC-v1/

Make sure the data file is here:

```text
data/CTMCV1.zip
```

The scripts will automatically extract it.

---

## Running

Run the full training script (basically the same as notebook)

```bash
python3 src/cmsc472_final.py
```


Alternatively, to run just the models:

```bash
python3 src/train.py
```

And then to generate the figures:

```bash
python3 experiments/figures.py
```