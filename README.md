# CTMC Cell Classification

A deep learning pipeline for classifying cell microscopy images using ResNet18 and PyTorch.

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

Make sure the data file is here:

```text
data/CTMCV1.zip
```

The script will automatically extract it.

---

## Running

Run the training script:

```bash
python src/cmsc472_final.py
```