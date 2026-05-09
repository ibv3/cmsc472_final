# figures.py

import json
import os

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.preprocessing import label_binarize


# =========================================================
# Directories
# =========================================================

RESULTS_DIR = "results"
FIGURE_DIR = "results/figures"

os.makedirs(FIGURE_DIR, exist_ok=True)


# =========================================================
# Load saved arrays
# =========================================================

all_labels = np.load("results/all_labels.npy")
all_preds = np.load("results/all_preds.npy")
all_probs = np.load("results/all_probs.npy")

train_loss = np.load("results/train_loss.npy")
test_loss = np.load("results/test_loss.npy")

train_acc = np.load("results/train_acc.npy")
test_acc = np.load("results/test_acc.npy")

with open("results/label_map.json", "r") as f:
    idx2label = json.load(f)

NUM_CLASSES = len(idx2label)


# =========================================================
# Training curves
# =========================================================

epochs = range(1, len(train_loss) + 1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(epochs, train_loss, label="train")
ax1.plot(epochs, test_loss, label="test")

ax1.set_title("Loss")
ax1.set_xlabel("Epoch")
ax1.legend()

ax2.plot(epochs, train_acc, label="train")
ax2.plot(epochs, test_acc, label="test")

ax2.set_title("Accuracy")
ax2.set_xlabel("Epoch")
ax2.set_ylim(0, 1)

ax2.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(FIGURE_DIR, "training_curves.png"),
    dpi=150
)

plt.show()


# =========================================================
# Confusion matrix
# =========================================================

cm = confusion_matrix(all_labels, all_preds)

cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

tick_labels = [
    idx2label[str(i)]
    for i in range(NUM_CLASSES)
]

fig, ax = plt.subplots(figsize=(14, 11))

sns.heatmap(
    cm_norm,
    annot=True,
    fmt=".2f",
    xticklabels=tick_labels,
    yticklabels=tick_labels,
    cmap="Blues",
    vmin=0,
    vmax=1,
    linewidths=0.4,
    annot_kws={"size": 8},
    ax=ax
)

ax.set_xlabel("Predicted label")
ax.set_ylabel("True label")

ax.set_title("Normalized Confusion Matrix")

plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

plt.tight_layout()

plt.savefig(
    os.path.join(FIGURE_DIR, "confusion_matrix.png"),
    dpi=150
)

plt.show()


# =========================================================
# ROC curves
# =========================================================

y_bin = label_binarize(
    all_labels,
    classes=list(range(NUM_CLASSES))
)

fig, axes = plt.subplots(3, 5, figsize=(18, 11))

axes = axes.flatten()

for i in range(NUM_CLASSES):

    fpr, tpr, _ = roc_curve(
        y_bin[:, i],
        all_probs[:, i]
    )

    axes[i].plot(
        fpr,
        tpr,
        lw=1.5
    )

    axes[i].plot(
        [0, 1],
        [0, 1],
        'k--',
        lw=0.8
    )

    axes[i].set_title(idx2label[str(i)])

    axes[i].set_xlim([0, 1])
    axes[i].set_ylim([0, 1])

    axes[i].set_xlabel("FPR")
    axes[i].set_ylabel("TPR")

for j in range(NUM_CLASSES, len(axes)):
    axes[j].axis('off')

plt.tight_layout()

plt.savefig(
    os.path.join(FIGURE_DIR, "roc_curves.png"),
    dpi=150
)

plt.show()

print("\nAll figures generated.")
