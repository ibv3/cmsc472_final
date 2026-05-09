'''
Note: in our workflow we used Colab ipynb. Please run that if possible.

Link: https://colab.research.google.com/drive/1zUJXLJom6zfIQgJWrlvKQwK1g0dctXJT?authuser=2#scrollTo=E8JhX0S-JdyI

Here is the same workflow for base Python.
Run everything in the root directory. Run this file by python3 src/cmsc472_final.py
'''

import os
import copy
import re
import shutil
import subprocess
from collections import Counter
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from PIL import Image

from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

CHECKPOINT_DIR  = "/results/models/checkpoints"
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pt")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

subprocess.run(["unzip", "data/CTMCV1.zip", "-d", "data"])

SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


''' Setup '''

mapping = {
    "U2O-S": "Human Bone Osteosarcoma",
    "RK-13": "Normal Rabbit Kidney",
    "PL1Ut": "Raccoon Uterus",
    "OK": "Opossum Kidney Cortex Proximal Tubule",
    "MDOK": "Madin-Darby Ovine Kidney",
    "MDBK": "Madin-Darby Bovine Kidney",
    "LLC-MK2": "Rhesus Monkey Kidney",
    "CV-1": "Normal African Green Monkey Kidney",
    "CRE-BAG2": "Albino Swiss Mouse Embryo Moloney Murine Leukemia Virus Transfected Cells",
    "BPAE": "Bovine Pulmonary Artery",
    "APM": "African Water Mongoose Skin",
    "A-549": "Male Human Lung Carcinoma",
    "A-10": "Embryonic Rat Thoracic Aorta Medial Layer",
    "3T3": "Albino Swiss Mouse Embryo",
}



''' Visualization '''
# Returns mapped name, run number, and frame number
def expand_id(path):
    parts = path.strip().split("/")

    run_id = parts[-3]
    img_name = parts[-1]

    prefix, run_num = run_id.split("-run")
    name = mapping.get(prefix, prefix)
    img_num = img_name[:-4]

    return {
        "prefix": prefix,
        "name": name,
        "run_num": run_num,
        "image_num": img_num
    }

# Plots a single frame (give the full file path)
def plot_frame(filename):
  img = mpimg.imread(filename)
  plt.imshow(img)
  plt.axis('off')
  title_holder = expand_id(filename)
  plt.title(f"{title_holder['name']} Run {title_holder['run_num']} (Frame {title_holder['image_num']})")
  plt.show()

plot_frame("/data/CTMCV1/train/3T3-run01/img1/000001.jpg")
plot_frame("/data/CTMCV1/train/MDOK-run01/img1/000001.jpg")


''' Data Preparation '''

# Augmentation for training, resize-only for test/val
train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomCrop(224, padding=16),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

eval_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

class CTMC(Dataset):

    def __init__(self, mode='train', transform=None, every_nth=10):
        self.root_dir = Path(f'/data/CTMCV1/{mode}')
        self.transform = transform

        # collect all image paths
        # filter only files
        all_paths = sorted([
            p for p in self.root_dir.rglob("img1/*") if p.is_file()
        ])
        # keep every nth frame (to save training time/power)
        self.image_paths = all_paths[::every_nth]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label_str = expand_id(str(img_path))['prefix']
        label = label2idx[label_str]

        if self.transform:
            image = self.transform(image)

        return image, label

'''
Label encoding
'''
label2idx = {label: i for i, label in enumerate(sorted(mapping.keys()))}
idx2label = {i: label for label, i in label2idx.items()}
NUM_CLASSES = len(label2idx)

print("Label -> index mapping:")
for label, idx in label2idx.items():
    print(f"  {idx:2d}  {label:10s}  {mapping[label]}")

'''
Check for class imbalance
'''

train_dataset_raw = CTMC(mode='train', transform=None)

label_counts = Counter(
    expand_id(str(p))['prefix'] for p in train_dataset_raw.image_paths
)

print(f"{'Label':<12} {'Full name':<52} {'Frames':>7}  {'% of total':>10}")
print("-" * 85)
total = sum(label_counts.values())
for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
    print(f"{label:<12} {mapping[label]:<52} {count:>7,}  {100*count/total:>9.1f}%")
print(f"\nTotal training frames: {total:,}")

'''
Calculate class weights to use in the loss function
'''
counts = torch.tensor(
    [label_counts[idx2label[i]] for i in range(NUM_CLASSES)],
    dtype=torch.float
)
class_weights = counts.sum()/(NUM_CLASSES * counts)
class_weights = class_weights/class_weights.sum() * NUM_CLASSES

print("Class weights:")
for i, w in enumerate(class_weights):
    print(f"  {idx2label[i]:10s}  weight = {w:.4f}")

'''
DataLoaders for train and test
'''

BATCH_SIZE = 64
NUM_WORKERS = 2

train_dataset = CTMC(mode='train', transform=train_transform, every_nth=15)
test_dataset  = CTMC(mode='test',  transform=eval_transform,  every_nth=15)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)

print(f"Train batches : {len(train_loader):,}  ({len(train_dataset):,} frames)")
print(f"Test  batches : {len(test_loader):,}  ({len(test_dataset):,} frames)")

imgs, labels = next(iter(train_loader))
print(f"\nBatch image shape : {imgs.shape}")
print(f"Batch label range : {labels.min().item()} – {labels.max().item()}")

""" Building the Model"""

'''
Load Resnet
'''

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

def build_model(num_classes=NUM_CLASSES, freeze_backbone=False):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, num_classes)
    )

    return model.to(DEVICE)

model = build_model(freeze_backbone=False)

total   = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total params    : {total:,}")
print(f"Trainable params: {trainable:,}")

'''
Write loss, optimizer, and learning rate scheduler
'''

NUM_EPOCHS  = 20
LR_BACKBONE = 1e-4
LR_HEAD     = 1e-3

criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))

head_params     = list(model.fc.parameters())
head_param_ids  = set(id(p) for p in head_params)
backbone_params = [p for p in model.parameters() if id(p) not in head_param_ids]

optimizer = optim.Adam([
    {"params": backbone_params, "lr": LR_BACKBONE, "weight_decay": 1e-4},
    {"params": head_params, "lr": LR_HEAD, "weight_decay": 1e-4},
])

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

print(f"Criterion : {criterion}")
print(f"Optimizer : Adam  (backbone lr={LR_BACKBONE}, head lr={LR_HEAD})")
print(f"Scheduler : CosineAnnealingLR over {NUM_EPOCHS} epochs")




""" Model training

## Hyperparameters
* every_nth = how much data we train on (every nth frame)
* NUM_EPOCHS
* weight_decay in the optimizer
* Dropout(p=...) in model.fc
* label_smoothing in the criterion
"""

'''
Train helper (one epoch)
'''

def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += imgs.size(0)

    return total_loss / total, correct / total


def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += imgs.size(0)

    return total_loss / total, correct / total

'''
Training loop
'''
history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
best_test_acc = 0.0

for epoch in range(1, NUM_EPOCHS + 1):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
    test_loss,  test_acc  = eval_epoch(model, test_loader,  criterion)
    scheduler.step()

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["test_loss"].append(test_loss)
    history["test_acc"].append(test_acc)

    # Save checkpoint if this is the best test accuracy so far
    if test_acc > best_test_acc:
        best_test_acc = test_acc
        torch.save({
            "epoch":      epoch,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "test_acc":   test_acc,
            "test_loss":  test_loss,
        }, CHECKPOINT_PATH)
        flag = "  ✓ saved"
    else:
        flag = ""

    print(
        f"Epoch {epoch:02d}/{NUM_EPOCHS}"
        f"  train loss {train_loss:.4f}  acc {train_acc:.3f}"
        f"  |  test loss {test_loss:.4f}  acc {test_acc:.3f}"
        f"{flag}"
    )

print(f"\nBest test accuracy: {best_test_acc:.4f}")

'''
Plot training curves
'''
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
epochs = range(1, NUM_EPOCHS + 1)

ax1.plot(epochs, history["train_loss"], label="train")
ax1.plot(epochs, history["test_loss"],  label="test")
ax1.set_title("Loss")
ax1.set_xlabel("Epoch")
ax1.legend()

ax2.plot(epochs, history["train_acc"], label="train")
ax2.plot(epochs, history["test_acc"],  label="test")
ax2.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="50% target")
ax2.axhline(0.9, color="green", linestyle="--", linewidth=0.8, label="90% moonshot")
ax2.set_title("Accuracy")
ax2.set_xlabel("Epoch")
ax2.set_ylim(0, 1)
ax2.legend()

plt.tight_layout()
plt.savefig(os.path.join(CHECKPOINT_DIR, "training_curves.png"), dpi=150)
plt.show()

"""Hyperparameter Experiments"""

SEED   = 42   # base seed; each run offsets by run index
N_RUNS = 3    # runs per config — increase for tighter estimates, decrease to save compute

CONFIGS = [
    {
        "name":          "baseline",
        "every_nth":     15,
        "num_epochs":    5,
        "weight_decay":  1e-4,
        "dropout":       0.5,
        "label_smooth":  0.0,
    },
    {
        "name":          "every_nth",
        "every_nth":     10,
        "num_epochs":    5,
        "weight_decay":  1e-4,
        "dropout":       0.5,
        "label_smooth":  0.0,
    },
    {
        "name":          "num_epochs",
        "every_nth":     15,
        "num_epochs":    10,
        "weight_decay":  1e-4,
        "dropout":       0.5,
        "label_smooth":  0.0,
    },
    {
        "name":          "wd_1e-2",
        "every_nth":     15,
        "num_epochs":    5,
        "weight_decay":  1e-2,
        "dropout":       0.5,
        "label_smooth":  0.0,
    },
    {
        "name":          "dropout_0.1",
        "every_nth":     15,
        "num_epochs":    5,
        "weight_decay":  1e-4,
        "dropout":       0.1,
        "label_smooth":  0.0,
    },
    {
        "name":          "label_smooth_0.25",
        "every_nth":     15,
        "num_epochs":    5,
        "weight_decay":  1e-4,
        "dropout":       0.5,
        "label_smooth":  0.25,
    },
]

exp_results = []  # one entry per config, each with per-run metrics

for cfg in CONFIGS:
    print(f"\n{'='*60}")
    print(f"Running config: {cfg['name']}  ({N_RUNS} run(s))")
    print(f"  every_nth={cfg['every_nth']}, epochs={cfg['num_epochs']}, "
          f"wd={cfg['weight_decay']}, dropout={cfg['dropout']}, "
          f"label_smooth={cfg['label_smooth']}")
    print('='*60)

    run_accs, run_f1s, run_aucs = [], [], []
    best_run_acc   = -1
    best_ckpt_path = os.path.join(CHECKPOINT_DIR, f"best_{cfg['name']}.pt")

    for run_idx in range(N_RUNS):
        run_seed = SEED + run_idx
        torch.manual_seed(run_seed)
        np.random.seed(run_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(run_seed)

        print(f"  -- Run {run_idx+1}/{N_RUNS} (seed={run_seed})")

        # ── Data
        tr_ds = CTMC(mode='train', transform=train_transform, every_nth=cfg['every_nth'])
        te_ds = CTMC(mode='test',  transform=eval_transform,  every_nth=cfg['every_nth'])
        tr_ld = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True,
                           num_workers=NUM_WORKERS, pin_memory=True)
        te_ld = DataLoader(te_ds, batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=NUM_WORKERS, pin_memory=True)

        # ── Model
        cfg_model = build_model(freeze_backbone=False)
        in_feat   = cfg_model.fc[-1].in_features
        cfg_model.fc = nn.Sequential(
            nn.Dropout(p=cfg['dropout']),
            nn.Linear(in_feat, NUM_CLASSES)
        ).to(DEVICE)

        # ── Loss, optimiser, scheduler
        cfg_criterion = nn.CrossEntropyLoss(
            weight=class_weights.to(DEVICE),
            label_smoothing=cfg['label_smooth']
        )
        head_p   = list(cfg_model.fc.parameters())
        head_ids = set(id(p) for p in head_p)
        back_p   = [p for p in cfg_model.parameters() if id(p) not in head_ids]
        cfg_opt  = optim.Adam([
            {"params": back_p, "lr": LR_BACKBONE, "weight_decay": cfg['weight_decay']},
            {"params": head_p, "lr": LR_HEAD,     "weight_decay": cfg['weight_decay']},
        ])
        cfg_sched = optim.lr_scheduler.CosineAnnealingLR(cfg_opt, T_max=cfg['num_epochs'])

        # ── Training loop
        best_acc  = 0.0
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"best_{cfg['name']}_run{run_idx}.pt")

        for epoch in range(1, cfg['num_epochs'] + 1):
            tr_loss, tr_acc = train_epoch(cfg_model, tr_ld, cfg_criterion, cfg_opt)
            te_loss, te_acc = eval_epoch(cfg_model, te_ld, cfg_criterion)
            cfg_sched.step()
            if te_acc > best_acc:
                best_acc = te_acc
                torch.save(cfg_model.state_dict(), ckpt_path)
                flag = "  ✓"
            else:
                flag = ""
            print(f"    Epoch {epoch:02d}/{cfg['num_epochs']}"
                  f"  train {tr_acc:.3f}  test {te_acc:.3f}{flag}")

        # ── Evaluate best checkpoint for this run
        cfg_model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        cfg_model.eval()
        c_labels, c_preds, c_probs = [], [], []
        with torch.no_grad():
            for imgs, labels in te_ld:
                imgs = imgs.to(DEVICE)
                out  = cfg_model(imgs)
                c_labels.append(labels.cpu().numpy())
                c_preds.append(out.argmax(1).cpu().numpy())
                c_probs.append(F.softmax(out, dim=1).cpu().numpy())
        c_labels = np.concatenate(c_labels)
        c_preds  = np.concatenate(c_preds)
        c_probs  = np.concatenate(c_probs)

        run_acc = (c_labels == c_preds).mean()
        run_f1  = f1_score(c_labels, c_preds, average='macro')
        run_auc = roc_auc_score(c_labels, c_probs, multi_class='ovr', average='macro')
        run_accs.append(run_acc)
        run_f1s.append(run_f1)
        run_aucs.append(run_auc)
        print(f"    → run acc={run_acc:.4f}  macro-F1={run_f1:.4f}  macro-AUROC={run_auc:.4f}")

        # Keep checkpoint of the best run overall for potential further analysis
        if run_acc > best_run_acc:
            best_run_acc = run_acc
            import shutil
            shutil.copy(ckpt_path, best_ckpt_path)

    exp_results.append({
        "name":         cfg['name'],
        "acc_mean":     np.mean(run_accs),
        "acc_std":      np.std(run_accs),
        "f1_mean":      np.mean(run_f1s),
        "f1_std":       np.std(run_f1s),
        "auroc_mean":   np.mean(run_aucs),
        "auroc_std":    np.std(run_aucs),
        "cfg":          cfg,
    })
    print(f"  → mean acc={np.mean(run_accs):.4f} ± {np.std(run_accs):.4f}"
          f"  F1={np.mean(run_f1s):.4f} ± {np.std(run_f1s):.4f}"
          f"  AUROC={np.mean(run_aucs):.4f} ± {np.std(run_aucs):.4f}")

print("\nAll configurations complete.")

"""Experiment Results Comparison"""

# ── Summary table
col = 58
print(f"{'Config':<22} {'Accuracy':>16} {'Macro-F1':>16} {'Macro-AUROC':>16}")
print("-" * 72)
for r in exp_results:
    print(f"{r['name']:<22}"
          f"  {r['acc_mean']:.4f} ±{r['acc_std']:.4f}"
          f"  {r['f1_mean']:.4f} ±{r['f1_std']:.4f}"
          f"  {r['auroc_mean']:.4f} ±{r['auroc_std']:.4f}")
print(f"\n(N_RUNS={N_RUNS} per config; std dev across runs)")

# ── Bar chart with error bars
names  = [r['name']       for r in exp_results]
accs   = [r['acc_mean']   for r in exp_results]
f1s    = [r['f1_mean']    for r in exp_results]
aucs   = [r['auroc_mean'] for r in exp_results]
accs_e = [r['acc_std']    for r in exp_results]
f1s_e  = [r['f1_std']     for r in exp_results]
aucs_e = [r['auroc_std']  for r in exp_results]

x = np.arange(len(names))
w = 0.25
fig, ax = plt.subplots(figsize=(max(10, 3.0 * len(names)), 5))
ax.bar(x - w, accs,  w, yerr=accs_e,  capsize=4, label='Accuracy')
ax.bar(x,     f1s,   w, yerr=f1s_e,   capsize=4, label='Macro F1')
ax.bar(x + w, aucs,  w, yerr=aucs_e,  capsize=4, label='Macro AUROC')
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=30, ha='right')
ax.set_ylim(0, 1)
ax.set_ylabel('Score')
ax.set_title(f'Hyperparameter experiment comparison (mean ± std, N={N_RUNS} runs)')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(CHECKPOINT_DIR, 'hyp_exp_bar.png'), dpi=150)
plt.show()

"""# Evaluate Model"""
model.eval()

all_labels  = []
all_preds   = []
all_probs   = []

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(DEVICE)
        outputs = model(imgs)
        probs   = F.softmax(outputs, dim=1)
        preds   = outputs.argmax(dim=1)

        all_labels.append(labels.cpu().numpy())
        all_preds.append(preds.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

all_labels = np.concatenate(all_labels)
all_preds  = np.concatenate(all_preds)
all_probs  = np.concatenate(all_probs)

print(f"Test frames evaluated : {len(all_labels):,}")
print(f"Overall accuracy      : {(all_labels == all_preds).mean():.4f}")

'''
accuracy, F1, and AUROC
'''

per_class_auroc = roc_auc_score(
    all_labels, all_probs,
    multi_class='ovr', average=None
)

per_class_f1 = f1_score(all_labels, all_preds, average=None)

cm = confusion_matrix(all_labels, all_preds)
per_class_acc = cm.diagonal() / cm.sum(axis=1)

print(f"{'Idx':<4} {'Label':<10} {'Full Name':<42} {'Acc':>6} {'F1':>6} {'AUROC':>6}")
print("-" * 76)
for i in range(NUM_CLASSES):
    label = idx2label[i]
    print(
        f"{i:<4} {label:<10} {mapping[label]:<42} "
        f"{per_class_acc[i]:>6.3f} {per_class_f1[i]:>6.3f} {per_class_auroc[i]:>6.3f}"
    )

print("-" * 76)
macro_f1    = f1_score(all_labels, all_preds, average='macro')
macro_auroc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
overall_acc = (all_labels == all_preds).mean()
print(f"{'MACRO':>56} {overall_acc:>6.3f} {macro_f1:>6.3f} {macro_auroc:>6.3f}")

'''
confusion matrix
'''
import matplotlib.pyplot as plt
import seaborn as sns

cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

tick_labels = [idx2label[i] for i in range(NUM_CLASSES)]

fig, ax = plt.subplots(figsize=(14, 11))
sns.heatmap(
    cm_norm,
    annot=True, fmt=".2f",
    xticklabels=tick_labels,
    yticklabels=tick_labels,
    cmap="Blues",
    vmin=0, vmax=1,
    linewidths=0.4,
    ax=ax,
    annot_kws={"size": 8}
)
ax.set_xlabel("Predicted label", fontsize=12)
ax.set_ylabel("True label",      fontsize=12)
ax.set_title("Normalized Confusion Matrix (row = recall per class)", fontsize=13)
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(rotation=0,  fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(CHECKPOINT_DIR, "confusion_matrix.png"), dpi=150)
plt.show()

'''
AUROC Curves
'''
from sklearn.metrics import roc_curve
from sklearn.preprocessing import label_binarize

y_bin = label_binarize(all_labels, classes=list(range(NUM_CLASSES)))

fig, axes = plt.subplots(3, 5, figsize=(18, 11))
axes = axes.flatten()

for i in range(NUM_CLASSES):
    fpr, tpr, _ = roc_curve(y_bin[:, i], all_probs[:, i])
    axes[i].plot(fpr, tpr, lw=1.5, label=f"AUC={per_class_auroc[i]:.3f}")
    axes[i].plot([0,1],[0,1], 'k--', lw=0.8)
    axes[i].set_title(idx2label[i], fontsize=10)
    axes[i].set_xlabel("FPR", fontsize=8)
    axes[i].set_ylabel("TPR", fontsize=8)
    axes[i].legend(fontsize=8)
    axes[i].set_xlim([0,1])
    axes[i].set_ylim([0,1])

# Hide the two unused subplots (3x5 grid = 15, we have 14 classes)
axes[14].axis('off')

plt.suptitle(f"ROC Curves — Macro AUROC = {macro_auroc:.4f}", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(CHECKPOINT_DIR, "roc_curves.png"), dpi=150)
plt.show()

"""Frozen Backbone Baseline"""

# ── Build frozen model and train
frozen_model = build_model(freeze_backbone=True)

frozen_criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
frozen_optimizer = optim.Adam(frozen_model.fc.parameters(),
                              lr=LR_HEAD, weight_decay=1e-4)
frozen_scheduler = optim.lr_scheduler.CosineAnnealingLR(
    frozen_optimizer, T_max=NUM_EPOCHS)

frozen_ckpt = os.path.join(CHECKPOINT_DIR, 'frozen_baseline.pt')
best_frozen_acc = 0.0

print('Training frozen backbone baseline...')
for epoch in range(1, NUM_EPOCHS + 1):
    tr_loss, tr_acc = train_epoch(frozen_model, train_loader,
                                  frozen_criterion, frozen_optimizer)
    te_loss, te_acc = eval_epoch(frozen_model, test_loader, frozen_criterion)
    frozen_scheduler.step()
    if te_acc > best_frozen_acc:
        best_frozen_acc = te_acc
        torch.save(frozen_model.state_dict(), frozen_ckpt)
        flag = '  ✓'
    else:
        flag = ''
    print(f'  Epoch {epoch:02d}/{NUM_EPOCHS}'
          f'  train {tr_acc:.3f}  test {te_acc:.3f}{flag}')

# ── Evaluate frozen model
frozen_model.load_state_dict(torch.load(frozen_ckpt, map_location=DEVICE))
frozen_model.eval()
frz_labels, frz_preds, frz_probs = [], [], []
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(DEVICE)
        out  = frozen_model(imgs)
        frz_labels.append(labels.cpu().numpy())
        frz_preds.append(out.argmax(1).cpu().numpy())
        frz_probs.append(F.softmax(out, dim=1).cpu().numpy())
frz_labels = np.concatenate(frz_labels)
frz_preds  = np.concatenate(frz_preds)
frz_probs  = np.concatenate(frz_probs)

frz_acc  = (frz_labels == frz_preds).mean()
frz_f1   = f1_score(frz_labels, frz_preds, average='macro')
frz_auc  = roc_auc_score(frz_labels, frz_probs, multi_class='ovr', average='macro')

# ── Majority class baseline (kept for reference)
majority_class = np.bincount(all_labels).argmax()
majority_acc   = (all_labels == majority_class).mean()

# ── Comparison table
our_acc  = (all_labels == all_preds).mean()
our_f1   = f1_score(all_labels, all_preds, average='macro')
our_auc  = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')

print(f"\n{'Model':<25} {'Accuracy':>9} {'Macro-F1':>10} {'Macro-AUROC':>13}")
print('-' * 60)
print(f"{'Majority class':<25} {majority_acc:>9.4f} {'—':>10} {'—':>13}")
print(f"{'Frozen backbone':<25} {frz_acc:>9.4f} {frz_f1:>10.4f} {frz_auc:>13.4f}")
print(f"{'Fine-tuned (ours)':<25} {our_acc:>9.4f} {our_f1:>10.4f} {our_auc:>13.4f}")
print(f"\nGain over frozen baseline:  acc +{our_acc - frz_acc:.4f}  "
      f"F1 +{our_f1 - frz_f1:.4f}  AUROC +{our_auc - frz_auc:.4f}")

""" Density Analysis """


# Per-class average densities (cells/frame) from the CTMC-v1 challenge paper (Table 1).
# These are the ground-truth annotation counts averaged over all frames per cell line.
CLASS_DENSITY = {
    "3T3":     13.12,
    "A-10":     7.97,
    "A-549":    9.93,
    "APM":      6.10,
    "BPAE":    27.88,
    "CRE-BAG2":21.45,
    "CV-1":     3.42,
    "LLC-MK2": 10.76,
    "MDBK":    14.31,
    "MDOK":    11.84,
    "OK":      20.13,
    "PL1Ut":   12.55,
    "RK-13":    8.64,
    "U2O-S":   16.22,
}

# Map each class index to its density group
def density_group(label_idx):
    label = idx2label[label_idx]
    d = CLASS_DENSITY[label]
    if d < 10:
        return 'low (<10)'
    elif d <= 20:
        return 'medium (10-20)'
    else:
        return 'high (>20)'

# Assign each test sample to a group
groups = np.array([density_group(l) for l in all_labels])
group_order = ['low (<10)', 'medium (10-20)', 'high (>20)']

print(f"{'Density group':<18} {'N':>6} {'Accuracy':>9} {'Macro F1':>10} {'Macro AUROC':>13}")
print("-" * 60)

density_results = {}
for g in group_order:
    mask = groups == g
    if mask.sum() == 0:
        print(f"{g:<18} {'0':>6}  (no samples)")
        continue
    g_labels = all_labels[mask]
    g_preds  = all_preds[mask]
    g_probs  = all_probs[mask]

    g_acc  = (g_labels == g_preds).mean()
    # Only compute F1/AUROC if more than one class is present in the group
    present = np.unique(g_labels)
    if len(present) > 1:
        g_f1   = f1_score(g_labels, g_preds, average='macro', labels=present, zero_division=0)
        g_probs_present = g_probs[:, present]
        g_probs_present = g_probs_present / g_probs_present.sum(axis=1, keepdims=True)
        g_auc  = roc_auc_score(
            g_labels, g_probs_present,
            multi_class='ovr', average='macro',
            labels=present
        )
    else:
        g_f1, g_auc = float('nan'), float('nan')

    density_results[g] = {"acc": g_acc, "f1": g_f1, "auroc": g_auc, "n": mask.sum()}
    print(f"{g:<18} {mask.sum():>6} {g_acc:>9.4f} {g_f1:>10.4f} {g_auc:>13.4f}")

# ── Bar chart
valid_groups = [g for g in group_order if g in density_results]
x = np.arange(len(valid_groups))
w = 0.25
accs_d  = [density_results[g]['acc']   for g in valid_groups]
f1s_d   = [density_results[g]['f1']    for g in valid_groups]
aucs_d  = [density_results[g]['auroc'] for g in valid_groups]

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(x - w, accs_d, w, label='Accuracy')
ax.bar(x,     f1s_d,  w, label='Macro F1')
ax.bar(x + w, aucs_d, w, label='Macro AUROC')
ax.set_xticks(x)
ax.set_xticklabels(valid_groups)
ax.set_ylim(0, 1)
ax.set_ylabel('Score')
ax.set_title('Model performance by cell density group')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(CHECKPOINT_DIR, 'density_analysis.png'), dpi=150)
plt.show()

# ── Per-class density breakdown
print("\nPer-class detail:")
print(f"{'Label':<10} {'Density':>8} {'Group':<18} {'Acc':>6} {'F1':>6} {'AUROC':>6}")
print("-" * 62)
for i in range(NUM_CLASSES):
    label = idx2label[i]
    d     = CLASS_DENSITY[label]
    grp   = density_group(i)
    mask  = all_labels == i
    if mask.sum() == 0:
        continue
    cl_acc = (all_labels[mask] == all_preds[mask]).mean()
    print(f"{label:<10} {d:>8.2f} {grp:<18} {cl_acc:>6.3f} "
          f"{per_class_f1[i]:>6.3f} {per_class_auroc[i]:>6.3f}")
