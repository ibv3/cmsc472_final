# train.py

import json
import os
import shutil
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


# =========================================================
# Setup
# =========================================================

CHECKPOINT_DIR = "results/checkpoints"
RESULTS_DIR = "results"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 64
NUM_WORKERS = 2
NUM_EPOCHS = 20

LR_BACKBONE = 1e-4
LR_HEAD = 1e-3

SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# =========================================================
# Extract dataset
# =========================================================

import zipfile

with zipfile.ZipFile("data/CTMCV1.zip", "r") as zip_ref:
    zip_ref.extractall("data")

DATA_ROOT = Path("data/CTMCV1")


# =========================================================
# Label mapping
# =========================================================

mapping = {
    "U2O-S": "Human Bone Osteosarcoma",
    "RK-13": "Normal Rabbit Kidney",
    "PL1Ut": "Raccoon Uterus",
    "OK": "Opossum Kidney Cortex Proximal Tubule",
    "MDOK": "Madin-Darby Ovine Kidney",
    "MDBK": "Madin-Darby Bovine Kidney",
    "LLC-MK2": "Rhesus Monkey Kidney",
    "CV-1": "Normal African Green Monkey Kidney",
    "CRE-BAG2": "Albino Swiss Mouse Embryo",
    "BPAE": "Bovine Pulmonary Artery",
    "APM": "African Water Mongoose Skin",
    "A-549": "Male Human Lung Carcinoma",
    "A-10": "Embryonic Rat Thoracic Aorta",
    "3T3": "Albino Swiss Mouse Embryo",
}


# =========================================================
# Utilities
# =========================================================

def expand_id(path):
    parts = str(path).split("/")

    run_id = parts[-3]
    img_name = parts[-1]

    prefix, run_num = run_id.split("-run")

    return {
        "prefix": prefix,
        "run_num": run_num,
        "image_num": img_name[:-4]
    }


label2idx = {label: i for i, label in enumerate(sorted(mapping.keys()))}
idx2label = {i: label for label, i in label2idx.items()}

NUM_CLASSES = len(label2idx)


# =========================================================
# Transforms
# =========================================================

train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomCrop(224, padding=16),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

eval_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


# =========================================================
# Dataset
# =========================================================

class CTMC(Dataset):

    def __init__(self, mode='train', transform=None, every_nth=15):

        self.root_dir = DATA_ROOT / mode
        self.transform = transform

        all_paths = sorted([
            p for p in self.root_dir.rglob("img1/*")
            if p.is_file()
        ])

        self.image_paths = all_paths[::every_nth]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        img_path = self.image_paths[idx]

        image = Image.open(img_path).convert("RGB")

        label_str = expand_id(img_path)["prefix"]
        label = label2idx[label_str]

        if self.transform:
            image = self.transform(image)

        return image, label


# =========================================================
# Datasets / loaders
# =========================================================

train_dataset = CTMC(
    mode='train',
    transform=train_transform,
    every_nth=15
)

test_dataset = CTMC(
    mode='test',
    transform=eval_transform,
    every_nth=15
)

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


# =========================================================
# Class weights
# =========================================================

label_counts = Counter(
    expand_id(str(p))['prefix']
    for p in train_dataset.image_paths
)

counts = torch.tensor(
    [label_counts[idx2label[i]] for i in range(NUM_CLASSES)],
    dtype=torch.float
)

class_weights = counts.sum() / (NUM_CLASSES * counts)
class_weights = class_weights / class_weights.sum() * NUM_CLASSES


# =========================================================
# Model
# =========================================================

def build_model(num_classes=NUM_CLASSES):

    model = models.resnet18(
        weights=models.ResNet18_Weights.DEFAULT
    )

    in_features = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, num_classes)
    )

    return model.to(DEVICE)


model = build_model()


# =========================================================
# Loss / optimizer
# =========================================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights.to(DEVICE)
)

head_params = list(model.fc.parameters())

head_ids = set(id(p) for p in head_params)

backbone_params = [
    p for p in model.parameters()
    if id(p) not in head_ids
]

optimizer = optim.Adam([
    {
        "params": backbone_params,
        "lr": LR_BACKBONE,
        "weight_decay": 1e-4,
    },
    {
        "params": head_params,
        "lr": LR_HEAD,
        "weight_decay": 1e-4,
    }
])

scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=NUM_EPOCHS
)


# =========================================================
# Train / eval helpers
# =========================================================

def train_epoch(model, loader):

    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for imgs, labels in loader:

        imgs = imgs.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(imgs)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        total_loss += loss.item() * imgs.size(0)

        preds = outputs.argmax(dim=1)

        correct += (preds == labels).sum().item()
        total += imgs.size(0)

    return total_loss / total, correct / total


def eval_epoch(model, loader):

    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():

        for imgs, labels in loader:

            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(imgs)

            loss = criterion(outputs, labels)

            total_loss += loss.item() * imgs.size(0)

            preds = outputs.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += imgs.size(0)

    return total_loss / total, correct / total


# =========================================================
# Training loop
# =========================================================

history = {
    "train_loss": [],
    "train_acc": [],
    "test_loss": [],
    "test_acc": [],
}

best_acc = 0.0

for epoch in range(NUM_EPOCHS):

    train_loss, train_acc = train_epoch(model, train_loader)

    test_loss, test_acc = eval_epoch(model, test_loader)

    scheduler.step()

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["test_loss"].append(test_loss)
    history["test_acc"].append(test_acc)

    print(
        f"Epoch {epoch+1:02d}/{NUM_EPOCHS}"
        f" | train acc={train_acc:.4f}"
        f" | test acc={test_acc:.4f}"
    )

    if test_acc > best_acc:

        best_acc = test_acc

        torch.save(
            model.state_dict(),
            os.path.join(CHECKPOINT_DIR, "best_model.pt")
        )


# =========================================================
# Evaluation
# =========================================================

model.load_state_dict(
    torch.load(
        os.path.join(CHECKPOINT_DIR, "best_model.pt"),
        map_location=DEVICE
    )
)

model.eval()

all_labels = []
all_preds = []
all_probs = []

with torch.no_grad():

    for imgs, labels in test_loader:

        imgs = imgs.to(DEVICE)

        outputs = model(imgs)

        probs = F.softmax(outputs, dim=1)

        preds = outputs.argmax(dim=1)

        all_labels.append(labels.numpy())
        all_preds.append(preds.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

all_labels = np.concatenate(all_labels)
all_preds = np.concatenate(all_preds)
all_probs = np.concatenate(all_probs)

accuracy = (all_labels == all_preds).mean()

macro_f1 = f1_score(
    all_labels,
    all_preds,
    average='macro'
)

macro_auroc = roc_auc_score(
    all_labels,
    all_probs,
    multi_class='ovr',
    average='macro'
)

print(f"\nAccuracy: {accuracy:.4f}")
print(f"Macro F1: {macro_f1:.4f}")
print(f"Macro AUROC: {macro_auroc:.4f}")


# =========================================================
# Save outputs for figures.py
# =========================================================

np.save("results/all_labels.npy", all_labels)
np.save("results/all_preds.npy", all_preds)
np.save("results/all_probs.npy", all_probs)

np.save("results/train_loss.npy", history["train_loss"])
np.save("results/test_loss.npy", history["test_loss"])
np.save("results/train_acc.npy", history["train_acc"])
np.save("results/test_acc.npy", history["test_acc"])

with open("results/label_map.json", "w") as f:
    json.dump(idx2label, f, indent=2)

print("\nSaved outputs for figures.py")
