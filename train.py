"""
ASL Hand Sign Digit Classification — Training Script
AI 100 — Final Project
Authors: Jasraj "Jay" Raval & Jack Sweeney

Intentional bugs were introduced to this file for the final project bug cases.
See bugs/README.md and bug_cases.xlsx for the full documentation of each case.

Current file: FIXED / WORKING version.
"""
import os, random, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from model import ASL_CNN
from utils import (plot_training_curves, plot_confusion_matrix,
                   plot_sample_predictions, save_classification_report,
                   check_dataset_structure)

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR     = "data"
RESULTS_DIR  = "results"
IMG_SIZE     = 64
BATCH_SIZE   = 32          # Bug Case 10: was changed to 4096
EPOCHS       = 20          # Bug Case 7:  was changed to 0
LR           = 0.001       # Bug Case 3:  was changed to 10.0
LR_STEP      = 7
LR_GAMMA     = 0.5
VAL_SPLIT    = 0.2
SEED         = 42
CHECKPOINT   = os.path.join(RESULTS_DIR, "best_model.pth")

def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)

set_seed(SEED)
os.makedirs(RESULTS_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── Transforms ────────────────────────────────────────────────────────────────
# Bug Case 1: Normalize mean/std was changed from [0.5,0.5,0.5] to [0.5]
# Bug Case 8: RandomRotation was changed from 10 to 180
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# ── Dataset ───────────────────────────────────────────────────────────────────
check_dataset_structure(DATA_DIR)
_full = datasets.ImageFolder(root=DATA_DIR)
n_val = int(len(_full) * VAL_SPLIT)
n_train = len(_full) - n_val
idx = list(range(len(_full)))
np.random.default_rng(SEED).shuffle(idx)
train_data = Subset(datasets.ImageFolder(DATA_DIR, transform=train_transform), idx[:n_train])
val_data   = Subset(datasets.ImageFolder(DATA_DIR, transform=val_transform),   idx[n_train:])
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
val_loader   = DataLoader(val_data,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
classes = _full.classes
print(f"Classes: {classes} | Train: {n_train} | Val: {n_val}")

# ── Model ─────────────────────────────────────────────────────────────────────
# Bug Case 2: MaxPool kernel changed from 2 to 3 (in model.py)
# Bug Case 4: Dropout p changed from 0.5 to 1.0 (in model.py)
# Bug Case 6: FC layer output changed from 256 to 512 (in model.py)
model     = ASL_CNN(num_classes=len(classes)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = StepLR(optimizer, step_size=LR_STEP, gamma=LR_GAMMA)
print(f"Parameters: {model.count_parameters():,}")

# ── Training Loop ─────────────────────────────────────────────────────────────
train_losses, val_losses, train_accs, val_accs = [], [], [], []
best_val_acc = 0.0

for epoch in range(1, EPOCHS + 1):
    model.train()
    rl, correct, total = 0.0, 0, 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()                          # Bug Case 9: this line was removed
        outputs = model(imgs)
        loss = criterion(outputs, labels)              # Bug Case 5: args were swapped
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        rl += loss.item() * imgs.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    train_losses.append(rl/total); train_accs.append(correct/total)

    model.eval()
    rl, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            rl += criterion(outputs, labels).item() * imgs.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
    val_losses.append(rl/total); val_accs.append(correct/total)
    scheduler.step()

    if val_accs[-1] > best_val_acc:
        best_val_acc = val_accs[-1]
        torch.save(model.state_dict(), CHECKPOINT)
    print(f"Epoch [{epoch:02d}/{EPOCHS}] Train: {train_accs[-1]:.4f} | Val: {val_accs[-1]:.4f}")

print(f"\nBest Val Accuracy: {best_val_acc:.4f}")

# ── Evaluation ────────────────────────────────────────────────────────────────
model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
model.eval()
all_preds, all_labels, all_imgs = [], [], []
with torch.no_grad():
    for imgs, labels in val_loader:
        preds = model(imgs.to(device)).argmax(1).cpu().numpy()
        all_preds.extend(preds); all_labels.extend(labels.numpy())
        all_imgs.append(imgs)
all_imgs = torch.cat(all_imgs)

plot_training_curves(train_losses, val_losses, train_accs, val_accs,
                     save_path=os.path.join(RESULTS_DIR, "training_curves.png"))
plot_confusion_matrix(all_labels, all_preds, classes,
                      save_path=os.path.join(RESULTS_DIR, "confusion_matrix.png"))
plot_sample_predictions(all_imgs, all_labels, all_preds, classes,
                        save_path=os.path.join(RESULTS_DIR, "sample_predictions.png"))
save_classification_report(all_labels, all_preds, classes,
                            save_path=os.path.join(RESULTS_DIR, "classification_report.txt"))
print("Done! All outputs saved to results/")
