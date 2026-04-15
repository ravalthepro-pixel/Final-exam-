"""
Utility Functions — ASL CNN
AI 100 — Final Project
Authors: Jasraj "Jay" Raval & Jack Sweeney
"""
import os
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from PIL import Image
from torchvision import transforms


def plot_training_curves(train_losses, val_losses, train_accs, val_accs,
                          save_path="results/training_curves.png"):
    epochs = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor("#F8FAFC")
    for ax in (ax1, ax2):
        ax.set_facecolor("#F8FAFC"); ax.grid(True, alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax1.plot(epochs, train_losses, label="Train Loss", color="#2563EB", lw=2, marker="o", markersize=3)
    ax1.plot(epochs, val_losses,   label="Val Loss",   color="#DC2626", lw=2, marker="s", markersize=3, linestyle="--")
    ax1.set_title("Loss Curves", fontsize=13, fontweight="bold"); ax1.set_xlabel("Epoch"); ax1.legend()
    ax2.plot(epochs, [a*100 for a in train_accs], label="Train Acc", color="#2563EB", lw=2, marker="o", markersize=3)
    ax2.plot(epochs, [a*100 for a in val_accs],   label="Val Acc",   color="#DC2626", lw=2, marker="s", markersize=3, linestyle="--")
    ax2.set_title("Accuracy Curves", fontsize=13, fontweight="bold"); ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)"); ax2.set_ylim(0, 105); ax2.legend()
    fig.suptitle('ASL CNN Training Curves | Jasraj "Jay" Raval & Jack Sweeney', fontsize=11, y=1.01)
    plt.tight_layout(); plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  [Saved] {save_path}")


def plot_confusion_matrix(all_labels, all_preds, classes,
                           save_path="results/confusion_matrix.png"):
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes,
                linewidths=0.5, ax=ax, annot_kws={"fontsize": 11})
    ax.set_title('Confusion Matrix | Jasraj "Jay" Raval & Jack Sweeney', fontsize=13, fontweight="bold")
    ax.set_ylabel("True Label"); ax.set_xlabel("Predicted Label")
    acc = np.trace(cm) / cm.sum()
    ax.text(0.98, 0.01, f"Accuracy: {acc:.2%}", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    plt.tight_layout(); plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  [Saved] {save_path}")


def plot_sample_predictions(images, all_labels, all_preds, classes, n=10,
                             save_path="results/sample_predictions.png"):
    indices = np.random.choice(len(all_preds), min(n, len(all_preds)), replace=False)
    fig, axes = plt.subplots(2, 5, figsize=(15, 7))
    fig.patch.set_facecolor("#111827")
    for i, idx in enumerate(indices):
        ax = axes[i//5][i%5]
        img = images[idx].permute(1,2,0).numpy()
        img = (img * 0.5 + 0.5).clip(0, 1)
        ax.imshow(img); ax.axis("off")
        color = "#4ADE80" if all_preds[idx]==all_labels[idx] else "#F87171"
        ax.set_title(f"True: {classes[all_labels[idx]]}\nPred: {classes[all_preds[idx]]}",
                     color=color, fontsize=10, fontweight="bold")
    fig.suptitle('Sample Predictions | Green=Correct Red=Incorrect', fontsize=12, color="white", y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#111827"); plt.close()
    print(f"  [Saved] {save_path}")


def save_classification_report(all_labels, all_preds, classes,
                                 save_path="results/classification_report.txt"):
    report = classification_report(all_labels, all_preds, target_names=classes, digits=4)
    print("\nClassification Report:\n", report)
    with open(save_path, "w") as f:
        f.write(f'ASL CNN — Classification Report\nJasraj "Jay" Raval & Jack Sweeney\n\n{report}')
    print(f"  [Saved] {save_path}")
    return report


def load_model(model_class, weights_path, num_classes=10, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_class(num_classes=num_classes)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device).eval()
    return model, device


def predict_image(model, image_path, classes, device, img_size=64):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)), transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5]),
    ])
    tensor = transform(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)
        conf, idx = probs.max(1)
    return classes[idx.item()], conf.item()


def check_dataset_structure(data_dir, expected_classes=10):
    if not os.path.isdir(data_dir):
        print(f"ERROR: '{data_dir}' not found. Create it with class subfolders 0-9.")
        return False
    subdirs = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    total = 0
    for s in subdirs:
        imgs = [f for f in os.listdir(os.path.join(data_dir,s)) if f.lower().endswith((".jpg",".jpeg",".png"))]
        total += len(imgs)
        print(f"  Class '{s}': {len(imgs)} images")
    print(f"  Total: {total} images across {len(subdirs)} classes")
    return len(subdirs) == expected_classes
