"""
================================================================================
BUG CASE 8 — RandomRotation: ±10° → ±180° (Label Semantic Corruption)
================================================================================
Student   : Jack Sweeney
Component : Data Preprocessing / Augmentation
GenAI Label: BAD

INITIAL SELF-REFLECTION (shallow — GenAI labeled Bad):
    "Rotating images more aggressively will make training harder.
    The model might get worse accuracy because the images look more
    distorted. I think this counts as too much augmentation."

GENAI SOCRATIC GUIDANCE:
    "You describe a quantitative effect but miss a qualitative one.
    Think about what a label means for a hand sign image. If you rotate
    an ASL digit '6' by 150 degrees, is the label still semantically
    correct? What does the training objective optimize when a label
    is semantically wrong? How does this differ from simply making
    the task harder?"

NEW REFLECTION (after guidance):
    "This is label corruption, not just harder augmentation. Rotating
    an ASL hand sign 150° produces an image that no longer corresponds
    to the signed digit. The model is trained with the same label '6'
    for both the canonical gesture and its 180° rotation — contradictory
    evidence no architecture can resolve. This is data poisoning through
    augmentation. Validation accuracy drops ~23% because the model learns
    incorrect features."

FIX: Change RandomRotation(degrees=180) → RandomRotation(degrees=10)
================================================================================
"""

import numpy as np
import math

np.random.seed(42)

# ── Simulate training with label-corrupting augmentation ─────────────────────

def rotate_label_validity(angle_deg):
    """
    For ASL hand signs, rotations beyond ~15° destroy semantic meaning.
    Returns True if the label is still semantically valid after rotation.
    """
    return abs(angle_deg) <= 15


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


def train_epoch(W, b, data_x, data_y, lr, max_rotation_deg, n_steps=200):
    """Train one epoch with the given max rotation. Returns avg loss."""
    total_loss = 0.0
    for _ in range(n_steps):
        idx   = np.random.randint(len(data_x))
        x_raw = data_x[idx].copy()
        label = data_y[idx]

        # Apply rotation augmentation (simulated as feature perturbation)
        angle = np.random.uniform(-max_rotation_deg, max_rotation_deg)
        # Simulate rotation effect: large rotations scramble spatial features
        if abs(angle) > 15:
            # Heavy rotation scrambles the feature vector (semantic corruption)
            rotation_noise = np.random.randn(*x_raw.shape) * (abs(angle) / 180.0)
            x_aug = x_raw + rotation_noise
        else:
            # Small rotation: minor perturbation, label still valid
            x_aug = x_raw + np.random.randn(*x_raw.shape) * 0.05

        probs = softmax(W @ x_aug + b)
        loss  = -np.log(probs[label] + 1e-12)
        total_loss += loss

        # Gradient update
        one_hot = np.zeros(len(probs)); one_hot[label] = 1.0
        grad_b  = probs - one_hot
        grad_W  = np.outer(grad_b, x_aug)
        W -= lr * grad_W
        b -= lr * grad_b

    return total_loss / n_steps


def evaluate(W, b, data_x, data_y):
    correct = 0
    for x, y in zip(data_x, data_y):
        pred = np.argmax(softmax(W @ x + b))
        correct += (pred == y)
    return correct / len(data_y)


# Synthetic dataset: 10 classes, 64-dim features
N_TRAIN, N_VAL, N_FEAT, N_CLS = 400, 100, 64, 10
X_train = np.random.randn(N_TRAIN, N_FEAT)
y_train = np.random.randint(0, N_CLS, N_TRAIN)
X_val   = np.random.randn(N_VAL,   N_FEAT)
y_val   = np.random.randint(0, N_CLS, N_VAL)
# Make data class-separable so correct augmentation shows improvement
for c in range(N_CLS):
    X_train[y_train == c] += c * 0.3
    X_val[y_val == c]     += c * 0.3

LR     = 0.005
EPOCHS = 12

print("Bug Case 8 — RandomRotation(180°) label semantic corruption")
print("Initial reflection was labeled BAD by GenAI.")
print()

# Train with correct augmentation (±10°)
W_good = np.random.randn(N_CLS, N_FEAT) * 0.05
b_good = np.zeros(N_CLS)
np.random.seed(7)
for _ in range(EPOCHS):
    train_epoch(W_good, b_good, X_train, y_train, LR, max_rotation_deg=10)
acc_good = evaluate(W_good, b_good, X_val, y_val)

# Train with buggy augmentation (±180°) — BUG
W_bug = np.random.randn(N_CLS, N_FEAT) * 0.05
b_bug = np.zeros(N_CLS)
np.random.seed(7)
for _ in range(EPOCHS):
    train_epoch(W_bug,  b_bug,  X_train, y_train, LR, max_rotation_deg=180)  # <-- BUG
acc_bug  = evaluate(W_bug,  b_bug,  X_val, y_val)

print(f"{'Config':<35} | {'Val Accuracy':>12} | Note")
print("-" * 65)
print(f"{'Correct: RandomRotation(±10°)':<35} | {acc_good:>11.1%} | label stays valid")
print(f"{'Buggy:   RandomRotation(±180°)':<35} | {acc_bug:>11.1%} | label corrupted by rotation")
print()
print(f"Accuracy drop: {acc_good - acc_bug:.1%}  — no crash, silent degradation")
print()

# Show label validity for a sample of augmented images
print("Label validity check for 10 augmented training samples (label='6'):")
print(f"{'Sample':>7} | {'Rotation':>10} | {'Label valid?'}")
print("-" * 38)
for i in range(10):
    angle = np.random.uniform(-180, 180)   # BUG: up to 180
    valid = rotate_label_validity(angle)
    print(f"{i+1:>7} | {angle:>9.1f}° | {'YES' if valid else 'NO — sign semantically destroyed'}")

print()
print("Fix: RandomRotation(degrees=180) → RandomRotation(degrees=10)")
