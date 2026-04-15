"""
================================================================================
BUG CASE 10 — BATCH_SIZE: 32 → 4096 (Large-Batch Generalization Degradation)
================================================================================
Student   : Jack Sweeney
Component : Training Configuration
GenAI Label: BAD

INITIAL SELF-REFLECTION (shallow — GenAI labeled Bad):
    "A bigger batch size means we process more images at once. This
    should make training faster and more stable since we're averaging
    gradients over more examples. I think accuracy might even improve."

GENAI SOCRATIC GUIDANCE:
    "You assume larger batches always help. What does gradient noise
    at batch=32 do to the optimizer's trajectory in the loss landscape?
    At batch=4096 (larger than your whole dataset), what happens to
    that noise? There is a well-studied phenomenon called the sharp
    minima problem — what does it predict about large-batch training's
    generalization, and why?"

NEW REFLECTION (after guidance):
    "Small batches introduce stochastic noise that implicitly pushes
    the optimizer toward flat minima — wide loss valleys where small
    input changes produce small output changes, which generalize well.
    With batch=4096 > dataset size, one batch per epoch means nearly
    deterministic gradients, converging to sharp minima that generalize
    poorly. Per Keskar et al. 2017, this causes a consistent ~7%
    accuracy drop with normal-looking loss curves and no crash."

FIX: Change BATCH_SIZE = 4096 → BATCH_SIZE = 32
================================================================================
"""

import numpy as np
import math

np.random.seed(42)

# BUG: batch size larger than the training dataset
BATCH_SIZE   = 4096   # <-- BUG: should be 32
DATASET_SIZE = 400
NUM_CLASSES  = 10
INPUT_SIZE   = 32
LR           = 0.01
EPOCHS       = 20


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


def train_and_eval(batch_size, seed=0):
    np.random.seed(seed)

    # Class-separable dataset so the effect is measurable
    X = np.random.randn(DATASET_SIZE, INPUT_SIZE)
    y = np.random.randint(0, NUM_CLASSES, DATASET_SIZE)
    for c in range(NUM_CLASSES):
        X[y == c] += c * 0.4

    X_val = np.random.randn(200, INPUT_SIZE)
    y_val = np.random.randint(0, NUM_CLASSES, 200)
    for c in range(NUM_CLASSES):
        X_val[y_val == c] += c * 0.4

    W = np.random.randn(NUM_CLASSES, INPUT_SIZE) * 0.05
    b = np.zeros(NUM_CLASSES)

    effective_batch = min(batch_size, DATASET_SIZE)
    batches_per_epoch = max(1, math.ceil(DATASET_SIZE / effective_batch))

    for _ in range(EPOCHS):
        idx = np.random.permutation(DATASET_SIZE)
        for bi in range(batches_per_epoch):
            batch_idx = idx[bi * effective_batch:(bi + 1) * effective_batch]
            gW = np.zeros_like(W)
            gb = np.zeros_like(b)
            for i in batch_idx:
                probs         = softmax(W @ X[i] + b)
                one_hot       = np.zeros(NUM_CLASSES); one_hot[y[i]] = 1.0
                d             = probs - one_hot
                gW           += np.outer(d, X[i])
                gb           += d
            n   = len(batch_idx)
            W  -= LR * gW / n
            b  -= LR * gb / n

    correct = sum(np.argmax(softmax(W @ X_val[i] + b)) == y_val[i] for i in range(len(y_val)))
    return correct / len(y_val), batches_per_epoch


print("Bug Case 10 — BATCH_SIZE=4096 large-batch generalization degradation")
print("Initial reflection was labeled BAD by GenAI.")
print(f"Training dataset size: {DATASET_SIZE}")
print(f"Buggy BATCH_SIZE:      {BATCH_SIZE}  (larger than dataset!)")
print()

acc_correct, b_correct = train_and_eval(32)
acc_buggy,   b_buggy   = train_and_eval(BATCH_SIZE)  # <-- BUG

print(f"{'Config':<30} | {'Batches/epoch':>13} | {'Val Accuracy':>12} | Note")
print("-" * 75)
print(f"{'batch=32  (correct)':<30} | {b_correct:>13} | {acc_correct:>11.1%} | "
      f"many noisy steps → flat minima")
print(f"{'batch=4096 (buggy)':<30} | {b_buggy:>13} | {acc_buggy:>11.1%} | "
      f"1 step/epoch → sharp minima")
print()
print(f"Accuracy drop: {acc_correct - acc_buggy:.1%}  — no crash, silent degradation")
print()
print("On GPU with limited VRAM, batch=4096 would also raise:")
print("  RuntimeError: CUDA out of memory.")
print()
print("Fix: BATCH_SIZE = 4096 → BATCH_SIZE = 32")
