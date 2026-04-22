"""
================================================================================
BUG CASE 4 — Dropout: p=0.5 → p=1.0 (Complete Neuron Zeroing)
================================================================================
Student   : Jack Sweeney
Component : Model Architecture
GenAI Label: BAD

INITIAL SELF-REFLECTION (shallow — GenAI labeled Bad):
    "I increased the dropout probability. This will regularize the
    model more aggressively. It might hurt accuracy a bit since more
    neurons are being dropped, but the model should still learn something."

GENAI SOCRATIC GUIDANCE:
    "You treat dropout as a spectrum where more is stronger regularization.
    But consider the boundary case: what does p=1.0 mean mathematically?
    What value does every neuron output during a training forward pass?
    What does the final Linear layer receive, and what gradient flows
    back through its weight matrix when that input is all zeros?"

NEW REFLECTION (after guidance):
    "Dropout(p=1.0) sets every activation to zero with probability 1.0 —
    meaning ALL outputs are zero on every training forward pass.
    The final Linear layer receives an all-zero vector every batch.
    Its output equals only its bias vector. Loss stays at ln(10) ≈ 2.3026
    (cross-entropy of a uniform distribution over 10 classes) forever.
    The gradient of the weight matrix W = input^T × delta = 0^T × delta = 0.
    No weight update ever occurs. Total learning failure, not regularization."

FIX: Change Dropout(p=1.0) → Dropout(p=0.5)
================================================================================
"""

import numpy as np
import math

np.random.seed(42)


def dropout(x, p, training=True):
    """Apply dropout with probability p of zeroing each element."""
    if not training or p == 0.0:
        return x
    if p == 1.0:
        return np.zeros_like(x)   # ALL neurons zeroed
    mask = (np.random.rand(*x.shape) > p).astype(np.float32)
    return x * mask / (1.0 - p)


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


def cross_entropy_loss(probs, label):
    return -np.log(probs[label] + 1e-12)


# Mini network: 32 input → 32 hidden → 10 output
INPUT, HIDDEN, OUTPUT = 32, 32, 10
W1 = np.random.randn(HIDDEN, INPUT)  * 0.1
b1 = np.zeros(HIDDEN)
W2 = np.random.randn(OUTPUT, HIDDEN) * 0.1
b2 = np.zeros(OUTPUT)
LR = 0.01

# BUG: dropout probability = 1.0 (should be 0.5)
DROPOUT_P = 1.0   # <-- BUG: should be 0.5

print("Bug Case 4 — Dropout(p=1.0) complete neuron zeroing")
print("Initial reflection was labeled BAD by GenAI.")
print(f"Expected loss at random chance: ln(10) = {math.log(10):.4f}")
print()
print(f"{'Epoch':>5} | {'Loss':>8} | {'Acc':>6} | {'All-zero after dropout?'}")
print("-" * 52)

for epoch in range(1, 9):
    correct = total = 0
    epoch_loss = 0.0

    for _ in range(50):
        x     = np.random.randn(INPUT)
        label = np.random.randint(0, OUTPUT)

        # Forward
        h_pre    = W1 @ x + b1
        h_act    = np.tanh(h_pre)
        h_drop   = dropout(h_act, DROPOUT_P, training=True)   # ALL ZEROS when p=1.0
        out      = W2 @ h_drop + b2
        probs    = softmax(out)
        loss     = cross_entropy_loss(probs, label)
        epoch_loss += loss

        # Backward
        one_hot = np.zeros(OUTPUT); one_hot[label] = 1.0
        dout    = probs - one_hot
        dW2     = np.outer(dout, h_drop)    # h_drop is all zeros → dW2 is all zeros
        db2     = dout
        dh_drop = W2.T @ dout
        dh_act  = dh_drop * (1 - h_act**2)  # tanh deriv
        dW1     = np.outer(dh_act, x)
        db1     = dh_act

        W1 -= LR * dW1
        b1 -= LR * db1
        W2 -= LR * dW2   # dW2 is all zeros — W2 never changes
        b2 -= LR * db2

        pred = np.argmax(probs)
        correct += (pred == label)
        total   += 1

    avg_loss = epoch_loss / 50
    acc      = correct / total
    all_zero = True   # always True when p=1.0

    print(f"{epoch:>5} | {avg_loss:>8.4f} | {acc:>5.1%} | {all_zero}")

print()
print(f"Loss is stuck at ≈ {math.log(10):.4f}. Accuracy at ≈ 10% random chance.")
print("W2 weight norm never changed from initialization.")
print("Fix: DROPOUT_P = 1.0 → DROPOUT_P = 0.5")
