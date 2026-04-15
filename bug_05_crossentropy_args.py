"""
================================================================================
BUG CASE 5 — CrossEntropyLoss Arguments Swapped
================================================================================
Student   : Jasraj "Jay" Raval
Component : Training Loop
GenAI Label: BAD

INITIAL SELF-REFLECTION (shallow — GenAI labeled Bad):
    "I swapped the two arguments. I think PyTorch might get confused
    because the inputs are in the wrong order, and there might be some
    kind of error about the types or something not matching."

GENAI SOCRATIC GUIDANCE:
    "Your reflection identifies something is wrong but not what
    specifically breaks. Look at the exact API signature of
    CrossEntropyLoss: what are the required shapes and dtypes of each
    positional argument? What exact shape and dtype does the function
    receive as 'input', and what does it receive as 'target'?
    What specific validation does it perform on each?"

NEW REFLECTION (after guidance):
    "CrossEntropyLoss(input, target) requires: input=[B, C] float logits,
    target=[B] int class indices. By swapping to criterion(labels, outputs),
    the function receives labels=[B] int as 'input' and outputs=[B, C] float
    as 'target'. The 'input' must be 2D with shape [B, C] — receiving a 1D
    integer array fails the shape validation immediately. This is a hard
    crash on the first batch before any weight update occurs."

FIX: Change criterion(labels, outputs) → criterion(outputs, labels)
================================================================================
"""

import numpy as np


def cross_entropy_loss(input_logits, target_indices):
    """
    input_logits   : array of shape [B, C] — float, model outputs
    target_indices : array of shape [B]    — int, class labels 0..C-1
    """
    # Validate input shape
    if input_logits.ndim != 2:
        raise ValueError(
            f"cross_entropy_loss: 'input' must be 2D with shape [B, C], "
            f"but got shape {input_logits.shape} (ndim={input_logits.ndim}). "
            f"Did you accidentally swap 'outputs' and 'labels'?"
        )
    # Validate target shape
    if target_indices.ndim != 1:
        raise ValueError(
            f"cross_entropy_loss: 'target' must be 1D with shape [B], "
            f"but got shape {target_indices.shape}."
        )
    B, C = input_logits.shape
    if target_indices.shape[0] != B:
        raise ValueError(f"Batch size mismatch: input has {B}, target has {target_indices.shape[0]}")

    # Softmax + NLL
    shifted  = input_logits - input_logits.max(axis=1, keepdims=True)
    exp_     = np.exp(shifted)
    probs    = exp_ / exp_.sum(axis=1, keepdims=True)
    log_probs = np.log(probs[np.arange(B), target_indices] + 1e-12)
    return -log_probs.mean()


np.random.seed(42)
BATCH_SIZE  = 32
NUM_CLASSES = 10

# Correct types: outputs=[B, C] float, labels=[B] int
outputs = np.random.randn(BATCH_SIZE, NUM_CLASSES).astype(np.float32)
labels  = np.random.randint(0, NUM_CLASSES, (BATCH_SIZE,))

print("Bug Case 5 — CrossEntropyLoss arguments swapped")
print("Initial reflection was labeled BAD by GenAI.")
print()
print(f"outputs shape: {outputs.shape}  dtype: {outputs.dtype}  ← model logits [B, C]")
print(f"labels  shape: {labels.shape}   dtype: {labels.dtype}  ← class indices [B]")
print()
print("Correct:  cross_entropy_loss(outputs, labels)  → [B,C] float as input")
print("Buggy:    cross_entropy_loss(labels, outputs)  → [B] int as input — CRASH")
print()
print("Running buggy call — this will crash:\n")

# BUG: arguments are swapped
loss = cross_entropy_loss(labels, outputs)   # <-- BUG: should be (outputs, labels)

print("Should never reach this line.")
