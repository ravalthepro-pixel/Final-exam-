"""
================================================================================
BUG CASE 6 — FC Hidden Layer: 256 → 512 (Downstream Dimension Mismatch)
================================================================================
Student   : Jack Sweeney
Component : Model Architecture
GenAI Label: BAD

INITIAL SELF-REFLECTION (shallow — GenAI labeled Bad):
    "I made the hidden layer bigger which should give the model more
    capacity. A wider layer might help with accuracy. I don't think
    this will break anything since it's just a size change."

GENAI SOCRATIC GUIDANCE:
    "You're thinking about this as an isolated change, but in a
    Sequential classifier each layer's output size is the next layer's
    input size. What does FC1 now output? What does FC2 expect as its
    input? What are the exact tensor shapes, and what happens when
    you try to multiply them?"

NEW REFLECTION (after guidance):
    "FC1 now outputs [B, 512]. FC2's weight matrix has shape [256, 10],
    meaning it expects an input of size 256. Matrix multiplication
    [B, 512] × [256, 10] fails because inner dimensions 512 ≠ 256.
    This is a RuntimeError on the first forward pass. The fix is either
    revert FC1 output to 256, or update FC2 input to 512."

FIX: Either revert Linear(8192, 256) OR update FC2 to Linear(512, 10).
================================================================================
"""

import numpy as np


def linear(x, W, b):
    """
    x : [in_features]
    W : [out_features, in_features]
    b : [out_features]
    """
    if x.shape[0] != W.shape[1]:
        raise ValueError(
            f"Linear layer shape mismatch: input size is {x.shape[0]} "
            f"but weight matrix expects {W.shape[1]}. "
            f"(Cannot multiply [{x.shape[0]}] by [{W.shape[1]}, {W.shape[0]}])"
        )
    return W @ x + b


np.random.seed(42)

FLATTEN_SIZE = 128 * 8 * 8   # = 8192
FC1_OUT_BUG  = 512            # <-- BUG: should be 256
FC2_IN       = 256            # FC2 still expects 256 — never updated
NUM_CLASSES  = 10

# Weight matrices
W_fc1 = np.random.randn(FC1_OUT_BUG, FLATTEN_SIZE) * 0.01  # [512, 8192]
b_fc1 = np.zeros(FC1_OUT_BUG)
W_fc2 = np.random.randn(NUM_CLASSES, FC2_IN) * 0.01         # [10, 256] ← still 256
b_fc2 = np.zeros(NUM_CLASSES)

# Simulate flattened conv output
flat_input = np.random.randn(FLATTEN_SIZE)

print("Bug Case 6 — FC hidden layer dimension mismatch")
print("Initial reflection was labeled BAD by GenAI.")
print()
print(f"Flattened conv output size : {FLATTEN_SIZE}")
print(f"FC1 weight shape           : {W_fc1.shape}  → outputs {FC1_OUT_BUG}")
print(f"FC2 weight shape           : {W_fc2.shape}  → expects {FC2_IN} as input")
print(f"Mismatch: FC1 outputs {FC1_OUT_BUG}, FC2 expects {FC2_IN}")
print()
print("Running forward pass — this will crash:\n")

# FC1 succeeds — outputs [512]
h = np.tanh(linear(flat_input, W_fc1, b_fc1))
print(f"FC1 output shape: {h.shape}")

# FC2 CRASHES — receives [512] but weight expects [256]
out = linear(h, W_fc2, b_fc2)   # <-- BUG: crashes here

print("Should never reach this line.")
