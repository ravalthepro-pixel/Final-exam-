"""
================================================================================
BUG CASE 2 — MaxPool kernel_size: 2 → 3 (Spatial Dimension Collapse)
================================================================================
Student   : Jack Sweeney
Component : Model Architecture
GenAI Label: BAD

INITIAL SELF-REFLECTION (shallow — GenAI labeled Bad):
    "I changed the pooling layer. The spatial size will probably be
    smaller now so something downstream might not match up. I think
    the model might not work correctly."

GENAI SOCRATIC GUIDANCE:
    "You noticed dimensions change but haven't computed them. What
    exact spatial size does each feature map become after MaxPool
    with kernel_size=3, stride=3, starting from 64? Work through
    all three blocks step by step. Then ask: what size does the
    Linear layer expect, and where does that number come from?"

NEW REFLECTION (after guidance):
    "Block 1: floor(64/3)=21. Block 2: floor(21/3)=7.
    Block 3: floor(7/3)=2. Flatten: 128 × 2 × 2 = 512. But
    Linear(8192, 256) has a weight matrix of shape [8192, 256].
    Multiplying [B, 512] by [8192, 256] is impossible — inner
    dimensions 512 ≠ 8192. Hard crash on first forward pass."

FIX: Restore kernel_size=2, stride=2 in all three MaxPool layers.
================================================================================
"""

import numpy as np


def maxpool2d(x, kernel_size, stride):
    """Applies 2D max pooling. x shape: [C, H, W]"""
    C, H, W = x.shape
    out_h = (H - kernel_size) // stride + 1
    out_w = (W - kernel_size) // stride + 1
    out = np.zeros((C, out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            region = x[:, i*stride:i*stride+kernel_size, j*stride:j*stride+kernel_size]
            out[:, i, j] = region.max(axis=(1, 2))
    return out


def linear_forward(x, W):
    """x: [in_features], W: [out_features, in_features] → [out_features]"""
    if x.shape[0] != W.shape[1]:
        raise ValueError(
            f"Linear layer weight mismatch: input has {x.shape[0]} features "
            f"but weight matrix expects {W.shape[1]}. "
            f"(Shapes cannot be multiplied: [{x.shape[0]}] vs [{W.shape[1]}, {W.shape[0]}])"
        )
    return W @ x


np.random.seed(42)

# Simulate one image going through the buggy CNN
x = np.random.rand(3, 64, 64).astype(np.float32)

print("Bug Case 2 — MaxPool kernel_size=3 spatial dimension collapse")
print("Initial reflection was labeled BAD by GenAI.")
print(f"Input shape: {x.shape}")
print()

# BUG: kernel_size=3, stride=3 instead of kernel_size=2, stride=2
KERNEL = 3   # <-- BUG: should be 2
STRIDE = 3   # <-- BUG: should be 2

x = np.random.rand(32, 64, 64)   # fake conv block 1 output: [32, 64, 64]
x = maxpool2d(x, KERNEL, STRIDE)
print(f"After Pool 1 (kernel={KERNEL}): {x.shape}  (expected [32, 32, 32])")

x2 = np.random.rand(64, x.shape[1], x.shape[2])
x2 = maxpool2d(x2, KERNEL, STRIDE)
print(f"After Pool 2 (kernel={KERNEL}): {x2.shape}  (expected [64, 16, 16])")

x3 = np.random.rand(128, x2.shape[1], x2.shape[2])
x3 = maxpool2d(x3, KERNEL, STRIDE)
print(f"After Pool 3 (kernel={KERNEL}): {x3.shape}  (expected [128, 8, 8])")

flattened_size = x3.shape[0] * x3.shape[1] * x3.shape[2]
expected_size  = 128 * 8 * 8
print(f"\nFlattened size: {flattened_size}  (expected {expected_size})")
print()

# Simulate the Linear layer weight matrix [256, 8192]
W_fc1 = np.random.rand(256, expected_size).astype(np.float32)
flat  = x3.flatten()

print(f"Linear layer expects input size: {W_fc1.shape[1]}")
print(f"Actual flattened input size:     {flat.shape[0]}")
print()
print("Running Linear forward — this will crash:\n")

# CRASHES HERE — ValueError
out = linear_forward(flat, W_fc1)

print("Should never reach this line.")
