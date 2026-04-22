"""
================================================================================
BUG CASE 1 — Normalize: 3-Channel Image → Single-Value Mean/Std
================================================================================
Student   : Jasraj "Jay" Raval
Component : Data Preprocessing
GenAI Label: BAD

INITIAL SELF-REFLECTION (shallow — GenAI labeled Bad):
    "I changed the normalize values. I think something might break
    because the numbers are different now, but I'm not sure what."

GENAI SOCRATIC GUIDANCE:
    "Your reflection doesn't identify what contract Normalize enforces
    between its arguments and the input tensor. What property of the
    image tensor determines how many values mean and std must have?
    Think about what 'channels' means in the context of RGB images
    and how normalization is applied per channel."

NEW REFLECTION (after guidance):
    "Normalize applies output[c] = (input[c] - mean[c]) / std[c] for
    each channel c. RGB images have 3 channels so mean and std must
    each be length 3. Providing a single scalar violates this contract —
    when the code tries to index mean[1] and mean[2] for the G and B
    channels, it fails with an IndexError because the list only has one
    element."

FIX: Change mean=[0.5] and std=[0.5] to mean=[0.5, 0.5, 0.5] and std=[0.5, 0.5, 0.5]
================================================================================
"""

import numpy as np


def normalize(image_chw, mean, std):
    """
    Normalize an image tensor of shape [C, H, W].
    mean and std must each have one value per channel.
    """
    c = image_chw.shape[0]
    if len(mean) != c or len(std) != c:
        raise ValueError(
            f"mean and std must each have {c} elements (one per channel), "
            f"but got mean={mean} (len={len(mean)}) and std={std} (len={len(std)})"
        )
    result = np.zeros_like(image_chw, dtype=np.float32)
    for i in range(c):
        result[i] = (image_chw[i] - mean[i]) / std[i]
    return result


# Simulate a [3, 64, 64] RGB image (channels-first)
np.random.seed(42)
fake_image = np.random.rand(3, 64, 64).astype(np.float32)

print("Bug Case 1 — Normalize channel mismatch")
print("Initial reflection was labeled BAD by GenAI.")
print(f"Image shape: {fake_image.shape}  (3 channels: R, G, B)")
print()

# BUG: mean and std are single-element lists — should be [0.5, 0.5, 0.5]
mean = [0.5]          # <-- BUG: only 1 element for a 3-channel image
std  = [0.5]          # <-- BUG

print(f"Buggy  mean={mean}, std={std}  (1 element, need 3)")
print("Calling normalize — this will crash:\n")

# CRASHES HERE — ValueError
result = normalize(fake_image, mean, std)

print("Should never reach this line.")
