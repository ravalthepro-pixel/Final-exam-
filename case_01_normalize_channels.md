# Bug Case 1 — Normalize: 3-Channel List → Single Scalar
**Student:** Jasraj "Jay" Raval  
**Component:** Data Preprocessing (`train.py`)  
**GenAI Label:** ✅ Good  

---

## What Was Changed

### Original Code (Working)
```python
transforms.Normalize(
    mean=[0.5, 0.5, 0.5],
    std=[0.5, 0.5, 0.5]
)
```

### Buggy Code
```python
transforms.Normalize(
    mean=[0.5],
    std=[0.5]
)
```

---

## Error Produced

```
ValueError: mean must have 1 element if it is not a sequence, got 1 elements
# or at runtime on first batch:
RuntimeError: input tensor and mean/std must have the same number of channels
```

The `Normalize` transform expects one mean and std value **per channel**. 
RGB images have 3 channels (R, G, B). Providing a single-element list 
causes a shape mismatch when PyTorch tries to apply the normalization.

---

## Initial Self-Reflection

> "I changed the mean and std from 3-element lists to single scalars. I think 
> this might cause a shape mismatch since the image tensor has 3 channels, but 
> I'm not sure exactly where it will crash."

**GenAI Label:** Good  
**GenAI Guidance:** "Your reflection correctly identifies the channel mismatch. 
PyTorch's Normalize expects sequences matching the number of input channels. 
Proceed with your reasoning."

---

## Root Cause Explanation

`torchvision.transforms.Normalize` applies the formula:

```
output[channel] = (input[channel] - mean[channel]) / std[channel]
```

It iterates over channels using the index of the `mean` and `std` lists.
When those lists contain only 1 element but the image tensor has 3 channels,
PyTorch cannot broadcast the normalization correctly.

**Tensor shape contract violated:**
```
Input:  [B, 3, 64, 64]   ← 3 channels
mean:   [1]               ← expects [3]
std:    [1]               ← expects [3]
```

---

## Fix

Restore `mean` and `std` to 3-element lists matching the number of RGB channels:

```python
transforms.Normalize(
    mean=[0.5, 0.5, 0.5],   # one value per channel: R, G, B
    std=[0.5, 0.5, 0.5]
)
```

---

## Lesson Learned

Preprocessing transforms have implicit contracts about tensor dimensions.
The number of elements in `mean` and `std` must always equal the number of
image channels. This is easy to miss when copy-pasting normalization code
from grayscale examples (which use single scalars) into RGB pipelines.
