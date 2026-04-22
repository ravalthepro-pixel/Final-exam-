# Bug Case 2 — MaxPool kernel_size: 2 → 3 (Spatial Dimension Collapse)
**Student:** Jack Sweeney  
**Component:** Model Architecture (`model.py`)  
**GenAI Label:** ❌ Bad → revised to deep understanding  

---

## What Was Changed

### Original Code (Working)
```python
nn.MaxPool2d(kernel_size=2, stride=2)
# Applied after each of 3 conv blocks
```

### Buggy Code
```python
nn.MaxPool2d(kernel_size=3, stride=3)
# Applied after each of 3 conv blocks
```

---

## Error Produced

```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (32x512 vs 8192x256)
```

This occurs at the first `nn.Linear(8192, 256)` layer in the classifier head,
because the flattened feature map is 512 instead of the expected 8192.

---

## Spatial Dimension Trace

With `kernel_size=2, stride=2` (original):
```
Input:        [B,   3, 64, 64]
Conv Block 1: [B,  32, 32, 32]   # 64 / 2 = 32
Conv Block 2: [B,  64, 16, 16]   # 32 / 2 = 16
Conv Block 3: [B, 128,  8,  8]   # 16 / 2 = 8
Flatten:      [B, 8192]          # 128 × 8 × 8 = 8192  ✓
```

With `kernel_size=3, stride=3` (buggy):
```
Input:        [B,   3, 64, 64]
Conv Block 1: [B,  32, 21, 21]   # floor(64 / 3) = 21
Conv Block 2: [B,  64,  7,  7]   # floor(21 / 3) = 7
Conv Block 3: [B, 128,  2,  2]   # floor( 7 / 3) = 2
Flatten:      [B, 512]           # 128 × 2 × 2 = 512  ✗ MISMATCH
```

The `nn.Linear(8192, 256)` weight matrix has shape `[8192, 256]`.
Multiplying a `[B, 512]` input against it is impossible — hard crash.

---

## Self-Reflection Process

**Initial Reflection (Bad):**
> "I made the pooling window bigger. I think the output will still work but 
> the spatial dimensions will be wrong — they won't divide evenly by 3 
> starting from 64."

**GenAI Label:** Bad  
**GenAI Socratic Guidance:** "You're on the right track about spatial dimensions 
changing, but think more carefully: what exact size does the feature map become 
after each block? Does an odd remainder cause a runtime error or silent 
precision loss?"

**Revised Reflection (after guidance):**
> "After Block 1: floor(64/3)=21, Block 2: floor(21/3)=7, Block 3: 
> floor(7/3)=2. Flatten gives 128×2×2=512, but Linear expects 8192. 
> This is a hard RuntimeError in the first forward pass — not a silent failure. 
> The bug is at the junction between the conv extractor and the classifier head, 
> where the flattened dimension is a fixed contract."

---

## Fix

Restore the original pooling configuration:

```python
nn.MaxPool2d(kernel_size=2, stride=2)
```

Or, if a different kernel size is desired, update the Linear layer to match:

```python
# If keeping kernel_size=3, stride=3:
nn.Linear(128 * 2 * 2, 256)   # 512 instead of 8192
```

---

## Lesson Learned

MaxPool spatial dimensions propagate multiplicatively through all downstream
layers. Any change to kernel size or stride invalidates the Flatten dimension
and the first classifier Linear layer's weight matrix. The shape contract must
be recalculated explicitly — it cannot be inferred from the architecture description alone.
