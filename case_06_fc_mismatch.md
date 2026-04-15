# Bug Case 6 — FC Hidden Layer: 256 → 512 (Downstream Dimension Mismatch)
**Student:** Jack Sweeney  
**Component:** Model Architecture (`model.py`)  
**GenAI Label:** ✅ Good  

---

## What Was Changed

### Original Code (Working)
```python
self.classifier = nn.Sequential(
    nn.Flatten(),
    nn.Linear(128 * 8 * 8, 256),   # FC1: 8192 → 256
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.5),
    nn.Linear(256, 10),             # FC2: 256 → 10  ✓ dimensions match
)
```

### Buggy Code
```python
self.classifier = nn.Sequential(
    nn.Flatten(),
    nn.Linear(128 * 8 * 8, 512),   # FC1 output changed: 8192 → 512
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.5),
    nn.Linear(256, 10),             # FC2 input STILL expects 256  ✗ MISMATCH
)
```

---

## Error Produced

```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (32x512 vs 256x10)
```

FC1 now outputs a `[B, 512]` tensor. FC2's weight matrix has shape `[256, 10]`
(input_features=256, output_features=10). PyTorch cannot multiply these:
`[B, 512] × [256, 10]` — the inner dimensions 512 ≠ 256.

---

## Dimension Chain Trace

```
Flatten output:  [B, 8192]
FC1 (8192→512):  [B, 512]    ← changed
After ReLU:      [B, 512]
After Dropout:   [B, 512]
FC2 (256→10):    CRASH       ← expects [B, 256], receives [B, 512]
```

---

## Initial Self-Reflection

> "I doubled the hidden layer size from 256 to 512. This probably won't crash 
> since the output layer is still Linear(512→10)... wait, no, the output layer 
> still says Linear(256→10). That will cause a shape mismatch."

**GenAI Label:** Good  
**GenAI Guidance:** "Correct — you caught the downstream mismatch. The output 
layer input dimension must match FC1's output dimension exactly. Any change to 
FC1's output size must be propagated to FC2's input size."

---

## Two Valid Fixes

**Option A — Revert FC1 to original:**
```python
nn.Linear(128 * 8 * 8, 256),   # FC1: 8192 → 256
nn.Linear(256, 10),             # FC2: 256  → 10  ✓
```

**Option B — Update FC2 to match new FC1 output:**
```python
nn.Linear(128 * 8 * 8, 512),   # FC1: 8192 → 512
nn.Linear(512, 10),             # FC2: 512  → 10  ✓
```

Option B is architecturally valid and increases model capacity slightly
(~130K additional parameters). Whether it improves accuracy depends on
whether the dataset is large enough to support the larger hidden dimension.

---

## Lesson Learned

In a sequential classifier head, every layer's output dimension must equal
the next layer's input dimension. This is an explicit contract that PyTorch
enforces at runtime during the first forward pass. When changing any layer's
width, always trace the change through every downstream layer before running.
A simple shape trace comment in the code prevents this entire class of bug:

```python
# Shape trace:
# Flatten    : [B, 8192]
# Linear FC1 : [B, 256]   ← if you change this output, update FC2's input
# Linear FC2 : [B, 10]
```
