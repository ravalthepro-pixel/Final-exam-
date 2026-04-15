# Bug Case 4 — Dropout: p=0.5 → p=1.0 (Complete Neuron Zeroing)
**Student:** Jack Sweeney  
**Component:** Model Architecture (`model.py`)  
**GenAI Label:** ✅ Good  

---

## What Was Changed

### Original Code (Working)
```python
self.classifier = nn.Sequential(
    nn.Flatten(),
    nn.Linear(128 * 8 * 8, 256),
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.5),          # drops 50% of neurons during training
    nn.Linear(256, 10),
)
```

### Buggy Code
```python
self.classifier = nn.Sequential(
    nn.Flatten(),
    nn.Linear(128 * 8 * 8, 256),
    nn.ReLU(inplace=True),
    nn.Dropout(p=1.0),          # drops 100% of neurons during training
    nn.Linear(256, 10),
)
```

---

## Observed Behavior

```
Epoch [01/20] Train Loss: 2.3026  Train Acc: 0.0982  Val Acc: 0.1000
Epoch [02/20] Train Loss: 2.3026  Train Acc: 0.1004  Val Acc: 0.1000
...
Epoch [20/20] Train Loss: 2.3026  Train Acc: 0.0996  Val Acc: 0.1000
```

Training and validation accuracy stayed exactly at ~10% — random chance on
a 10-class balanced problem — for all 20 epochs. Loss remained at 2.3026,
which is precisely `ln(10)` — the cross-entropy of a uniform distribution
over 10 classes. This is a silent failure: no error, no warning, no crash.

---

## Initial Self-Reflection

> "Setting dropout to 1.0 means every neuron is dropped during training. 
> The model probably won't learn anything."

**GenAI Label:** Good  
**GenAI Guidance:** "Correct. p=1.0 zeroes all outputs of the dropout layer 
every forward pass during training. Because all activations entering the final 
Linear layer are zero, gradients through that layer are also zero — no weight 
update can occur downstream of the Dropout layer."

---

## Root Cause Explanation

`nn.Dropout(p=1.0)` sets **every neuron output to zero** during every training
forward pass. The final `nn.Linear(256, 10)` receives an all-zero input `[B, 256]`
for every batch:

```
FC1 output:        [B, 256]  ← some nonzero activations
After Dropout(1.0): [B, 256]  ← all zeros
FC2 input:         [B, 256]  ← all zeros → output is bias vector only
```

The output logits become the bias vector of FC2, which starts near zero.
Cross-entropy of near-uniform logits over 10 classes ≈ `ln(10)` ≈ 2.3026.
No gradient flows back through FC2's weight matrix because the input is zero.
FC1 also receives no gradient signal. The model is structurally frozen.

**Why this is dangerous:** There is no error. The model trains for 20 epochs
and saves a `best_model.pth` checkpoint — which contains weights that have
never been updated from initialization.

---

## Fix

```python
nn.Dropout(p=0.5)   # standard default: drop 50% of neurons
```

Valid range: `0.0 ≤ p < 1.0`. Setting `p=0.0` disables dropout entirely.
Setting `p=1.0` is only valid for specific research purposes and will always
prevent learning in standard classifier heads.

---

## Lesson Learned

A Dropout probability of 1.0 is a silent model-killer. The loss value `2.3026`
(= ln(10)) is a reliable diagnostic indicator that the model is outputting
near-uniform distributions — a strong signal that gradient flow is broken
somewhere in the classifier head. Always check loss against `ln(num_classes)`
as a baseline sanity check.
