# Bug Case 7 — EPOCHS: 20 → 0 (No Training Occurs)
**Student:** Jasraj "Jay" Raval  
**Component:** Training Configuration (`train.py`)  
**GenAI Label:** ✅ Good  

---

## What Was Changed

### Original Code (Working)
```python
EPOCHS = 20
```

### Buggy Code
```python
EPOCHS = 0
```

---

## Observed Behavior

```
[1/6] Validating dataset structure... OK
[2/6] Loading dataset... Train: 1649 | Val: 413
[3/6] Building model... Parameters: 2,401,738
[4/6] Training for 0 epochs...
---------------------------------------------------------
Training complete in 0.0s (0.0 min)
Best validation accuracy: 0.0000
Best checkpoint saved to: results/best_model.pth
[5/6] Evaluating best checkpoint...
[6/6] Generating outputs...
```

Training completes in under one second with **no error and no warning**.
A `best_model.pth` checkpoint is saved — containing only the random 
initialization weights. Validation accuracy: **~10.0%** (random chance).

The loss value is exactly `ln(10) ≈ 2.3026` — the cross-entropy of a
uniform distribution over 10 balanced classes — confirming the model
outputs are effectively random.

---

## Why ~10% and Not Random?

With Kaiming He initialization, the weight distribution is not perfectly
uniform, so outputs are not exactly uniform either. But with no training,
the model has no information about the task. On a 10-class balanced
validation set of 412 images, ~10% accuracy (≈41 correct) is the
statistically expected result from random prediction.

The confusion matrix from a EPOCHS=0 run shows:
- No strong diagonal (no class is reliably predicted)
- Near-uniform off-diagonal distribution
- Per-class accuracy ranges from ~5% to ~15% (random variation)

---

## Initial Self-Reflection

> "Setting epochs to 0 means the training loop never runs. The model stays 
> at random initialization and should get ~10% accuracy on a 10-class 
> balanced problem."

**GenAI Label:** Good  
**GenAI Guidance:** "Correct. Zero epochs means no weight updates — the model 
stays at initialization. The ~10% expectation follows from balanced class 
distribution and random prediction."

---

## Why This Is a Silent Failure

This is the simplest possible bug, but it illustrates the most dangerous
failure category: **the code completes without error and produces a
plausible-looking output file.** A developer who only checks:
- ✅ Script ran without crashing
- ✅ `best_model.pth` was saved
- ✅ Plots were generated

...would not detect that no training occurred. Only inspecting the
**actual validation accuracy** reveals the problem.

---

## Fix

```python
EPOCHS = 20   # or any positive integer
```

---

## Lesson Learned

Always validate training outcomes, not just training completion. A clean
terminal output is not evidence that a model learned anything. The minimum
acceptable post-training check is:

```
assert best_val_acc > (1.0 / num_classes + 0.10), \
    f"Model barely beats random chance: {best_val_acc:.4f}"
```

This would catch EPOCHS=0, Dropout(1.0), and any other silent total-failure bug.
