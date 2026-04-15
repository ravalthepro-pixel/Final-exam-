# Bug Case 3 — Learning Rate: 0.001 → 10.0 (Gradient Explosion)
**Student:** Jasraj "Jay" Raval  
**Component:** Training Configuration (`train.py`)  
**GenAI Label:** ✅ Good  

---

## What Was Changed

### Original Code (Working)
```python
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

### Buggy Code
```python
optimizer = optim.Adam(model.parameters(), lr=10.0)
```

---

## Observed Behavior

```
Epoch [01/20] Train Loss: 14.3821  Val Loss: 18.2134  Val Acc: 0.0965
Epoch [02/20] Train Loss: nan      Val Loss: nan       Val Acc: 0.1000
Epoch [03/20] Train Loss: nan      Val Loss: nan       Val Acc: 0.1000
...
```

Training loss became `NaN` at epoch 2. Validation accuracy stayed at ~10%
(random chance on a 10-class problem) for all remaining epochs. The model
was unrecoverable without restarting from random initialization.

---

## Initial Self-Reflection

> "A learning rate of 10 is extremely large for Adam. I expect training loss 
> to explode or oscillate rather than converge. The gradients will overshoot 
> any minimum."

**GenAI Label:** Good  
**GenAI Guidance:** "Correct diagnosis. Large LR causes gradient explosion in 
Adam. The weight updates become so large that activations overflow to inf, 
which then propagates to NaN through subsequent operations. Continue with your fix."

---

## Root Cause Explanation

Adam's weight update rule:

```
θ ← θ - lr × m̂ / (√v̂ + ε)
```

With `lr=10.0`, each weight update is 10,000× larger than intended. In the
first backward pass, gradients are applied to random-initialized weights that
were not designed to absorb such large updates. Activations overflow to `inf`,
and any arithmetic involving `inf` produces `NaN` — which then propagates
through the entire computation graph.

**Why NaN is unrecoverable mid-training:**
Once any weight becomes `NaN`, every subsequent forward pass produces `NaN`
outputs, `NaN` loss, and `NaN` gradients. The optimizer cannot recover because
the gradient signal is lost. Training must be restarted.

---

## Learning Rate Sensitivity Reference

| LR Value | Behavior | Failure Type |
|---|---|---|
| 0.0001 | Very slow convergence | Silent (underfitting) |
| 0.001 | Stable — **standard default** | None |
| 0.01 | Unstable, oscillating loss | Visible instability |
| 0.1 | Likely divergence | Obvious failure |
| 10.0 | NaN at epoch 2 | Immediate crash |

---

## Fix

```python
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

Adam's standard starting learning rate is `1e-3`. It is the most widely
validated default across image classification tasks.

---

## Lesson Learned

The learning rate is the most sensitive hyperparameter in Adam. Values that
are too large produce fast, obvious, unrecoverable failures (NaN loss).
Values that are too small produce slow, hard-to-detect underfitting.
Always start at `lr=0.001` and adjust by at most one order of magnitude
at a time, monitoring loss curves after each change.
