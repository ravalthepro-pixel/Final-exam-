# Bug Case 10 — BATCH_SIZE: 32 → 4096 (Large-Batch Training Degradation)
**Student:** Jack Sweeney  
**Component:** Training Configuration (`train.py`)  
**GenAI Label:** ❌ Bad → revised to deep understanding  

---

## What Was Changed

### Original Code (Working)
```python
BATCH_SIZE = 32
```

### Buggy Code
```python
BATCH_SIZE = 4096
```

---

## Observed Behavior

With batch size 4096 and a training set of ~1,649 images:
- The DataLoader produced **1 batch per epoch** (since 4096 > 1649, PyTorch
  clamps the batch to the dataset size)
- Training ran for all 20 epochs with **no crash on our CPU machine**
  (dataset fit in RAM; no OOM error)
- Validation accuracy degraded noticeably:

```
Baseline (batch=32):    val accuracy ≈ 94.8%
Buggy    (batch=4096):  val accuracy ≈ 87.6%
Accuracy drop:          ≈ 7.2 percentage points
```

Loss converged, but to a worse final value. The confusion matrix showed
broader off-diagonal spread — the model generalized more poorly across
all digit classes, with no single class catastrophically failing.

---

## Self-Reflection Process

**Initial Reflection (Bad):**
> "A batch size of 4096 is way larger than our dataset of ~1650 training 
> images. I think the DataLoader will just return one giant batch per epoch, 
> but I'm not sure if this crashes or just trains badly."

**GenAI Label:** Bad  
**GenAI Socratic Guidance (Claude):** "You're right that behavior changes, 
but think more carefully: what specifically happens to gradient quality and 
model generalization with very large batches? Is this a runtime error, a 
silent performance degradation, or both? Consider also memory implications 
on different hardware."

**Revised Reflection (after guidance):**
> "With batch size 4096 > dataset size 1650, the entire training set becomes 
> one batch. Each gradient update uses the loss averaged over all training 
> examples — a 'full-batch gradient.' This has low noise but leads to 
> convergence to sharper minima (Keskar et al. 2017), which generalize 
> worse to unseen data. Additionally, on memory-limited hardware (small GPU 
> VRAM), loading 4096 image tensors simultaneously would cause an OOM crash. 
> On our CPU machine with sufficient RAM, it ran but degraded generalization."

---

## Root Cause: Sharp Minima and Gradient Noise

**Why small batches generalize better:**

Small batches (e.g., 32) introduce stochastic noise into gradient estimates.
This noise acts as implicit regularization — the optimizer explores a
wider region of the loss landscape and tends to find **flat minima**, which
generalize better to unseen data (lower sensitivity to input perturbations).

Large batches (e.g., 4096) produce highly accurate gradient estimates.
The optimizer converges efficiently but tends to find **sharp minima** —
narrow loss-landscape regions where small input changes cause large output
changes. Sharp minima generalize poorly.

```
Small batch (noise):  ~~~→ lands in wide, flat minimum  → good generalization
Large batch (precise): ──→ converges to sharp minimum   → poor generalization
```

This is the core finding of Keskar et al. (2017), "On Large-Batch Training
for Deep Learning: Generalization Gap and Sharp Minima."

**Hardware dependency:**

| Hardware | RAM / VRAM | Batch=4096 Result |
|---|---|---|
| CPU (our machine) | 16GB RAM | No crash; 7.2% acc drop |
| GPU (8GB VRAM) | 8GB VRAM | Likely OOM crash |
| GPU (40GB A100) | 40GB VRAM | No crash; similar acc drop |

The same bug has hardware-dependent symptoms — making it one of the hardest
bugs to reason about without execution access.

---

## Fix

```python
BATCH_SIZE = 32   # standard default for small datasets
```

**Batch size guidelines:**
- `16–64`: Standard range for small datasets (~1K–10K images)
- `32`: Most widely validated default for image classification with Adam
- `256+`: Appropriate only with learning rate scaling (linear scaling rule)
- `> dataset_size`: Almost always a mistake in standard training

---

## Lesson Learned

Large batch sizes do not simply "slow down" training — they change the
optimization trajectory in ways that systematically harm generalization.
This is a silent failure: training completes normally, loss converges,
and the model appears to have learned. Only careful evaluation of validation
accuracy (and ideally the confusion matrix) reveals the degradation.

Additionally, this bug demonstrated that GenAI tools cannot predict
hardware-specific behavior (OOM vs. no crash) without execution access —
a fundamental limitation of text-only reasoning about system behavior.
