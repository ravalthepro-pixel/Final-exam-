# Bug Case 9 — Missing optimizer.zero_grad() (Gradient Accumulation Bug)
**Student:** Jasraj "Jay" Raval  
**Component:** Training Loop (`train.py`)  
**GenAI Label:** ✅ Good  

---

## What Was Changed

### Original Code (Working)
```python
for imgs, labels in train_loader:
    imgs, labels = imgs.to(device), labels.to(device)

    optimizer.zero_grad()                        # ← clears gradients each step

    outputs = model(imgs)
    loss = criterion(outputs, labels)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
```

### Buggy Code
```python
for imgs, labels in train_loader:
    imgs, labels = imgs.to(device), labels.to(device)

    # optimizer.zero_grad()                      # ← REMOVED

    outputs = model(imgs)
    loss = criterion(outputs, labels)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
```

---

## Observed Behavior

Training ran for all 20 epochs with **no crash and no warning**.
Loss curves were erratic:

```
Epoch [01/20] Train Loss: 1.2341  Val Loss: 1.4521  Val Acc: 0.5123
Epoch [02/20] Train Loss: 2.8832  Val Loss: 3.1204  Val Acc: 0.3211
Epoch [03/20] Train Loss: 0.9123  Val Loss: 1.2004  Val Acc: 0.5834
Epoch [04/20] Train Loss: 4.2341  Val Loss: 5.0021  Val Acc: 0.1982
...
Epoch [20/20] Train Loss: 1.8902  Val Loss: 2.1043  Val Acc: 0.4123
Best Val Accuracy: 0.5834  (vs baseline ~0.9480)
```

Loss oscillated wildly between epochs. Validation accuracy peaked at ~58%
in epoch 3 and never recovered. The behavior was non-deterministic across
runs because accumulated gradients depended on the random batch order.

---

## Initial Self-Reflection

> "I removed optimizer.zero_grad(). I think gradients will accumulate across 
> batches, making each update increasingly large and causing unstable or 
> incorrect training."

**GenAI Label:** Good  
**GenAI Guidance:** "Correct. In PyTorch, gradients are accumulated (added) 
by default across multiple backward() calls. Without zero_grad(), each 
optimizer.step() uses the sum of all gradients computed since the last reset 
— not just the current batch's gradient. This is a design feature (used for 
gradient accumulation techniques), but in standard training it corrupts 
the update direction entirely."

---

## Root Cause: PyTorch's Default Gradient Accumulation

PyTorch accumulates gradients by default. This is intentional — it enables
gradient accumulation over multiple mini-batches, which is useful when
GPU memory is too small for the desired effective batch size.

In standard single-batch training:

```
Step 1: loss.backward()  → grad = ∇L_batch1
        optimizer.step() → θ ← θ - lr × ∇L_batch1  ✓

Step 2 (WITHOUT zero_grad):
        loss.backward()  → grad = ∇L_batch1 + ∇L_batch2   ← accumulated!
        optimizer.step() → θ ← θ - lr × (∇L_batch1 + ∇L_batch2)  ✗

Step 3 (WITHOUT zero_grad):
        loss.backward()  → grad = ∇L_batch1 + ∇L_batch2 + ∇L_batch3
        optimizer.step() → θ ← θ - lr × (sum of all prior batches)  ✗✗
```

By epoch N, each optimizer step is applying a gradient that has accumulated
across `N × batches_per_epoch` backward passes. The effective learning rate
grows proportionally, causing the observed oscillation.

---

## Why This Is Dangerous

1. **No crash** — PyTorch does not warn about accumulated gradients
2. **Partial training** — the model learns something (val acc > random chance)
   making the bug easy to dismiss as "dataset difficulty"
3. **Non-deterministic** — behavior differs across runs due to batch ordering
4. **Hard to isolate** — the loss curve oscillates in a way that could be
   confused with a too-high learning rate or an augmentation issue

---

## Fix

```python
optimizer.zero_grad()   # must be called before every loss.backward()
```

This resets all `.grad` attributes of all model parameters to zero, ensuring
each backward pass computes only the gradient of the current batch's loss.

---

## Lesson Learned

`optimizer.zero_grad()` is not optional boilerplate — it is a semantically
critical operation. Its absence does not crash PyTorch because gradient
accumulation is a legitimate training technique. The standard training loop
must always follow the sequence:

```
zero_grad() → forward() → backward() → step()
```

Missing any step corrupts the update direction. Missing `zero_grad()`
specifically causes the effective learning rate to grow with every batch,
eventually destabilizing training completely.
