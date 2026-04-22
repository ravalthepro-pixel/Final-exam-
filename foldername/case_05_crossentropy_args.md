# Bug Case 5 — CrossEntropyLoss Arguments Swapped
**Student:** Jasraj "Jay" Raval  
**Component:** Training Loop (`train.py`)  
**GenAI Label:** ❌ Bad → revised to deep understanding  

---

## What Was Changed

### Original Code (Working)
```python
loss = criterion(outputs, labels)
# criterion = nn.CrossEntropyLoss()
# outputs shape: [B, 10]  — raw logits, dtype=float32
# labels  shape: [B]      — class indices, dtype=int64 (long)
```

### Buggy Code
```python
loss = criterion(labels, outputs)
# Arguments swapped: target passed as input, input passed as target
```

---

## Error Produced

```
ValueError: Expected target size (32, 10), got torch.Size([32])
```

The error fires on the **first training batch**, so training never begins.
This is a hard crash — the most diagnosable category of bug.

---

## CrossEntropyLoss API Contract

```
torch.nn.CrossEntropyLoss()(input, target)

    input  : Tensor of shape [B, C]  — raw logits (float32)
               C = number of classes
    target : Tensor of shape [B]     — class indices (int64 / long)
               values in range [0, C-1]
```

When arguments are swapped:
```
criterion(labels, outputs)
  → input  receives: labels  [B]      dtype=int64  ← wrong shape AND dtype
  → target receives: outputs [B, 10]  dtype=float  ← wrong shape AND dtype

PyTorch expects target to have shape [B], but receives [B, 10]
→ ValueError: Expected target size (32, 10), got torch.Size([32])
```

The error message is slightly confusing: it says it expected `(32, 10)` because
PyTorch is trying to interpret the 2D float tensor as a target and looking for
a matching shape — not because `[B, 10]` is correct for the target.

---

## Self-Reflection Process

**Initial Reflection (Bad):**
> "I swapped the arguments to CrossEntropyLoss. I think the shapes might be 
> wrong — labels are integers and outputs are logits, so passing them in the 
> wrong order might throw an error."

**GenAI Label:** Bad  
**GenAI Socratic Guidance:** "Think more carefully: what are the exact tensor 
shapes and dtypes of each argument? CrossEntropyLoss expects (input, target) 
where input is [B, C] floats and target is [B] long ints. What happens to each 
when these are swapped?"

**Revised Reflection (after guidance):**
> "CrossEntropyLoss(labels, outputs) passes a [B] int64 tensor as input and a 
> [B, 10] float tensor as target. PyTorch tries to broadcast the target against 
> the input shape and raises ValueError because [B,10] cannot serve as a scalar 
> index target. The fix is to restore the correct argument order: criterion(outputs, labels)."

---

## Fix

```python
loss = criterion(outputs, labels)
#                ↑         ↑
#           [B, 10]    [B] long
#           input      target
```

---

## Lesson Learned

PyTorch loss functions have strict positional argument contracts that are
enforced at runtime, not at definition time. The error message for swapped
arguments can be misleading — it reports a shape expectation that seems to
imply the 2D tensor should be the target. Always verify the API signature:
`CrossEntropyLoss(input, target)` where `input` is always the model output.
