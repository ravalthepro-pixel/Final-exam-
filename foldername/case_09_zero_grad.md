# Case 09 — Missing optimizer.zero_grad() (Gradient Accumulation Bug)
**Student:** Jasraj "Jay" Raval | **Component:** Training Loop (`train.py`) | **GenAI Label:** ❌ Bad

---

## What Was Changed
`optimizer.zero_grad()` was removed from the training loop, exploiting PyTorch's default behavior of accumulating gradients additively across backward() calls rather than overwriting them.

## Original Code
```python
optimizer.zero_grad()       # resets gradients each step
outputs = model(imgs)
loss = criterion(outputs, labels)
loss.backward()
optimizer.step()
```

## Buggy Code
```python
# optimizer.zero_grad()    # <-- BUG: line removed
outputs = model(imgs)
loss = criterion(outputs, labels)
loss.backward()
optimizer.step()
```

## Observed Behavior (Silent Failure — No Crash)
```
Step  1 | Correct Loss: 2.1764 | Buggy Loss: 2.1764 | Grad Norm: 4.67
Step 10 | Correct Loss: 2.1705 | Buggy Loss: 2.1930 | Grad Norm: 14.29
Step 20 | Correct Loss: 2.6698 | Buggy Loss: 2.8075 | Grad Norm: 21.84 ← exploding
Step 80 | Correct Loss: 2.1389 | Buggy Loss: 8.6454 | Grad Norm: 64.04 ← exploding
Effective LR at step 80 is ~80× the intended value.
```

## Initial Self-Reflection (Shallow — GenAI labeled ❌ Bad)
> "I removed a line from the training loop. I think the gradients might not reset properly between batches, so the training might become a bit unstable or perhaps converge more slowly than it normally would."

## GenAI Socratic Guidance
> "You say the training might be 'a bit unstable or slower' but that significantly understates what actually happens. In PyTorch, gradients are accumulated — meaning added to the existing .grad tensors — by default with each backward() call. After N batches without calling zero_grad(), what does the gradient tensor for any given parameter actually represent mathematically? How does the magnitude of the gradient signal change as training progresses through more and more batches? Is the consequence just slower convergence, or is the direction of every weight update itself corrupted?"

## New Self-Reflection (After Guidance)
> "Without zero_grad(), PyTorch adds each new batch's gradient to all previously accumulated gradients from every prior batch in the entire training run. After step N, each parameter's gradient tensor contains the sum of gradients from all N batches seen so far. When optimizer.step() fires, it applies this accumulated sum multiplied by the learning rate as the weight update. In our experiment the gradient norm grew from 4.67 at step 1 to 64.04 at step 80 — a roughly 14x increase — representing an effective learning rate that had grown 80x beyond its intended value. The update direction is completely corrupted because it reflects the sum of all past gradient signals rather than just the current batch, causing the buggy training loss to oscillate wildly while the correct training loss steadily decreased."

## Fix
```python
optimizer.zero_grad()  # must come before every loss.backward()
```

## Lesson Learned
optimizer.zero_grad() is not optional boilerplate. Its absence does not crash PyTorch because gradient accumulation is a legitimate technique — but in standard training it corrupts every weight update. The standard training loop sequence is always: zero_grad → forward → backward → step.
