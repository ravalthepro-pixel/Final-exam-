# Case 04 — Dropout: p=0.5 → p=1.0 (Complete Neuron Zeroing)
**Student:** Jack Sweeney | **Component:** Architecture (`model.py`) | **GenAI Label:** ❌ Bad

---

## What Was Changed
`nn.Dropout(p=0.5)` was changed to `nn.Dropout(p=1.0)`, treating p=1.0 as "more aggressive regularization" without recognizing it as a degenerate case that destroys gradient flow entirely.

## Original Code
```python
nn.Dropout(p=0.5)
```

## Buggy Code
```python
nn.Dropout(p=1.0)  # <-- BUG
```

## Observed Behavior (Silent Failure — No Crash)
```
Epoch 1 | Loss: 2.3034 | Acc: 16% | Dropout output all-zero: True
Epoch 2 | Loss: 2.3037 | Acc:  4% | Dropout output all-zero: True
...
Epoch 8 | Loss: 2.3124 | Acc:  2% | Dropout output all-zero: True
Loss stuck at ln(10) = 2.3026. No learning occurred.
```

## Initial Self-Reflection (Shallow — GenAI labeled ❌ Bad)
> "I increased the dropout probability to a higher value. This should make the regularization more aggressive. It might hurt the final accuracy somewhat since more neurons are being disabled on every forward pass, but I think the model will still be able to learn something meaningful from the training data."

## GenAI Socratic Guidance
> "You are thinking of dropout as a continuous dial where turning it up just means stronger regularization, but consider the boundary case of p=1.0 specifically. What numerical value does every single neuron output during a training forward pass when p=1.0? Given that output, what does the final Linear layer receive as its input vector? What does the gradient of that Linear layer's weight matrix look like when the input is that value — does any weight update ever happen?"

## New Self-Reflection (After Guidance)
> "Dropout(p=1.0) sets every activation to exactly zero with probability 1.0 on every single training forward pass. The final nn.Linear(256, 10) layer therefore receives an all-zero input vector for every batch throughout all training epochs. The gradient of the weight matrix W equals the outer product of the upstream delta and the input, which is always zero when the input is all zeros — so the weights of that layer never change from initialization. The training loss stays pinned at ln(10) = 2.3026 and accuracy stays at 10% random chance for the entire run with no error raised."

## Fix
```python
nn.Dropout(p=0.5)
```

## Lesson Learned
p=1.0 is not strong regularization — it is total gradient blockage. Loss stuck at ln(num_classes) is a reliable indicator that the final linear layer's input is all zeros on every batch.
