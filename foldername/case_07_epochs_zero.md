# Case 07 — EPOCHS: 20 → 0 (Silent Failure — No Training)
**Student:** Jasraj "Jay" Raval | **Component:** Training Config (`train.py`) | **GenAI Label:** ❌ Bad

---

## What Was Changed
`EPOCHS = 20` was changed to `EPOCHS = 0`, causing the training loop to execute zero times. The script completes silently with no error, saves a checkpoint, and generates all output files — but the model was never trained.

## Original Code
```python
EPOCHS = 20
```

## Buggy Code
```python
EPOCHS = 0  # <-- BUG
```

## Observed Behavior (Silent Failure — No Crash)
```
Training loop executed 0 times. best_val_acc = 0.0
Validation accuracy (untrained):  9.3%   (expected ≈ 10%)
Expected random-chance loss:      2.3026  (= ln(10))
Weights unchanged from init:      True
No error was raised. Script appears to have succeeded.
```

## Initial Self-Reflection (Shallow — GenAI labeled ❌ Bad)
> "With zero epochs the training loop will not run at all. The model will probably just give bad results since it never got to train on any data. I should set it back to 20 to fix this."

## GenAI Socratic Guidance
> "You identified the surface consequence — bad results — but your reflection does not engage with the mechanism or explain why this failure mode is particularly dangerous. Think about this: what exactly does a randomly initialized model output when it has never seen any training data? What numerical accuracy would you mathematically expect on a 10-class perfectly balanced validation set? What does the saved checkpoint file actually contain? And crucially — why is this specific type of failure more dangerous to a developer than a script that crashes with an error message?"

## New Self-Reflection (After Guidance)
> "A model with random Kaiming He initialization outputs near-uniform logits over all 10 classes. On a balanced 10-class validation set the expected accuracy is exactly 1/10 = 10%, and the cross-entropy loss equals ln(10) = 2.3026 — confirmed in our experiment at 9.3% accuracy. The saved checkpoint file best_model.pth contains only the random initialization weights with zero gradient updates applied. The reason this is more dangerous than a runtime crash is that the script completes without any error, generates all output files including plots, and saves a checkpoint that looks indistinguishable from a trained model. A developer who only checks that the script ran without crashing would unknowingly submit a model that does nothing more than random guessing."

## Fix
```python
EPOCHS = 20
```

## Lesson Learned
Always validate training outcomes — not just training completion. A clean terminal output is not evidence that a model learned anything. Checking that val accuracy meaningfully exceeds 1/num_classes is the minimum acceptable post-training sanity check.
