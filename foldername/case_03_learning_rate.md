# Case 03 — Learning Rate: 0.001 → 10.0 (Gradient Explosion)
**Student:** Jasraj "Jay" Raval | **Component:** Training Config (`train.py`) | **GenAI Label:** ❌ Bad

---

## What Was Changed
`lr=0.001` in the Adam optimizer was changed to `lr=10.0` — an increase of four orders of magnitude — simulating a decimal-point typo that causes immediate gradient explosion.

## Original Code
```python
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

## Buggy Code
```python
optimizer = optim.Adam(model.parameters(), lr=10.0)  # <-- BUG
```

## Observed Behavior
```
Step  1 |  Loss: 3.7377 | Max|W|:   23.28 | ok
Step  2 |  Loss: 46.659 | Max|W|:  261.6x growth
Step  3 |  Loss: 87.498 | Max|W|: exploding
...
AssertionError: TRAINING FAILED: average loss 24.20 >> 5.0
```
Loss explodes immediately. Weights grew 261x in a single step. Training is unrecoverable.

## Initial Self-Reflection (Shallow — GenAI labeled ❌ Bad)
> "The learning rate is much bigger now, so the model will probably learn a lot faster. It might overshoot the optimal weights a little bit, but I think it will still converge eventually, just maybe not as smoothly or cleanly as it would with the original value."

## GenAI Socratic Guidance
> "You are assuming that a larger learning rate simply means faster learning, but think carefully about what Adam's update rule actually computes at lr=10. How large would a single weight update be compared to the initialized weight values? What happens to the activations passing through the network when the weights have changed by that magnitude after just one step? Is there a specific mathematical state — not just slow convergence — that can make the entire model permanently unusable?"

## New Self-Reflection (After Guidance)
> "With lr=10.0, Adam's update multiplies each weight change by roughly 10,000 times the intended magnitude. After just one step the maximum weight grows from about 0.39 to over 100 — a 261x increase in a single update. The loss immediately jumps to around 27 rather than converging toward zero, and the weights continue growing with each step. The model oscillates between extreme loss values and never converges. This is permanent gradient explosion: the model cannot recover without restarting from fresh random initialization with the correct lr=0.001."

## Fix
```python
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

## Lesson Learned
lr=0.001 is the standard Adam default for image classification. Deviations by more than one order of magnitude reliably cause training failure. Always start here and adjust incrementally.
