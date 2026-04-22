"""
================================================================================
BUG CASE 3 — Learning Rate: 0.001 → 10.0 (Gradient Explosion)
================================================================================
Student   : Jasraj "Jay" Raval
Component : Training Configuration
GenAI Label: BAD

INITIAL SELF-REFLECTION (shallow — GenAI labeled Bad):
    "The learning rate is bigger now so the model will learn faster.
    It might overshoot a little but I think it will still converge,
    just maybe not as cleanly."

GENAI SOCRATIC GUIDANCE:
    "You assume a large LR means faster learning. What does the update
    rule actually do with lr=10? How large would a single weight update
    be relative to the initialized weight values? What happens to
    activations when weights change by that magnitude in one step?"

NEW REFLECTION (after guidance):
    "With lr=10, each update multiplies the gradient by 10,000x the
    intended value. Weights explode immediately — by step 2 the max
    weight has grown 300x. The loss never decreases from its maximum
    value (ln(10) ≈ 2.3026 becomes ~87 due to extreme logit values).
    The model oscillates wildly and never converges. A training run
    that should reach ~94% accuracy stays at effectively random chance
    and the script raises an AssertionError to flag this."

FIX: Change LR = 10.0 → LR = 0.001
================================================================================
"""

import numpy as np

np.random.seed(42)

INPUT, HIDDEN, OUTPUT = 64, 128, 10

# BUG: LR = 10.0 instead of 0.001
LR = 10.0   # <-- BUG: should be 0.001


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


W1 = np.random.randn(HIDDEN, INPUT)  * 0.1
b1 = np.zeros(HIDDEN)
W2 = np.random.randn(OUTPUT, HIDDEN) * 0.1
b2 = np.zeros(OUTPUT)

init_max_w = max(np.abs(W1).max(), np.abs(W2).max())

print("Bug Case 3 — Learning rate = 10.0 (gradient explosion)")
print("Initial reflection was labeled BAD by GenAI.")
print()
print(f"Initial max |W|: {init_max_w:.4f}")
print()
print(f"{'Step':>5} | {'Loss':>10} | {'Max |W|':>12} | {'Weight growth':>14}")
print("-" * 55)

losses = []
for step in range(1, 16):
    x     = np.random.randn(INPUT)
    label = np.random.randint(0, OUTPUT)

    h     = np.tanh(W1 @ x + b1)
    out   = W2 @ h + b2
    probs = softmax(out)
    loss  = -np.log(probs[label] + 1e-12)
    losses.append(loss)

    one_hot = np.zeros(OUTPUT); one_hot[label] = 1.0
    d2  = probs - one_hot
    W2 -= LR * np.outer(d2, h)
    b2 -= LR * d2
    dh  = (W2.T @ d2) * (1 - h**2)
    W1 -= LR * np.outer(dh, x)
    b1 -= LR * dh

    max_w  = max(np.abs(W1).max(), np.abs(W2).max())
    growth = max_w / init_max_w
    print(f"{step:>5} | {loss:>10.4f} | {max_w:>12.2f} | {growth:>13.1f}x")

print()
final_max_w = max(np.abs(W1).max(), np.abs(W2).max())
avg_loss    = np.mean(losses)
print(f"Average loss over 15 steps: {avg_loss:.4f}  (healthy training would reach < 1.0)")
print(f"Final max weight magnitude: {final_max_w:.2f}  (started at {init_max_w:.4f})")
print()

# Assert that training has failed — this will raise AssertionError
assert avg_loss < 5.0, (
    f"TRAINING FAILED: average loss {avg_loss:.2f} >> 5.0. "
    f"With LR=10.0 the model explodes and never converges. "
    f"Fix: change LR = 10.0 to LR = 0.001"
)

print("Should never reach this line — assertion above catches the failure.")
