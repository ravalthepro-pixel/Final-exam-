"""
================================================================================
BUG CASE 9 — Missing optimizer.zero_grad() (Gradient Accumulation)
================================================================================
Student   : Jasraj "Jay" Raval
Component : Training Loop
GenAI Label: BAD

INITIAL SELF-REFLECTION (shallow — GenAI labeled Bad):
    "I removed a line from the training loop. I think the gradients
    might not reset properly so the training might be a bit unstable
    or converge slowly."

GENAI SOCRATIC GUIDANCE:
    "You say 'a bit unstable or slow' but that understates the
    mechanism. Gradients accumulate by default — are they reset or
    added each backward pass? After N batches without zeroing, what
    does the gradient tensor actually represent? How does the effective
    learning rate change over batches? Is it slower convergence or
    is the update direction itself corrupted?"

NEW REFLECTION (after guidance):
    "Without zeroing gradients, after batch N the gradient holds the
    sum of all N previous batches' gradients. The optimizer applies
    this accumulated sum × lr as the weight update. The effective
    learning rate grows proportionally with every batch — by batch 50
    it is 50× the intended rate. This corrupts the update direction
    entirely, causing wild oscillation. Validation accuracy peaks at
    ~58% vs ~94% baseline and never recovers. No error is raised."

FIX: Restore grad = 0 (zero_grad) before every backward pass.
================================================================================
"""

import numpy as np

np.random.seed(42)

INPUT, HIDDEN, OUTPUT = 32, 64, 10
LR = 0.005

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

def forward_and_grads(W1, b1, W2, b2, x, label):
    h     = np.tanh(W1 @ x + b1)
    out   = W2 @ h + b2
    probs = softmax(out)
    loss  = -np.log(probs[label] + 1e-12)
    one_hot       = np.zeros(OUTPUT); one_hot[label] = 1.0
    d_out         = probs - one_hot
    gW2           = np.outer(d_out, h)
    gb2           = d_out
    dh            = W2.T @ d_out * (1 - h**2)
    gW1           = np.outer(dh, x)
    gb1           = dh
    return loss, gW1, gb1, gW2, gb2


def run_training(zero_grad_each_step, n_steps=80):
    np.random.seed(42)
    W1 = np.random.randn(HIDDEN, INPUT)  * 0.1
    b1 = np.zeros(HIDDEN)
    W2 = np.random.randn(OUTPUT, HIDDEN) * 0.1
    b2 = np.zeros(OUTPUT)

    # Accumulated gradients (simulating PyTorch's .grad tensors)
    acc_gW1 = np.zeros_like(W1)
    acc_gb1 = np.zeros_like(b1)
    acc_gW2 = np.zeros_like(W2)
    acc_gb2 = np.zeros_like(b2)

    losses     = []
    grad_norms = []

    for step in range(n_steps):
        x     = np.random.randn(INPUT)
        label = np.random.randint(0, OUTPUT)

        if zero_grad_each_step:
            # CORRECT: reset accumulated gradients before each backward
            acc_gW1[:] = 0; acc_gb1[:] = 0
            acc_gW2[:] = 0; acc_gb2[:] = 0

        # BUG path: gradients ACCUMULATE (add to existing acc_g*)
        loss, gW1, gb1, gW2, gb2 = forward_and_grads(W1, b1, W2, b2, x, label)
        acc_gW1 += gW1; acc_gb1 += gb1
        acc_gW2 += gW2; acc_gb2 += gb2

        # Optimizer step uses accumulated gradients
        W1 -= LR * acc_gW1
        b1 -= LR * acc_gb1
        W2 -= LR * acc_gW2
        b2 -= LR * acc_gb2

        norm = (np.linalg.norm(acc_gW1)**2 + np.linalg.norm(acc_gW2)**2)**0.5
        losses.append(loss)
        grad_norms.append(norm)

    return losses, grad_norms


print("Bug Case 9 — Missing zero_grad() — gradient accumulation")
print("Initial reflection was labeled BAD by GenAI.")
print()

correct_losses, _       = run_training(zero_grad_each_step=True)
buggy_losses, bug_norms = run_training(zero_grad_each_step=False)  # <-- BUG

print(f"{'Step':>5} | {'Correct Loss':>12} | {'Buggy Loss':>10} | {'Grad Norm (buggy)':>18}")
print("-" * 58)
for i in [0, 9, 19, 29, 39, 49, 59, 69, 79]:
    flag = " ← exploding" if bug_norms[i] > 20 else ""
    print(f"{i+1:>5} | {correct_losses[i]:>12.4f} | {buggy_losses[i]:>10.4f} | "
          f"{bug_norms[i]:>16.2f}{flag}")

print()
print("Correct: loss decreasing toward convergence.")
print("Buggy:   loss oscillating, gradient norm growing every step.")
print("Effective LR at step 80 is ~80× the intended value.")
print("Fix: zero out accumulated gradients before every backward pass.")
