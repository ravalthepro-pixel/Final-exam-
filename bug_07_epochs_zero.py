"""
================================================================================
BUG CASE 7 — EPOCHS: 20 → 0 (Silent Failure — No Training Occurs)
================================================================================
Student   : Jasraj "Jay" Raval
Component : Training Configuration
GenAI Label: BAD

INITIAL SELF-REFLECTION (shallow — GenAI labeled Bad):
    "With zero epochs the training won't run. The model will probably
    just give bad results since it didn't train. I should set it back
    to 20."

GENAI SOCRATIC GUIDANCE:
    "You identified the surface consequence but not the mechanism or
    danger. What does the model output when it has never been trained?
    What is the mathematical expected accuracy on a 10-class balanced
    validation set from a randomly initialized model? What does the
    saved checkpoint actually contain? Why is this failure mode more
    dangerous than a runtime crash?"

NEW REFLECTION (after guidance):
    "A randomly initialized model outputs near-uniform logits over 10
    classes. Expected accuracy on a balanced 10-class set = 1/10 = 10%.
    Cross-entropy of a uniform distribution = ln(10) ≈ 2.3026. The danger:
    the script completes with no error, saves a checkpoint, and generates
    all output files. A developer checking only for a clean terminal output
    would submit a model that has never been trained. This is more dangerous
    than a crash because there is no signal that anything went wrong."

FIX: Change EPOCHS = 0 → EPOCHS = 20
================================================================================
"""

import numpy as np
import math

np.random.seed(42)

# BUG: training loop will never execute
EPOCHS      = 0    # <-- BUG: should be 20
INPUT_SIZE  = 64
HIDDEN_SIZE = 128
NUM_CLASSES = 10
LR          = 0.01

# Random initialization
W1 = np.random.randn(HIDDEN_SIZE, INPUT_SIZE)  * 0.1
b1 = np.zeros(HIDDEN_SIZE)
W2 = np.random.randn(NUM_CLASSES, HIDDEN_SIZE) * 0.1
b2 = np.zeros(NUM_CLASSES)

# Snapshot initial weights to prove they never change
W1_initial = W1.copy()
W2_initial = W2.copy()


def forward(x):
    h    = np.tanh(W1 @ x + b1)
    out  = W2 @ h + b2
    e    = np.exp(out - out.max())
    return e / e.sum()


def accuracy_on_val(n_samples=300):
    correct = 0
    for _ in range(n_samples):
        x     = np.random.randn(INPUT_SIZE)
        label = np.random.randint(0, NUM_CLASSES)
        pred  = np.argmax(forward(x))
        correct += (pred == label)
    return correct / n_samples


print("Bug Case 7 — EPOCHS=0 silent failure (no training occurs)")
print("Initial reflection was labeled BAD by GenAI.")
print(f"EPOCHS = {EPOCHS}  (should be 20)")
print()

best_val_acc = 0.0

# BUG: this loop executes 0 times
for epoch in range(1, EPOCHS + 1):
    # Fake training step (never runs)
    x      = np.random.randn(INPUT_SIZE)
    label  = np.random.randint(0, NUM_CLASSES)
    probs  = forward(x)
    # ... gradient update would happen here ...
    best_val_acc = 0.95  # never reached

print(f"Training loop executed 0 times.")
print(f"best_val_acc never updated from: {best_val_acc}")
print()

val_acc  = accuracy_on_val()
rand_loss = math.log(NUM_CLASSES)

weights_unchanged = np.allclose(W1, W1_initial) and np.allclose(W2, W2_initial)

print(f"Validation accuracy (untrained):  {val_acc:.1%}   (expected ≈ 10%)")
print(f"Expected random-chance loss:      {rand_loss:.4f}  (= ln(10))")
print(f"Weights unchanged from init:      {weights_unchanged}")
print()
print("No error was raised. Script appears to have succeeded.")
print("Checkpoint would be saved with random initialization weights.")
print("This is a silent failure — only accuracy inspection reveals the bug.")
print(f"Fix: EPOCHS = 0 → EPOCHS = 20")
