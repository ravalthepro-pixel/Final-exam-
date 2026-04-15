# Bug Case 8 — RandomRotation: ±10° → ±180° (Label Semantic Corruption)
**Student:** Jack Sweeney  
**Component:** Data Preprocessing / Augmentation (`train.py`)  
**GenAI Label:** ❌ Bad → revised to deep understanding  

---

## What Was Changed

### Original Code (Working)
```python
transforms.RandomRotation(degrees=10)
# Rotates image by a random angle in [-10°, +10°]
```

### Buggy Code
```python
transforms.RandomRotation(degrees=180)
# Rotates image by a random angle in [-180°, +180°]
```

---

## Observed Behavior

Training ran for all 20 epochs with no error. Loss decreased normally.
But validation accuracy degraded significantly:

```
Baseline (±10° rotation):   val accuracy ≈ 94.8%
Buggy    (±180° rotation):  val accuracy ≈ 71.3%
Accuracy drop:              ≈ 23.5 percentage points
```

The confusion matrix revealed **systematic errors on rotationally-similar
digit pairs** — particularly digits whose ASL hand signs are visually related
when rotated (e.g., 6 and 9):

```
Confusion matrix excerpt (buggy run):
True\Pred   6     9
    6      29    11     ← 11 images of '6' predicted as '9'
    9       9    32     ← 9  images of '9' predicted as '6'
```

In the baseline run, these off-diagonal values were ≤ 2.

---

## Self-Reflection Process

**Initial Reflection (Bad):**
> "Rotating up to 180 degrees seems too aggressive. A hand sign rotated 
> 180 degrees looks completely different. I think the model will train 
> but generalize poorly."

**GenAI Label:** Bad  
**GenAI Socratic Guidance (Claude):** "You identified a semantic issue, but 
think more carefully about the training dynamics: does aggressive augmentation 
always hurt? Consider what happens to training loss vs. validation loss 
separately. Could it also cause an entirely different problem with label validity?"

**Revised Reflection (after guidance):**
> "With 180° rotations, augmented images may not correspond to valid ASL signs 
> anymore — the label becomes semantically incorrect. This is a form of data 
> poisoning via augmentation. The model receives the same label for both the 
> canonical gesture and its 180° rotation, which look nothing alike. Training 
> loss stays high because the model sees conflicting evidence for each class. 
> Validation accuracy drops because the model learns incorrect features that 
> appear in rotated-but-mislabeled training examples."

---

## Root Cause: Invariance Hypothesis Violation

Data augmentation encodes an implicit hypothesis:
**"This transformation preserves the semantic class of the image."**

For ASL digit recognition:
- ✅ `RandomRotation(±10°)` — a hand sign tilted 10° is still the same sign
- ❌ `RandomRotation(±180°)` — a hand sign rotated 180° is not the same sign

A hand making the ASL digit '6' rotated 180° does not look like '6'. It
may resemble '9', another digit, or no valid digit at all. By labeling it
as '6' anyway, the dataset gains corrupted training examples that actively
contradict correct examples. This is **augmentation-induced label noise**.

```
Training example (canonical):  image of '6' → label 6  ✓
Training example (180° rotated): rotated '6' → label 6  ✗ (looks like '9')
Model sees conflicting evidence: same label, contradictory features
```

---

## Fix

```python
transforms.RandomRotation(degrees=10)   # ±10° preserves ASL sign semantics
```

The appropriate rotation range depends on domain knowledge about how much
variation exists in real-world capture angles. For hand signs photographed
against a plain background, ±10°–15° is generally safe.

---

## Lesson Learned

Augmentation choices are not neutral implementation details. Each augmentation
type encodes a hypothesis about which transformations preserve class semantics
in the specific task domain. Valid augmentation for one domain (e.g., rotating
a cat photo is fine — cats exist at all orientations) may be invalid in another
(rotating a hand sign changes its meaning). Domain knowledge must validate
augmentation choices before training — not after observing degraded accuracy.
