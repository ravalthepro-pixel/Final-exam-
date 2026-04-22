# Case 06 — FC Hidden Layer: 256 → 512 (Downstream Dimension Mismatch)
**Student:** Jack Sweeney | **Component:** Architecture (`model.py`) | **GenAI Label:** ❌ Bad

---

## What Was Changed
`nn.Linear(128*8*8, 256)` (FC1) was changed to `nn.Linear(128*8*8, 512)` — doubling the hidden layer width — but `nn.Linear(256, 10)` (FC2) was not updated, creating an impossible matrix multiplication downstream.

## Original Code
```python
nn.Linear(128 * 8 * 8, 256)  # FC1: outputs 256
nn.Linear(256, 10)            # FC2: expects 256 ✓
```

## Buggy Code
```python
nn.Linear(128 * 8 * 8, 512)  # FC1: outputs 512 now  <-- BUG
nn.Linear(256, 10)            # FC2: still expects 256 ✗
```

## Error Produced
```
ValueError: Linear layer shape mismatch: input size is 512 but
weight matrix expects 256. (Cannot multiply [512] by [256, 10])
```

## Initial Self-Reflection (Shallow — GenAI labeled ❌ Bad)
> "I made the hidden layer wider which should give the model more capacity to learn complex features. A wider layer with more neurons might even help with accuracy on the ASL dataset. I do not think this will break anything since it is just a size increase to make the network more powerful."

## GenAI Socratic Guidance
> "You are thinking about this change in isolation, but in a Sequential classifier every layer's output size becomes the very next layer's required input size. Walk through the shapes step by step: what does FC1 now output after your change? What does FC2's weight matrix expect as its input dimension based on how it was defined? Write out the exact tensor shapes at each step and identify the specific point where the multiplication becomes mathematically impossible."

## New Self-Reflection (After Guidance)
> "After working through the shapes: FC1 is now nn.Linear(8192, 512) so it outputs a tensor of shape [B, 512]. The next layer, FC2, is still nn.Linear(256, 10), which has a weight matrix of shape [10, 256]. When PyTorch tries to compute FC2(FC1_output), it needs to multiply [B, 512] by [256, 10], but the inner dimensions 512 and 256 do not match, making the matrix multiplication impossible. PyTorch raises a RuntimeError on the first forward pass. The fix is either to revert FC1 back to output 256, or to also update FC2 to nn.Linear(512, 10) so the dimensions are consistent."

## Fix
```python
# Option A — revert FC1:
nn.Linear(128 * 8 * 8, 256)
nn.Linear(256, 10)

# Option B — update FC2 to match:
nn.Linear(128 * 8 * 8, 512)
nn.Linear(512, 10)
```

## Lesson Learned
In a Sequential classifier, every layer's output dimension is the next layer's input contract. Changing one layer's width without tracing the change through all downstream layers always produces a crash on first forward pass.
