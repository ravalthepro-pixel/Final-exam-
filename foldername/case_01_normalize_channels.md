# Case 01 — Normalize: 3-Channel List → Single Scalar
**Student:** Jasraj "Jay" Raval | **Component:** Preprocessing (`train.py`) | **GenAI Label:** ❌ Bad

---

## What Was Changed
`transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])` was changed to
`transforms.Normalize(mean=[0.5], std=[0.5])` — both arguments reduced from 3-element lists to single-element lists, simulating a copy-paste error from a grayscale pipeline into an RGB one.

## Original Code
```python
transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
```

## Buggy Code
```python
transforms.Normalize(mean=[0.5], std=[0.5])  # <-- BUG
```

## Error Produced
```
ValueError: mean and std must each have 3 elements (one per channel),
but got mean=[0.5] (len=1) and std=[0.5] (len=1)
```

## Initial Self-Reflection (Shallow — GenAI labeled ❌ Bad)
> "I changed the normalize values from three-element lists down to single scalars. I think something might break because the numbers look different now and maybe PyTorch does not expect them to be that short, but I am not entirely sure what the actual error will be or exactly where in the code it will happen."

## GenAI Socratic Guidance
> "Your reflection identifies that something might break but does not explain the underlying contract that Normalize enforces between its arguments and the input tensor. Consider this: what property of the image tensor determines exactly how many values mean and std must each contain? Think about what 'channels' means in the context of an RGB image and how PyTorch applies the normalization formula independently for each channel. Once you can answer that, you will know precisely why a single-element list fails and exactly where the error is raised."

## New Self-Reflection (After Guidance)
> "After working through the guidance, I understand that transforms.Normalize applies the formula output[c] = (input[c] - mean[c]) / std[c] independently for each color channel c. An RGB image has three channels — Red, Green, and Blue — so mean and std must each be length-3 sequences providing one value per channel. By passing a single-element list [0.5], PyTorch cannot index mean[1] or mean[2] for the G and B channels, and raises a ValueError immediately on the first batch. The fix is to restore both arguments to their three-element form: mean=[0.5, 0.5, 0.5] and std=[0.5, 0.5, 0.5]."

## Fix
```python
transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
```

## Lesson Learned
Normalize has an implicit per-channel contract. The length of mean and std must always equal the number of image channels. Adapting grayscale code for RGB without updating these lists is a very common source of this error.
