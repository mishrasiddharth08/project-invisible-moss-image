# Adapter validation

MOSS uses Forge Neo's normal adapter loader. The extension adds MOSS names to
Forge's key map; Forge itself parses and applies LoRA, LoKr, LoHa, GLoRA, BOFT,
DoRA and supported diff adapters.

Validated with tiny synthetic weights against the installed Forge backend:

- LoRA changes only the requested q third of a fused qkv weight.
- LoKr changes only the requested k third.
- LoRA is merged into a real INT8 ConvRot tensor and requantized by Forge.
- Unfused weights use the same native path without slice offsets.

Run the native checks from the Forge root:

```text
venv\Scripts\python.exe -m unittest discover -s "<extension>\tests" -p test_native_adapters.py -v
```

These tests prove key routing and numerical merging for small synthetic
adapters. The optional latest DMD LoRA is now recognized with a pinned identity;
its 252 targets and dimensions match both shipped INT8 and BF16 checkpoint
headers. Its INT8-route runtime loading, repeat determinism, pixel effect and exact
unpatch recovery passed bounded live tests; see [SPEED.md](SPEED.md). Broad quality
claims and full BF16 adapter runtime tests remain unverified. Quantized numerical evidence covers the released INT8 ConvRot route;
other quantization formats require their own real-file checks.
