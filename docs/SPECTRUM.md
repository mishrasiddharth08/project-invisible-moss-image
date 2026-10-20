# Optional Spectrum-style acceleration

This page describes the original **Balanced** mode. New inline Manual/Adaptive
hidden-state forecasting and the DMD preset are documented in [SPEED.md](SPEED.md),
including their separate upstream license and current validation boundary.

MOSS did not previously use Spectrum. This package adds an independent,
velocity-level adaptation of the Chebyshev/ridge forecasting idea in
[Spectrum](https://github.com/hanjq17/Spectrum). It is not the upstream
feature-forecasting implementation or a claim of official MOSS support.

After restarting Forge: **Settings > Stable Diffusion > [MOSS] Spectrum
acceleration > Balanced**, then Apply settings. **Off** is the default and
retains full model inference. The control applies only to MOSS.

Balanced is an approximation: identical seeds can produce different pixels.
It supports single-pass Euler with the MOSS schedule. Other samplers, hires,
custom attention patches and ControlNet inputs take the full-inference path.
Short runs and forecasts failing the guards also run the actual model.

## Behavior

- Independent positive/negative CFG histories; at most two histories per pass.
- Degree-one Chebyshev ridge fit, four real anchors, ridge 0.01, blend 0.75.
- First eight and final six steps always real; at most one forecast every three
  steps. Forecasts are never added back into the history.
- Nonfinite, excessive-coefficient or excessive-change predictions fall back
  to a real forward. Branch/layout keys stay separate; changed conditioning
  or a branch's geometry/device/skip-layer configuration restarts warmup.
- Combined retained histories/conditioning are capped at 64 MiB. Temporary
  arithmetic allocations are additional; this is not a total VRAM cap.
- Native per-pass ModelPatcher clones preserve fixed LoRA/LoKr patches. Batch
  completion and request closure release tensors, the denoiser reference and
  the owned model option. Existing request-close hooks are preserved.
- PNG metadata records actual real/forecast/rejected calls and cache size.

## Measured evidence: RTX 5090, SFT INT8 ConvRot

Matched 1024-square requests, Euler/MOSS, 50 steps, CFG 3, seed 123456, full
installed extension set. Timing starts after conditioning and ends after
sampling/decode; it excludes prompt encoding, saving and API transport. Exact
conditioning/attention shapes were warmed. Cross-request conditioning cache
was disabled in the controlled server during the final experiment; the helper
now sets that policy before startup for reproducibility.

| Case | Off seconds | Balanced seconds | Less time | Retained cache |
| --- | ---: | ---: | ---: | ---: |
| Woman's portrait | 20.6386 | 15.3419 | 25.66% | 8.69 MiB |
| Hands holding a cup | 20.0694 | 15.2931 | 23.80% | 8.49 MiB |

Both Balanced runs used 76 real and 24 forecast DiT calls, with zero rejected
forecasts. Text, portrait and hands pairs were visually inspected locally:
composition, legible text, facial features and hand structure stayed close,
with small texture/detail differences. This limited comparison cannot promise
unchanged identity, anatomy, text quality or performance on every prompt.

Exploratory end-to-end timings included prompt-cache advantages and are not
used for the speedup claim. A longer early sequence stalled; request references
retained by the first cleanup implementation were removed. The corrected
multi-request sequence completed, and a weak-reference regression verifies
that a resident model cannot retain the closed request through Spectrum.

Tested quantization: the released INT8 ConvRot SFT/encoder pair. Other weight
formats, full-size adapter combinations, img2img quality and physical smaller
GPUs need separate live tests. Keep Off for exact full-inference comparisons.

Reproduce with `tools/ui_check.py --forge "<Forge>" --output "<results>"
--spectrum-benchmark --spectrum-cases portrait,hands`. Output images stay local;
the GitHub README continues to contain only the requested face portrait.
