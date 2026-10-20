# Optional MOSS speed controls

The always-visible MOSS panel has inline controls; it creates no accordion.
They appear after a normal Forge restart. Existing global settings remain a
fallback for API callers without script arguments. All defaults are Off.

- **Balanced** retains the prior bounded velocity-forecast adapter.
- **Manual / Adaptive** forecast last-block image features and recompute the
  native timestep-conditioned output head. They use Rkkss' Chebyshev/ridge
  predictor and manual/adaptive scheduling, including warmup, detail guard,
  local p95/p99 drift checks, forecast caps and final actual steps. The predictor
  policy is extracted from [ComfyUI-Spectrum-MOSS](https://github.com/Rkkss/ComfyUI-Spectrum-MOSS/tree/3fde0674b29342565e7f42a0674af48d65aabc4a),
  GPL-3.0-only; see LICENSE-SPECTRUM and NOTICE. ComfyUI is not installed.
- Spectrum requires single-pass Euler + MOSS. Unsupported samplers, controls,
  patches, layouts and insufficient history use actual evaluations. Histories
  are request/CFG/conditioning specific and released on close, error or cancel.
  CPU feature caching is the default; GPU caching is optional. The cache limit
  bounds retained context/anchors, not total process memory or forecast scratch.
  A low limit may prevent acceleration. Forecasting changes results; no speed
  or quality ratio is promised. Upstream recommends 50 steps and tested Euler.

## Optional 8-step DMD preset

Select an **SFT** checkpoint and its matching MOSS encoder/VAE, then enable the
inline DMD checkbox. The preset sets Euler/MOSS, 8 steps, CFG 1 and adapter
strength 0.7 (adjustable). Spectrum is forced Off, preserving all eight actual
evaluations. Negative prompts are ignored at CFG 1. HiRes and additional prompt
LoRAs are rejected by this preset. Disable it to use ordinary manual LoRAs.
No settings or model weights are permanently modified or merged.

Download the public Apache-2.0 [latest DMD LoRA](https://huggingface.co/lodestones/MOSS-Image-SFT-8-steps-dmd/blob/f95df83b5e90ae91ea412434b3f9b7f1c8c6314f/lora_latest.safetensors)
only when authorized, and put it at
`models/Lora/MOSS/MOSS-Image-SFT-8steps-DMD-latest.safetensors`.
The original filename `lora_latest.safetensors` is also recognized. Refresh
LoRAs after installation. Size: 1,189,176,800 bytes; SHA256:
`be82ab2bef78194651815bb6de443cec67553e01c14559f2ad365449445f0452`.
The preset verifies the file and fails clearly when missing or mismatched.
All 252 projection targets in the supplied header map to Forge's fused QKV
adapter path. Existing Forge LoRA loading/release remains responsible for weights.

The [publisher's current demo](https://huggingface.co/spaces/hugging-apps/moss-image-8-step-dmd/blob/main/app.py)
uses ordinary flow-matching Euler/MOSS shift 5. The model card's old data-space
sampler section is explicitly deprecated; this integration does not use it.
Base checkpoints and the older generator/time-embedding/dataspace assets are
unsupported. The base checkpoint and encoder precision policy is unchanged.

## Validation boundary (2026-10-20)

CPU numerical checks cover the native head, real versus forecast scheduling,
CFG separation, cache limits, step rewind, hook removal on failure, DMD routing
and all 252 adapter targets. The installed Gradio panel is built in memory to
check controls and absence of accordions. CI runs the numerical subset on CPU.
The local full CPU suite passed 63 tests; the installed Forge adapter class was
skipped because it initializes the GPU backend. The numerical CI entry point
passed 22 tests locally and the existing no-Torch subset passed 12 (overlapping
suites, not additive). The real Gradio 4.40 panel was constructed successfully
with all 11 inline controls and no accordion. All 756 DMD adapter tensors were
read on CPU, float32 and finite, with 252 alpha values equal to 128. These checks
prove file loadability and routing dimensions, not inference or visual quality.

No new real-image, speed, visual-quality, INT8 DMD, BF16 or physical-card results
are claimed for this integration. During local preparation Forge reported busy;
its running process was preserved. The requested DMD LoRA was installed locally
and its complete file size and SHA256 verified; model execution is still pending.
UI restart, sequential fixed-seed Off/Manual/Adaptive comparisons and DMD
generation, timings, memory and cancellation/recovery remain pending. Historical
MOSS results elsewhere in this repository predate these features.
