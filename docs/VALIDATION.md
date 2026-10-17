# Local validation - 17 October 2026

Base source: eduardoabreu81/moss-image-forge-neo at
`ce01661dce6626464607382112b3fa05af432075` (AGPL-3.0; upstream notices retained).
Model source: Comfy-Org/MOSS-Image at
`6c170adf6af419f9877872ae9c7d397fcb341b99`.

Environment: installed Forge Neo; Python 3.13; Torch 2.13.0+cu130;
NVIDIA RTX 5090, approximately 32 GiB VRAM; approximately 96 GiB RAM.

Passed:
- 32 CPU regression tests, including upstream numerical comparisons, fused
  projection shapes, scheduler, LoRA key routing and component contracts.
- Tiny transformer CUDA comparisons with reference in FP32 and BF16, including
  separate and batched negative-pass block skipping. This uses torch attention
  and does not establish full-sized ConvRot or Forge attention correctness.
- Actual installed Forge compatibility check and extension startup/registration,
  with built-in extensions enabled and other third-party extensions disabled.
- Publisher SHA256 verification of SFT checkpoint, Qwen3.5 encoder and FLUX.2 VAE.
- Downloaded checkpoint architecture and normalized VAE header contracts.

Optimization decisions:
- Keep a single nonpersistent positional table; key by geometry, prompt length,
  head size, RoPE base, device and dtype. The buffer follows model device moves.
- Decide all-negative batches in Python rather than synchronizing a GPU flag.
- Rejected Qwen GQA change: TURBO measured a regression on this runtime (512
  tokens: about 0.919 versus 0.058 ms and 111 versus 58 MiB). Original code kept.
  These are kernel microbenchmarks, not full generation speeds.
- No torch compilation, sampler approximations, layer deletion or automatic
  GPU-library upgrades added. Forge owns quantization, memory and attention.

Actual image results (INT8 ConvRot SFT + INT8 ConvRot original Qwen3.5 encoder):

| Native request | Seconds including model loading/transfers | Peak allocated / reserved GiB |
| --- | ---: | ---: |
| txt2img, 512 square, 4 steps | 28.48 | 13.22 / 13.48 |
| txt2img, 1024 square, 50 steps | 39.96 | 14.82 / 15.80 |
| portrait, 2048 square, 50 steps | 115.01 | 18.87 / see raw result |
| JSON wrapper, 1248x1664, 50 steps | 61.09 | 16.91 / see raw result |
| img2img, 512 square, 12 requested steps, strength 0.6 | 13.72 | 13.22 / 13.51 |

All used seed 123456, Euler, MOSS schedule, CFG 3, batch 1. Saved PNG pixels
matched returned gallery images; saved seed metadata and the actual MOSS engine
were asserted. Full 1024 output was visually inspected: red mug, wooden table,
and legible "MOSS WORKS" card matched the prompt. GPU peaks cover this process,
not global GPU occupancy. Timings are individual runs, not a speedup comparison.
Raw requests/results and PNGs are retained in the workspace's `moss-validation/`.

Four actual Forge adapter tests passed: fused LoRA, fused LoKr, unfused LoRA,
and INT8 ConvRot LoRA requantization. These use tiny synthetic weights, not
full-size released adapters.

Harness cleanup: initial one-shot runs produced valid files but hit Windows
0xC0000005 while an unclosed progress bar was finalized. The checker now closes
Forge total progress and calls sd_models.unload_model_weights before exit.
A subsequent fixed-seed generation exited with code 0. Forge core was unchanged;
this fix is confined to the one-shot checker.

Limits: internal, version-bound registration hooks are used; compatibility is
checked against this installed Forge, not promised for future versions. All
third-party extension combinations, LoRA files, Base/BF16 full model generation,
physical smaller GPUs, masks/outpaint, hires workflows, cancellation/recovery,
and browser/API parity require separate tests. These runs used native processing
directly, not the browser gallery or HTTP endpoints. Structure alone cannot prove
arbitrary model weight provenance. No physical low-VRAM claim is made.

## Full installed extension startup correction

The first package's isolated checks missed MiniMax H3's loader wrapper. Normal
startup inspected that wrapper's source and incorrectly disabled MOSS. The
compatibility check now follows reachable __wrapped__, closure, default and
partial functions, with cycle/size bounds. A single reachable function must
contain the full required contract; unsupported loaders still fail closed.

Optional MOSS-owned Model Autolink registration adds exact architecture
fingerprints and the correct encoder/preset policy in memory. It repairs only
the validated MOSS encoder's stale classification and changes no Autolink files.
Preset defaults validate actual local headers, handle Smart Rename paths and
repair stale MOSS-specific selections without changing other model presets.

43 CPU regressions passed; four real Forge adapter tests passed previously.
A subsequent real full GUI startup with all installed extensions exposed the
moss dropdown and correct components. A native HTTP txt2img request (512 square,
4 steps, seed 123456, Euler/MOSS/CFG3) returned a nonblank image with correct
MOSS metadata and saved output; request time was 17.10 seconds. This extends
the earlier isolated evidence, not every possible plugin/version combination.

## Spectrum-style velocity forecasting

53 CPU regressions passed, including real-only history, CFG separation,
conditioning invalidation, finite/change guards, memory limits, preservation of
other close hooks, and weak-reference proof that a resident model does not
retain closed requests. The earlier four actual Forge adapter checks remain
separate evidence; full-weight adapter/Spectrum combinations were not tested.

Corrected full-extension multi-request sequence completed with actual SFT INT8
ConvRot inference. Matched 1024-square, 50-step requests recorded sampling/decode
times of 20.6386 -> 15.3419 s (portrait) and 20.0694 -> 15.2931 s (hands).
Each accelerated request made 76 real and 24 forecast DiT calls; retained cache
was 8.69 and 8.49 MiB respectively. Portrait, hands and exploratory text pairs
were visually inspected locally. Details and limits: SPECTRUM.md.
