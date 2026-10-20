# Project Invisible MOSS-Image for Forge Neo

Native MOSS generation using Forge Neo's checkpoint selector, **moss** preset,
normal **Generate**, img2img, saving and adapter system. No extra environment,
server, private model panel or Forge core-file edits.

Based on [Eduardo Abreu's native integration](https://github.com/eduardoabreu81/moss-image-forge-neo)
at `ce01661dce6626464607382112b3fa05af432075`. Original licenses, notices and
reference tests are retained. Local improvements reuse one position table,
remove an unnecessary GPU synchronization, strengthen component validation,
and add verified downloads and reproducible native checks.

## Install

1. In Forge Neo's Extensions page, install from:
   `https://github.com/mishrasiddharth08/project-invisible-moss-image`.
2. Restart. Use an up-to-date Forge **Neo**.
3. Download matching files from [Comfy-Org/MOSS-Image](https://huggingface.co/Comfy-Org/MOSS-Image):

| File | Forge folder |
| --- | --- |
| `moss_image_sft_int8_convrot.safetensors` | `models/Stable-diffusion/MOSS` |
| `qwen3.5_9b_int8_convrot.safetensors` | `models/text_encoder/MOSS` |
| `flux2-vae.safetensors` | `models/VAE/MOSS` |

Optional setup helper: `tools/download_models.py --forge "<Forge folder>"`.
Run with Python 3.11+. It downloads about 15.5 GB, resumes partial files, and
checks fixed publisher SHA256 hashes at revision
`6c170adf6af419f9877872ae9c7d397fcb341b99`. No generation-time downloads or
shared GPU-library changes. Do not reuse a Qwen-Image prompt-enhancer encoder.

4. Choose **moss**, the SFT checkpoint, and both components under
   **VAE / Text Encoder**. Leave **Diffusion in Low Bits** at **Automatic**;
   the files already contain INT8 ConvRot weights.
5. Start with **Euler**, **MOSS**, **50 steps**, **CFG 3**, **1024×1024**,
   batch 1, empty negative prompt. Use normal **Generate**.

## Measured speed and memory

RTX 5090, Forge `neo-2.30`, Torch `2.13.0+cu130`, SFT INT8 ConvRot and original
Qwen3.5 INT8 ConvRot encoder. Individual native runs. Seconds include model
loading/transfers, encoding, sampling, decoding and saving; exclude app startup.
Memory is the process's peak allocated GPU memory.

| Request | Seconds | Peak GiB |
| --- | ---: | ---: |
| txt2img, 1024×1024, 50 steps | 39.96 | 14.82 |
| Portrait, 2048×2048, 50 steps | 115.01 | 18.87 |
| JSON-request portrait, 1248×1664, 50 steps | 61.09 | 16.91 |
| img2img, 512×512, 12 requested steps, strength 0.6 | 13.72 | 13.22 |

These are measurements, not speedup or physical low-VRAM claims. Forge owns
memory, offloading, quantization and attention. The bounded position buffer
follows model device moves. An alternative Qwen attention implementation was
benchmarked, found slower, and discarded.

The [official guide](https://github.com/OpenMOSS/MOSS-Image/blob/main/USERGUIDE.md)
recommends roughly 1–2 MP; 2048 square is its documented 4MP extrapolation
endpoint. Both sides must be positive multiples of 16. Batch 1 is conservative.

## Optional Spectrum acceleration

Spectrum-style velocity forecasting is now available under **Settings > Stable
Diffusion > [MOSS] Spectrum acceleration > Balanced**. Restart after updating.
Off is the default; Balanced approximates selected steps and can change pixels.
Matched RTX 5090 tests reduced sampling/decode time by **24–26%**, with about
**8.5 MiB** retained cache at 1024 square. This is not a full-request speedup
claim. See [method, measurements and limits](docs/SPECTRUM.md).

## Actual local outputs

![2048-square SFT portrait](docs/examples/portrait-2048.png)

The helper maps JSON `prompt`, `negative_prompt`, `style` and `aspect_ratio` to
normal request fields. This is a request-wrapper test, not raw JSON/layout
interpretation. The supplied `model: ideogram-4` was explicitly overridden for
this MOSS comparison and recorded alongside the actual model. In the GUI, use
normal prompt/negative/size controls. The JSON output matched pose, clothing,
cars and cool grain well but cropped the boots. Source preservation cannot be
tested without a source image. Not every instruction passed.

## Quantization and adapters

**Live validated:** SFT INT8 ConvRot pair. BF16/FP16/FP32 have structural support.
Other Forge-native formats have different evidence levels; GGUF, FP8, NVFP4
and W4A4 are untested for MOSS. GPTQ/AWQ/NF4 and Nunchaku are unsupported by
this native route. See the [format matrix](docs/QUANTIZATION.md).

Use the normal Forge LoRA picker. Four actual Forge numerical tests cover
fused/unfused LoRA, fused LoKr and INT8 ConvRot LoRA requantization. Full-size
released MOSS adapters, other adapter types and all combinations remain
untested. See [adapter evidence](docs/ADAPTERS.md).

## Validation and limits

53 regression tests plus four actual Forge adapter tests passed. Tiny-model
CUDA reference checks passed in FP32/BF16. Live txt2img/img2img checked saved
pixels against returned images, metadata, checkpoint and actual MOSS engine.
Portraits were visually inspected.

Initial tests used native processing with third-party extensions disabled.
The installed full extension set subsequently passed real UI startup, MOSS
dropdown/component selection and a saved 512px HTTP API generation. Loader
compatibility now follows cooperative wrappers, and optional Model Autolink
registration distinguishes MOSS's Qwen3.5 encoder from Qwen-Image encoders.
Renamed compatible assets are discovered and stale MOSS preset names repaired.
Browser gallery interaction, complete HTTP API parity, all third-party combinations,
cancellation, masks/outpaint, hires and model-switch recovery need separate
checks. Base/BF16 full images and physical smaller GPUs were not tested.
Registration uses internal version-bound hooks with startup compatibility checks.
See [full evidence](docs/VALIDATION.md).

To disable: choose another preset, remove this extension folder, restart Forge.
Separate model files remain. Restart after extension updates.

## Credits and license

- Eduardo Abreu: original native Forge integration and tests.
- [OpenMOSS](https://github.com/OpenMOSS/MOSS-Image): model and reference pipeline.
- [Forge Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo): host.
- ComfyUI / Comfy-Org: encoder implementation, packaging and quantization kernels.
- Qwen team and Hugging Face Transformers: encoder and reference implementation.

Extension: **AGPL-3.0**, [LICENSE](LICENSE). Preserved third-party terms:
[NOTICE](NOTICE), [LICENSE-APACHE](LICENSE-APACHE). Weights are separate; check
their publisher licenses. No credentials or model weights are included.

## Latest bounded support audit

54 CPU regressions and four native numerical adapter tests passed after strengthening
encoder norm rank/shape validation. Existing INT8 checkpoint/encoder outputs remain
exactly equal to the prior 1024-square 12-step fixed-seed baseline after reload.
The earlier audit found no compatible trained MOSS adapter among 622 local headers;
the newly supported optional DMD LoRA has a separate validation boundary below.
BF16 full-model and physical smaller-card tests remain blocked/untested. Earlier
6-32 GiB Forge reservation runs are soft-policy simulations, not hard allocation caps
or physical-card certification. Actual clicked UI acceptance remains unverified.

Current release boundary and post-training plan: [Publication status](docs/PUBLICATION-STATUS.md).

Current post-training evidence: [2026-10-18 validation](docs/POST-TRAINING-20261018.md). Training completed; bounded real generation tests passed. Remaining format, hardware and full-acceptance limits still apply.

Optional inline Spectrum Manual/Adaptive controls and the pinned 8-step SFT DMD
LoRA preset: [setup, upstream attribution and current validation limits](docs/SPEED.md).

Latest bounded runtime results: [2026-10-20 inline Spectrum and DMD verification](docs/LIVE-20261020.md). Includes cancellation/error recovery and measured speed limits; no universal quality or hardware certification.
