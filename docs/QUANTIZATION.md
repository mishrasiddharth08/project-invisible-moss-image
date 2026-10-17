# MOSS-Image quantization in Forge Neo

Status: 2026-10-17, for the installed Forge Neo tree used by this extension.

The extension does not implement a second quantizer. Its DiT and Qwen3.5 layers are built through Forge operations, so a format works only when Forge recognizes the file, the tensor names/shapes still match MOSS-Image, and the installed GPU has the required kernel.

## Current support matrix

| Weight format | DiT | Qwen3.5 encoder | Evidence and limits |
|---|---|---|---|
| INT8 tensorwise + ConvRot (`comfy_quant`) | Live validated SFT | Live validated | RTX 5090: native 1024-square txt2img, 50 steps, 39.96 seconds including load/transfers, 14.82 GiB peak allocated. Native 512-square img2img also passed. Base is not locally live tested. Recommended low-memory route. |
| BF16 safetensors | Supported | Structurally supported | Official BF16 DiT header matches exactly. No completed live BF16 MOSS run is recorded here. |
| FP16 safetensors | Supported | Structurally supported | The MOSS DiT explicitly permits FP16 and keeps its residual stream in FP32 to avoid overflow. No official/local MOSS FP16 pair has been validated. |
| FP32 safetensors | Supported, impractical | Structurally supported, impractical | Ordinary Forge loading works, but weight memory is roughly twice BF16/FP16 and is not a useful 32 GB target. |
| FP8 E4M3/E5M2, raw or Forge/Comfy scaled metadata | Loader-supported; untested for MOSS | Loader-supported; untested for MOSS | Forge recognizes native FP8 and `comfy_quant` FP8. Canonical MOSS filenames, shapes, quality and live output are still unverified. |
| MXFP8 | Loader-supported; untested for MOSS | Conditional and untested | Requires supported hardware and correct block scales. Qwen embedding supports only FP8 E4M3/E5M2 or INT8 metadata, so a converter must leave the embedding compatible. |
| NVFP4 | Loader-supported; untested for MOSS | Conditional and untested | Requires NVFP4-capable hardware and valid dual scales. A fully NVFP4 Qwen file is not accepted by Forge's embedding loader. |
| ConvRot W4A4 (`convrot_w4a4`) | Loader-supported; untested for MOSS | Conditional and untested | Requires INT8-capable hardware. Linear layers are supported; the Qwen embedding must remain BF16/FP16/FP8/INT8. This is not the same format as INT4 ConvRot used by other projects. |
| Asymmetric W4A8 INT8 / W6A8 INT8 | Loader-supported; untested for MOSS | Conditional and untested | Packed width, group scales and codebook must match Forge. Qwen embedding must remain in a supported format. |
| GGUF Q2/Q3/Q4/Q5/Q6/Q8/BF16 | Forge backend supports these types; MOSS untested | Forge backend supports these types; MOSS untested | A MOSS-specific GGUF must preserve/remap the native tensor contract. No official/local MOSS GGUF header or live run proves this route. |
| `int4_convrot` / `convrotint4` files | Unsupported as named | Unsupported as named | These are not registered format names in this Forge loader. Do not relabel them as `convrot_w4a4` without an exact converter and numerical validation. |
| Plain INT8/INT4 tensors without scales and `comfy_quant` metadata | Unsupported | Unsupported | Forge has no information needed to reconstruct the weights correctly. |
| GPTQ, AWQ, bitsandbytes NF4/INT4 | Unsupported | Unsupported | Installed Forge does not detect these layouts for this native path; their `qweight/qzeros/scales` tensors do not match the MOSS model contract. |
| Nunchaku/SVDQuant | Unsupported | Unsupported | Forge's Nunchaku branches target named architectures, not MOSS-Image. |

`Supported` means the native path can load the contract. It does not mean every converter output is compatible. `Untested` must remain opt-in and must not be auto-selected.

## Memory and speed

- Weight storage is approximately 4 bytes/value for FP32, 2 for BF16/FP16, 1 for FP8/INT8, and 0.5 for 4-bit before scales and unquantized tensors. Actual VRAM also includes activations, prompt states, VAE, CUDA kernels and temporary buffers.
- INT8 ConvRot is the safest memory-first choice currently. Speed is kernel-, layer-shape- and offload-dependent; do not promise a fixed multiplier.
- Forge sets text-encoder mixed-precision matrix multiplies to full precision. Quantized Qwen weights reduce storage/offload pressure, while prompt compute is dequantized for compatibility. The DiT can use its quantized linear path directly.
- Batch 1 and lower pixel count reduce activation memory more reliably than changing weight format. The VAE should remain the validated FLUX.2 VAE.

## Minimal gaps before enabling more formats

1. Add a real header fixture for each exact MOSS DiT and encoder pair, then require strict state-dict shape matching.
2. Add one fixed-seed live run that checks prompt encoding, denoising, non-blank pixels, save/gallery/metadata and reload. Record peak VRAM and time; label quality separately.
3. For MXFP8/NVFP4/W4A4/W4A8/W6A8, reject a quantized Qwen embedding unless Forge's embedding loader supports that exact format. Mixed-format encoders are acceptable only when explicitly validated.
4. For GGUF, add MOSS-specific key mapping only after a real file shows a mismatch. Do not guess mappings.

GPTQ/AWQ/NF4 would require new backend operators and conversion rules, not a small extension patch. They should stay unsupported rather than silently dequantizing or substituting another model.
