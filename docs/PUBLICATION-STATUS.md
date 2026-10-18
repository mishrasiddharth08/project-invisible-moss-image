Current status: [Post-training validation](POST-TRAINING-20261018.md) supersedes the earlier pause below.

# Publication status — 2026-10-18

GPU validation is paused while AI Toolkit training runs. No Forge restart, CUDA import/allocation, dependency installation or model download was performed for this publication. Resume only after confirmed training completion and safe Forge idle checks; elapsed time or a single utilization sample is insufficient.

Current focused checks: 12 standard-library tests passed for assets, encoder shape rejection, AutoLink and wrapped-loader discovery. These complement the earlier 54 CPU and four numerical adapter checks; they do not rerun generation. Prior real INT8 checkpoint + Qwen3.5 encoder + FLUX2 VAE generation succeeded on RTX 5090. BF16 and physical smaller-card acceptance remain open. Full blueprint certification is not claimed.

## Post-training acceptance and second publication

After training: one job at a time. MOSS first: cold/warm fixed-seed generation, dropdown/AutoLink selection, output/gallery/save metadata, cancellation/recovery, release and repeated-run memory. FLUX3 second: preserve original FP8 baseline, then opt-in workflow alpha/64-bit seeds/layout with official captions, references and failure cases. BF16/INT8 real-weight profiles require matching available assets and remain blocked when absent. Refiner comes third: valid ordered target profiles with explicit matching TE/VAE, callback/negative-prompt boundaries, release and restore. Record cold/warm wall time, peak GPU and host RAM, actual outputs/hashes, failure recovery and repeated-run leaks. Allocator budgets on RTX 5090 are simulations, not physical 6/8/10/12/16/20/24/32 GB card certification. No numerical visual/performance threshold has been supplied. Publish subsequent reviewed fixes and measured results to these same repositories after testing; never include private prompts/images/logs or weights.
