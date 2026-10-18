# Post-training validation — 2026-10-18

Training completion and exclusive compute availability were verified before testing; jobs ran sequentially, FLUX3 then MOSS then Refiner. No downloads.

54 installed CPU regressions and four numerical tests against the installed Forge backend passed. Three actual native Gradio Generate calls used MOSS SFT INT8 ConvRot, matching Qwen3.5-9B INT8 ConvRot encoder and FLUX2 VAE,1024x1024,12steps, seed20261017, Euler/MOSS, Spectrum Off. Cold/warm/repeat times: 50.231s, 13.122s, 12.108s. These native request timings include temporary profile overrides and cleanup, not isolated sampler timings. All three images match historical pixel SHA256 `fc20f1a628da7ce9c5dd462d0d44b629c8f0f7979c1102b276e5a059920ab8d7` exactly. Served gallery PNG and saved metadata were inspected; the red mug/green apple scene was visually coherent. Persisted user configuration remained unchanged.

Sampled whole-device and system RAM peaks (aggregate host/device, not isolated MOSS process or enforced caps): `{"sampled_whole_device_peak_gib": 18.21, "sampled_system_ram_used_peak_gib": 38.83}`. One-second sampling can miss brief peaks.

Actual native Gradio Generate is not a clicked-browser claim. BF16 checkpoint/encoder, physical6/8/10/12/16/20/24/32GB cards, a genuine matching trained adapter, long-duration leak testing and complete Blueprint acceptance remain open. Earlier VRAM policy runs on RTX5090 are simulations. Private prompts/images/raw logs and weights are excluded.

After further reviewed fixes or acceptance work, update these same repositories again; no universal certification is claimed.
