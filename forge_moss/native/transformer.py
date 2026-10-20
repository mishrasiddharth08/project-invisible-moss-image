# MOSS-Image support for Forge Neo
# The MOSS-Image DiT, ported from the official pipeline (OpenMOSS-Team/MOSS-Image-SFT, pipeline_moss_image.py,
# MOSSImageTransformer; see MOSS_IMAGE_REVISION in native/__init__.py).
#
# The prompt goes through a projection and two text-only refinement blocks; then 42 single-stream blocks run the
# joint sequence [image, text] under one adaLN modulation shared by every block (computed once from the time). Each
# block is attention + a SiTU-gated MLP, with per-head RMS norms on q and k and a 4-axis RoPE: (r, f, h, w), the image
# centred on its grid at f = text length, the text along f. The final layer reads the image tokens only.
#
# Upstream's guidance skips block 22 in the negative-prompt pass of CFG ("uncond_skip_layers"): Forge runs the
# positive and negative passes as one batch or two and says which is which in transformer_options["cond_or_uncond"],
# so the skipped blocks are skipped per sample (settings.uncond_skip_layers).
#
# Layers are plain torch modules: inside Forge's using_forge_operations they become Forge operations (dtype,
# INT8 ConvRot quantization, offload, LoRA), outside it they stay torch modules for CPU tests. Differences from
# upstream: a batch runs together instead of sample by sample (Forge never batches prompts of different lengths for
# this model, see patches._hook_cond_concat, so no sample needs a text mask); Forge's attention runs through
# kernels.attention; under float16 the residual stream stays in float32.

import torch
import torch.nn as nn

from . import kernels, settings

TIME_DIM = 256
EPS = 1e-6
UNCOND = 1  # backend.sampling.sampling_function.UNCOND


def time_embedding(sigma: torch.Tensor, dim: int = TIME_DIM) -> torch.Tensor:
    # upstream's timesteps are the scheduler's float32 sigmas times 1000; the sinusoid is computed in float64 and fed
    # in float32
    times = (sigma.float() * 1000.0).double()
    half = dim // 2
    freq = 10000 ** (-torch.arange(half, device=sigma.device, dtype=torch.float64) / half)
    phase = times[:, None] * freq
    return torch.cat((phase.cos(), phase.sin()), dim=-1).float()


def rope_table(height: int, width: int, text_length: int, head_dim: int, base: float, device) -> torch.Tensor:
    """(H*W + T, head_dim): sin then cos of the four axes' phases, image tokens first, in float32."""
    num_bands = head_dim // 8
    bands = 1 / (base ** (torch.arange(num_bands, dtype=torch.float32) / num_bands))
    bands = bands.to(device)
    h, w = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing="ij")
    image = torch.stack((torch.zeros_like(h), torch.full_like(h, text_length), h, w), -1).flatten(0, 1).float()
    image[:, 2] -= (height - 1) / 2
    image[:, 3] -= (width - 1) / 2
    text = torch.zeros(text_length, 4, device=device, dtype=torch.float32)
    text[:, 1] = torch.arange(text_length, device=device)
    phase = torch.cat((image, text), 0).unsqueeze(-1) * bands
    return torch.cat((phase.sin(), phase.cos()), -2).flatten(-2)


def rotate(x: torch.Tensor, rope: torch.Tensor) -> torch.Tensor:
    # x (B, L, H, D), rope (L, D): rotation by halves, in x's dtype (upstream casts the table to the activations')
    sin, cos = rope.chunk(2, -1)
    sin = torch.cat((sin, sin), -1)[:, None]
    cos = torch.cat((cos, cos), -1)[:, None]
    a, b = x.chunk(2, -1)
    return x * cos + torch.cat((-b, a), -1) * sin


def attention(q, k, v, rope, heads, transformer_options=None):
    """q, k, v (B, L, H*D) -> (B, L, H*D), RoPE on q and k, every token attending to every other."""
    shape = (*q.shape[:2], heads, q.shape[-1] // heads)
    q = rotate(q.reshape(shape), rope).transpose(1, 2)
    k = rotate(k.reshape(shape), rope).transpose(1, 2)
    v = v.reshape(shape).transpose(1, 2)
    return kernels.attention(q.to(v.dtype), k.to(v.dtype), v, transformer_options=transformer_options)


def situ(hidden, gate):
    return (4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)) * (25.0 * torch.tanh(hidden / 25.0))


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # torch.ones, not empty: the checkpoint overwrites it
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        y = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + EPS)
        return y.to(x.dtype) * kernels.cast_to(self.weight, x.dtype, x.device)


def head_norm(norm, x, heads):
    return norm(x.unflatten(-1, (heads, -1))).flatten(2)


class TextBlock(nn.Module):
    """Text refinement: pre-norm attention and SiTU MLP, no modulation, biases on every projection."""

    def __init__(self, dim, ffn_dim, heads):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = RMSNorm(dim), RMSNorm(dim)
        self.q_norm, self.k_norm = RMSNorm(dim // heads), RMSNorm(dim // heads)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.ffn_gate = nn.Linear(dim, ffn_dim)
        self.ffn_up = nn.Linear(dim, ffn_dim)
        self.ffn_down = nn.Linear(ffn_dim, dim)

    def forward(self, x, rope, dtype, transformer_options=None):
        q, k, v = self.qkv(self.norm1(x).to(dtype)).chunk(3, -1)
        q, k = head_norm(self.q_norm, q, self.heads), head_norm(self.k_norm, k, self.heads)
        x = x + self.proj(attention(q, k, v, rope, self.heads, transformer_options).to(dtype)).to(x.dtype)
        y = self.norm2(x).to(dtype)
        return x + self.ffn_down(situ(self.ffn_up(y), self.ffn_gate(y))).to(x.dtype)


class TextProjection(nn.Module):
    def __init__(self, text_dim, dim, ffn_dim, heads):
        super().__init__()
        self.proj = nn.Linear(text_dim, dim)
        self.blocks = nn.ModuleList([TextBlock(dim, ffn_dim, heads) for _ in range(2)])

    def forward(self, context, rope, dtype, res_dtype, transformer_options=None):
        x = self.proj(context.to(dtype)).to(res_dtype)
        for block in self.blocks:
            x = block(x, rope, dtype, transformer_options)
        return x


class Block(nn.Module):
    def __init__(self, dim, ffn_dim, heads, fused_qkv=False):
        super().__init__()
        self.heads = heads
        self.fused_qkv = fused_qkv
        self.norm1, self.norm2 = RMSNorm(dim), RMSNorm(dim)
        self.q_norm, self.k_norm = RMSNorm(dim // heads), RMSNorm(dim // heads)
        # Comfy-Org's files fuse q, k, v into one projection; upstream's diffusers export keeps them apart
        names = ("qkv", "o") if fused_qkv else ("q", "k", "v", "o")
        self.proj = nn.ModuleDict({n: nn.Linear(dim, 3 * dim if n == "qkv" else dim, bias=False) for n in names})
        self.proj["ffn_up_gate"] = nn.Linear(dim, 2 * ffn_dim, bias=False)
        self.proj["ffn_down"] = nn.Linear(ffn_dim, dim, bias=False)

    def forward(self, x, rope, mod, dtype, transformer_options=None):
        # mod (B, 6, D): attention shift, scale, gate, then the MLP's
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = (m[:, None] for m in mod.unbind(1))
        y = (self.norm1(x) * (1 + scale_a) + shift_a).to(dtype)
        if self.fused_qkv:
            q, k, v = self.proj["qkv"](y).chunk(3, -1)
        else:
            q, k, v = self.proj["q"](y), self.proj["k"](y), self.proj["v"](y)
        q, k = head_norm(self.q_norm, q, self.heads), head_norm(self.k_norm, k, self.heads)
        out = self.proj["o"](attention(q, k, v, rope, self.heads, transformer_options).to(dtype))
        x = x + (out * gate_a).to(x.dtype)
        y = (self.norm2(x) * (1 + scale_f) + shift_f).to(dtype)
        hidden, gate = self.proj["ffn_up_gate"](y).chunk(2, -1)
        return x + (self.proj["ffn_down"](situ(hidden, gate)) * gate_f).to(x.dtype)


class MossImageDiT(nn.Module):
    def __init__(self, dim=3072, ffn_dim=8192, num_heads=24, num_layers=42, text_dim=4096, freq_dim=TIME_DIM,
                 latent_channels=128, rope_base=10000.0, fused_qkv=False, **kwargs):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.latent_channels = latent_channels
        self.freq_dim = freq_dim
        self.rope_base = rope_base
        self.video_embedder = nn.Linear(latent_channels, dim)
        self.text_embedder = TextProjection(text_dim, dim, ffn_dim, num_heads)
        self.time_embedding = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.shared_block_modulation = nn.Linear(dim, 6 * dim)
        self.layers = nn.ModuleList([Block(dim, ffn_dim, num_heads, fused_qkv) for _ in range(num_layers)])
        self.final_norm_video = RMSNorm(dim)
        self.final_linear_video = nn.Linear(dim, latent_channels, bias=False)
        self.video_head_modulation = nn.Linear(dim, 2 * dim)
        # One geometry only; no prompt states or model weights are retained here.
        self.register_buffer("_rope_cache", None, persistent=False)
        self._rope_key = None

    def positional_encoding(self, height, width, text_length, device, dtype):
        key = (height, width, text_length, self.dim // self.num_heads, self.rope_base, device, dtype)
        if (self._rope_key != key or self._rope_cache is None
                or self._rope_cache.device != device or self._rope_cache.dtype != dtype):
            self._rope_cache = rope_table(height, width, text_length, self.dim // self.num_heads,
                                          self.rope_base, device).to(dtype)
            self._rope_key = key
        return self._rope_cache

    def skipped(self, transformer_options, batch, device):
        """(B,) True where block skipping applies (the negative-prompt pass), or None when nothing is skipped."""
        layers = settings.uncond_skip_layers(len(self.layers))
        chunks = (transformer_options or {}).get("cond_or_uncond")
        if not layers or not chunks or UNCOND not in chunks or batch % len(chunks):
            return None
        per_chunk = batch // len(chunks)
        flags = torch.tensor([c == UNCOND for c in chunks], device=device).repeat_interleave(per_chunk)
        return layers, flags, all(c == UNCOND for c in chunks)

    def forward(self, x, timestep, context, transformer_options=None, **kwargs):
        session = (transformer_options or {}).get('moss_spectrum_session')
        if session is not None:
            return session.run(self._forward_full, x, timestep, context,
                               transformer_options=transformer_options, **kwargs)
        return self._forward_full(x, timestep, context, transformer_options=transformer_options, **kwargs)

    def _forward_full(self, x, timestep, context, transformer_options=None, **kwargs):
        """x (B, 128, H, W) latents, timestep (B,) = sigma, context (B, T, 4096) -> velocity (B, 128, H, W)."""
        b, _, h, w = x.shape
        device = x.device
        # Forge hands the inputs over in the computation dtype (k_model.apply_model); the weights may be stored in
        # another one (INT8, manual cast), so the context is what says which
        dtype = context.dtype
        # fp16 overflows the residual stream of large DiTs (the Iris and FLUX 3 ports saw it); float32 there
        res_dtype = torch.float32 if dtype == torch.float16 else dtype
        context = context.to(device=device, dtype=dtype)
        if context.shape[0] != b:
            context = context.expand(b, -1, -1)

        time = self.time_embedding(time_embedding(timestep.reshape(-1).expand(b), self.freq_dim).to(dtype)).float()
        mod = self.shared_block_modulation(time.to(dtype)).view(b, 6, -1)
        head = self.video_head_modulation(time.to(dtype)).view(b, 2, -1)

        image_tokens = h * w
        rope = self.positional_encoding(h, w, context.shape[1], device, dtype)
        video = self.video_embedder(x.flatten(2).transpose(1, 2).to(dtype)).to(res_dtype)
        text = self.text_embedder(context, rope[image_tokens:], dtype, res_dtype, transformer_options)
        tokens = torch.cat((video, text), 1)
        del video, text

        skip = self.skipped(transformer_options, b, device)
        for index, layer in enumerate(self.layers):
            if skip is not None and index in skip[0]:
                if skip[2]:
                    continue
                tokens = torch.where(skip[1][:, None, None], tokens,
                                     layer(tokens, rope, mod, dtype, transformer_options))
            else:
                tokens = layer(tokens, rope, mod, dtype, transformer_options)

        out = self.final_norm_video(tokens[:, :image_tokens]) * (1 + head[:, 1, None]) + head[:, 0, None]
        out = self.final_linear_video(out.to(dtype))
        return out.transpose(1, 2).reshape(b, self.latent_channels, h, w)

    def forecast_head(self, x, timestep, dtype, hidden):
        """Recompute the native timestep-conditioned head on forecast image tokens."""
        b, _, h, w = x.shape
        time = self.time_embedding(time_embedding(timestep.reshape(-1).expand(b), self.freq_dim).to(dtype)).float()
        head = self.video_head_modulation(time.to(dtype)).view(b, 2, -1)
        res_dtype = torch.float32 if dtype == torch.float16 else dtype
        hidden = hidden.to(device=x.device, dtype=res_dtype)
        out = self.final_norm_video(hidden) * (1 + head[:, 1, None]) + head[:, 0, None]
        return self.final_linear_video(out.to(dtype)).transpose(1, 2).reshape(b, self.latent_channels, h, w)
