# MOSS-Image support for Forge Neo
# What the loader hook checks and cleans in the selected files before Forge builds the components. No Forge
# imports: the CPU tests run it on the real checkpoints' headers.

# the Qwen3.5-9B language tower: 32 layers, hidden 4096, a Gated DeltaNet in layer 0 and full attention in layer 3.
# Comfy-Org's qwen3.5_9b_* files store it as model.language_model.* (with the vision tower, the multi-token
# prediction head and lm_head beside it), transformers' Qwen3_5Model as language_model.*, a text-only export as model.*
TE_PREFIXES = ("model.language_model.", "language_model.", "model.")
TE_KEYS = ("layers.0.linear_attn.A_log", "layers.3.self_attn.q_norm.weight", "norm.weight")
TE_NORM = "norm.weight"
TE_HIDDEN = 4096
TE_MTP_KEY = "mtp.fc.weight"
MISSING_TE = ("MOSS-Image needs its Qwen3.5 9B text encoder: select qwen3.5_9b_int8_convrot.safetensors under "
              "VAE / Text Encoder")
WRONG_TE = ("MOSS-Image needs the Qwen3.5 9B text encoder, and the Qwen3.5 selected under VAE / Text Encoder is "
            "another size: select qwen3.5_9b_int8_convrot.safetensors")
WRONG_TE_MODEL = ("MOSS-Image cannot validate this Qwen3.5 multimodal encoder (its MTP head is missing): "
                  "select the official qwen3.5_9b_int8_convrot.safetensors")
MISSING_VAE = "MOSS-Image needs the FLUX.2 autoencoder: select flux2-vae.safetensors under VAE / Text Encoder"
VAE_SHAPES = {
    "encoder.conv_in.weight": (128, 3, 3, 3),
    "encoder.conv_out.weight": (64, 512, 3, 3),
    "decoder.conv_in.weight": (512, 32, 3, 3),
    "decoder.conv_out.weight": (3, 128, 3, 3),
    "quant_conv.weight": (64, 64, 1, 1),
    "post_quant_conv.weight": (32, 32, 1, 1),
    "bn.running_mean": (128,),
    "bn.running_var": (128,),
}

# per-block attention settings ComfyUI stores next to the INT8 weights ({"attention": "comfy_kitchen_int8"}): a
# ComfyUI kernel choice, not a weight. rope.bands is upstream's RoPE frequency buffer, a constant the DiT derives
DROPPED_SUFFIXES = (".comfy_attention.config",)
DROPPED_KEYS = ("rope.bands",)


def shape_of(value):
    # a torch tensor, or a safetensors header entry {"dtype", "shape", "data_offsets"}
    if isinstance(value, dict):
        return tuple(value["shape"])
    return tuple(value.shape)


def text_encoder_prefix(state_dict) -> str:
    """The prefix of the Qwen3.5 language tower's keys, or the user-facing error when the file is not one."""
    if not isinstance(state_dict, dict) or len(state_dict) <= 16:
        raise ValueError(MISSING_TE)
    prefix = next((p for p in TE_PREFIXES if all(f"{p}{k}" in state_dict for k in TE_KEYS)), None)
    if prefix is None:
        raise ValueError(MISSING_TE)
    if shape_of(state_dict[f"{prefix}{TE_NORM}"])[0] != TE_HIDDEN:
        raise ValueError(WRONG_TE)
    # Comfy-Org's validated release includes its vision tower, LM head and MTP head. A local Qwen-Image 2.1 PE file
    # kept the first two but removed MTP. That shape difference cannot prove weight provenance, so fail closed for
    # such multimodal exports; standalone language-tower exports remain structurally valid.
    if prefix == "model.language_model." and any(k.startswith("model.visual.") for k in state_dict):
        if "lm_head.weight" in state_dict and TE_MTP_KEY not in state_dict:
            raise ValueError(WRONG_TE_MODEL)
    return prefix


def prepare_text_encoder(state_dict: dict) -> dict:
    """The language tower under model.* (Qwen35TextModel's names), every other part of the file left out. Keys move
    out of the given dict, so the vision tower and the heads are not kept alive by it."""
    prefix = text_encoder_prefix(state_dict)
    keys = [k for k in state_dict if k.startswith(prefix) and not k.startswith(f"{prefix}visual.")]
    return {f"model.{k[len(prefix):]}": state_dict.pop(k) for k in keys}


def check_vae(state_dict) -> None:
    if not isinstance(state_dict, dict) or any(key not in state_dict for key in VAE_SHAPES):
        raise ValueError(MISSING_VAE)
    try:
        valid = all(shape_of(state_dict[key]) == expected for key, expected in VAE_SHAPES.items())
    except (KeyError, TypeError, ValueError):
        valid = False
    if not valid:
        raise ValueError(MISSING_VAE)


def clean_transformer_state_dict(state_dict) -> None:
    for key in [k for k in state_dict if k.endswith(DROPPED_SUFFIXES) or k in DROPPED_KEYS]:
        del state_dict[key]
