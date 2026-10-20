"""Pinned optional DMD LoRA; uses Forge's existing LoRA lifecycle, never merges weights."""
import hashlib
import math
from pathlib import Path

FILENAME = 'MOSS-Image-SFT-8steps-DMD-latest.safetensors'
SHA256 = 'be82ab2bef78194651815bb6de443cec67553e01c14559f2ad365449445f0452'
SIZE = 1189176800
REVISION = 'f95df83b5e90ae91ea412434b3f9b7f1c8c6314f'
_verified = {}


def verified_adapter(networks):
    candidates = {id(n): n for n in networks.available_networks.values()
                  if Path(n.filename).name.lower() in {FILENAME.lower(), 'lora_latest.safetensors'}}
    for network in candidates.values():
        path = Path(network.filename)
        stat = path.stat()
        signature = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
        if signature not in _verified:
            digest = ''
            if stat.st_size == SIZE:
                with path.open('rb') as handle:
                    digest = hashlib.file_digest(handle, 'sha256').hexdigest()
            _verified[signature] = digest == SHA256
        if _verified[signature]:
            return network.name
    raise ValueError(f'MOSS DMD adapter missing or hash mismatch. Install the pinned lora_latest.safetensors '
                     f'as models/Lora/MOSS/{FILENAME}, then refresh LoRAs. See docs/SPEED.md.')


def apply(p, strength):
    if not getattr(p.sd_model, 'is_moss_image', False):
        raise ValueError('MOSS DMD requires a MOSS-Image SFT checkpoint.')
    checkpoint = str(getattr(getattr(p.sd_model, 'sd_checkpoint_info', None), 'filename', ''))
    if 'sft' not in Path(checkpoint).name.lower():
        raise ValueError('MOSS DMD requires the SFT checkpoint; Base and unidentified checkpoints are unsupported.')
    if not math.isfinite(float(strength)) or not 0 < float(strength) <= 2:
        raise ValueError('MOSS DMD strength must be finite and in (0, 2].')
    if getattr(p, 'enable_hr', False) or getattr(p, 'disable_extra_networks', False):
        raise ValueError('MOSS DMD requires a single pass with LoRAs enabled.')
    import networks
    networks.list_available_networks()
    name = verified_adapter(networks)
    if any('<lora:' in prompt.lower() for prompt in p.all_prompts):
        raise ValueError('MOSS DMD preset cannot combine prompt LoRAs; remove those tags or disable the preset.')
    tag = f'<lora:{name}:{float(strength):g}>'
    p.all_prompts = [f'{prompt} {tag}' for prompt in p.all_prompts]
    p.steps, p.cfg_scale, p.sampler_name, p.scheduler = 8, 1.0, 'Euler', 'MOSS'
    p.extra_generation_params.update({'MOSS DMD': f'latest {REVISION[:12]}',
                                    'MOSS DMD strength': float(strength), 'MOSS DMD SHA256': SHA256})
