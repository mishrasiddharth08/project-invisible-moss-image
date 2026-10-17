"""Read small safetensors headers to discover compatible local MOSS components."""
import json
from pathlib import Path
import struct

from . import checkpoints, detect


def header(path):
    with Path(path).open('rb') as handle:
        length = struct.unpack('<Q', handle.read(8))[0]
        if not 2 <= length <= 16 * 1024 * 1024:
            raise ValueError('Invalid safetensors header length')
        return json.loads(handle.read(length))


def valid_checkpoint(path):
    try:
        return detect.detect(header(path)) is not None
    except (OSError, ValueError, KeyError, TypeError, struct.error):
        return False


def valid_encoder(path):
    try:
        checkpoints.text_encoder_prefix(header(path))
        return True
    except (OSError, ValueError, KeyError, TypeError, struct.error):
        return False


def valid_vae(path):
    try:
        state = header(path)
        for prefix, replacement in (('encoder.quant_conv.', 'quant_conv.'),
                                    ('decoder.post_quant_conv.', 'post_quant_conv.')):
            for key in list(state):
                if key.startswith(prefix):
                    state[replacement + key[len(prefix):]] = state.pop(key)
        checkpoints.check_vae(state)
        return True
    except (OSError, ValueError, KeyError, TypeError, struct.error):
        return False


def discover(models_root):
    """Names narrow discovery; architecture, never a filename alone, decides validity."""
    root = Path(models_root)
    choices = (
        (root / 'Stable-diffusion', lambda name: 'moss' in name, valid_checkpoint),
        (root / 'text_encoder', lambda name: 'qwen' in name and ('3.5' in name or '35' in name), valid_encoder),
        (root / 'VAE', lambda name: 'flux' in name and ('2' in name), valid_vae),
    )
    found = []
    for directory, hint, validator in choices:
        preferred = {'moss_image_sft_int8_convrot.safetensors', 'qwen3.5_9b_int8_convrot.safetensors', 'flux2-vae.safetensors'}
        paths = sorted(directory.rglob('*.safetensors'), key=lambda p: (p.name not in preferred, str(p)))
        match = next((path for path in paths if hint(path.name.lower()) and validator(path)), None)
        if match is None:
            return None
        found.append(match)
    return tuple(found)
