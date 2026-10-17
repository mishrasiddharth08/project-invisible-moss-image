"""Optional in-memory registration with Project Invisible Model Autolink."""
import importlib
from pathlib import Path

from . import assets

FAMILY = "moss_image"
ENCODER = "qwen35_9b_moss"
PRESET = "moss"


def _add_exact(registry, kind, label, fp):
    entries = registry.prototypes[kind]
    if not any(old_label == label and old_fp.identity == fp.identity for old_label, old_fp in entries):
        entries.append((label, fp))


def register() -> bool:
    """Teach an already-running Model Autolink about validated local MOSS files."""
    try:
        compatibility = importlib.import_module("model_autolink.compatibility")
        fingerprinting = importlib.import_module("model_autolink.fingerprint")
        integration = importlib.import_module("model_autolink.integration")
        registry_module = importlib.import_module("model_autolink.registry")
        from modules import paths_internal
    except (ImportError, AttributeError):
        return False

    resolver = getattr(integration, "_resolver", None)
    registry = getattr(resolver, "registry", None)
    if registry is None:
        return False

    found = assets.discover(paths_internal.models_path)
    if found is None:
        return False
    checkpoint, encoder, _vae = (Path(path).resolve() for path in found)

    checkpoint_fp = fingerprinting.fingerprint(checkpoint)
    encoder_fp = fingerprinting.fingerprint(encoder)

    compatibility.MATRIX.setdefault(FAMILY, {
        "slots": [[ENCODER], ["flux2_vae"]],
        "description": "MOSS-Image Base / SFT",
    })
    integration.FAMILY_PRESETS.setdefault(FAMILY, PRESET)
    _add_exact(registry, "checkpoints", FAMILY, checkpoint_fp)
    _add_exact(registry, "components", ENCODER, encoder_fp)

    wanted = str(encoder).casefold()
    item = next((entry for entry in registry.index if str(Path(entry.path).resolve()).casefold() == wanted), None)
    if item is None:
        item = registry_module.IndexedFile(
            str(encoder), encoder_fp, (ENCODER,), 1.0,
            fingerprinting.quantization(encoder, encoder_fp),
        )
        registry.index.append(item)
    else:
        # Replace only this exact, structurally validated file's stale Qwen-Image 2.1 classification.
        item.fp = encoder_fp
        item.components = (ENCODER,)
        item.confidence = 1.0
        item.quant = fingerprinting.quantization(encoder, encoder_fp)

    print("[MOSS-Image] Model Autolink runtime support registered")
    return True
