# MOSS-Image support for Forge Neo
# A "moss" UI preset, added at runtime: PresetArch gains a member and the preset's options are registered through
# Forge Neo's own presets.register, so they match every other preset. Same mechanism as the FLUX 3 extension.

from enum import Enum

from modules_forge import presets

PRESET = "moss"

# upstream's sampling settings, for the SFT and Base checkpoints alike: flow-matching Euler, 50 steps, CFG 3
SAMPLER = "Euler"
SCHEDULER = "MOSS"
FALLBACK_SCHEDULER = "Simple"
STEPS = 50
CFG = 3.0


def _add_enum_member(enum_cls: type[Enum], name: str) -> Enum:
    if name in enum_cls.__members__:
        return enum_cls[name]

    value = max(m.value for m in enum_cls) + 1
    member = object.__new__(enum_cls)
    member._name_ = name
    member._value_ = value
    member.__objclass__ = enum_cls
    member._sort_order_ = len(enum_cls._member_names_)

    # EnumType.__setattr__ refuses new members, so go through type
    type.__setattr__(enum_cls, name, member)
    enum_cls._member_map_[name] = member
    enum_cls._member_names_.append(name)
    enum_cls._value2member_map_[value] = member
    if hasattr(enum_cls, "_hashable_values_"):
        enum_cls._hashable_values_.append(value)
    return member


def _is_preset_option(key: str) -> bool:
    return key.startswith(f"{PRESET}_") or key.endswith(f"_{PRESET}")


def register() -> None:
    arch = _add_enum_member(presets.PresetArch, PRESET)

    from modules import sd_schedulers

    presets.SAMPLERS[arch] = SAMPLER
    # without the MOSS scheduler (it failed to register), Simple: the same shifted curve, not snapped
    presets.SCHEDULERS[arch] = SCHEDULER if SCHEDULER in sd_schedulers.schedulers_map else FALLBACK_SCHEDULER
    presets.STEPS[arch] = STEPS
    presets.CFG[arch] = CFG

    from modules import shared

    templates: dict = {}
    presets.register(templates)
    for key, info in templates.items():
        if _is_preset_option(key) and key not in shared.opts.data_labels:
            shared.opts.add_option(key, info)

    # Give a new model preset its own validated selection before shared preset
    # integrations can inherit the previously selected model's components.
    from modules import paths, sd_models
    from . import assets

    found = assets.discover(paths.models_path)
    if found:
        checkpoint, encoder, vae = found
        checkpoint_key = f'forge_checkpoint_{PRESET}'
        modules_key = f'forge_additional_modules_{PRESET}'
        saved_checkpoint = shared.opts.data.get(checkpoint_key)
        matching = next((info for info in sd_models.checkpoints_list.values()
                         if saved_checkpoint in (info.name, info.title, info.filename)
                         and assets.valid_checkpoint(info.filename)), None)
        if matching is None:
            shared.opts.data[checkpoint_key] = checkpoint.name
        saved_modules = shared.opts.data.get(modules_key) or []
        if not (len(saved_modules) == 2 and any(assets.valid_encoder(p) for p in saved_modules)
                and any(assets.valid_vae(p) for p in saved_modules)):
            shared.opts.data[modules_key] = [str(encoder), str(vae)]
