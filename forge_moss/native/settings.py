# MOSS-Image support for Forge Neo
# The negative-prompt block skipping of upstream's guidance, as a Forge setting: the blocks (zero-based) the DiT
# leaves out in the CFG negative pass. Upstream's default is block 22; an empty value is plain CFG.

OPTION = "moss_uncond_skip_layers"
DEFAULT = "22"
LABEL = "[MOSS-Image] Blocks skipped in the negative-prompt pass"
INFO = "zero-based, comma-separated; 22 = upstream's guidance; empty = plain CFG"
# Forge Neo's "Stable Diffusion" settings section and category, where its per-model options ([Klein], [Krea2]...) live
# (options_section splits its ("sd", "Stable Diffusion", "sd") the same way)
SECTION = ("sd", "Stable Diffusion")
CATEGORY = "sd"


def parse(text: str, depth: int) -> frozenset[int]:
    """The valid block indices in text; anything else is ignored."""
    layers = set()
    for part in str(text or "").replace(";", ",").replace(" ", ",").split(","):
        if part.strip().lstrip("-").isdigit() and 0 <= int(part) < depth:
            layers.add(int(part))
    return frozenset(layers)


def uncond_skip_layers(depth: int) -> frozenset[int]:
    try:
        from modules.shared import opts

        text = getattr(opts, OPTION, DEFAULT)
    except Exception:  # outside Forge Neo (CPU tests), or before the options exist
        text = DEFAULT
    return parse(text, depth)


def register() -> None:
    from modules import shared
    from modules.options import OptionInfo

    if OPTION not in shared.opts.data_labels:
        shared.opts.add_option(OPTION, OptionInfo(DEFAULT, LABEL, section=SECTION, category_id=CATEGORY,
                                                  infotext="MOSS skip").info(INFO))
    if 'moss_spectrum' not in shared.opts.data_labels:
        import gradio as gr
        shared.opts.add_option('moss_spectrum', OptionInfo('Off', '[MOSS] Spectrum acceleration', gr.Dropdown,
            {'choices': ['Off', 'Balanced']}, section=SECTION, category_id=CATEGORY,
            infotext='MOSS Spectrum mode').info(
                'Balanced forecasts some steps and changes the result. Euler + MOSS only; Off preserves full inference.'))
