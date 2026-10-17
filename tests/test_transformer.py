# MOSS-Image support for Forge Neo
# The DiT port against the official pipeline's MOSSImageTransformer (tests/reference/moss_image_dit.py): plain, with
# the negative-pass block skipping Forge's cond/uncond batches ask for, and with Comfy-Org's fused q/k/v. detect() and
# the built module against the real checkpoints' headers (tests/fixtures).

import json
import unittest
from pathlib import Path
from unittest import mock

import torch

from forge_moss.native import checkpoints, detect, settings
from forge_moss.native.transformer import MossImageDiT
from tests.reference import moss_image_dit

FIXTURES = Path(__file__).parent / "fixtures"
TINY = {"dim": 64, "ffn_dim": 96, "num_heads": 4, "num_layers": 4, "text_dim": 32, "freq_dim": 256,
        "latent_channels": 8, "rope_base": 10000.0}
QUANT_SUFFIXES = (".comfy_quant", ".weight_scale")


def randomized(model, seed):
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, param in model.named_parameters():
            scale = 0.3 if "norm" in name else 0.1
            param.copy_(torch.randn(param.shape, generator=generator) * scale + (1.0 if "norm" in name else 0.0))
    return model


def pair(seed=0, fused=False):
    reference = randomized(moss_image_dit.MOSSImageTransformer(**TINY), seed).eval()
    model = MossImageDiT(**TINY, fused_qkv=fused).eval()
    state = dict(reference.state_dict())
    checkpoints.clean_transformer_state_dict(state)
    if fused:
        for i in range(TINY["num_layers"]):
            parts = [state.pop(f"layers.{i}.proj.{p}.weight") for p in "qkv"]
            state[f"layers.{i}.proj.qkv.weight"] = torch.cat(parts)
    model.load_state_dict(state, strict=True)
    return reference, model


def inputs(batch=2, height=3, width=5, text=7, seed=1):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(batch, TINY["latent_channels"], height, width, generator=generator)
    context = torch.randn(batch, text, TINY["text_dim"], generator=generator)
    sigma = torch.full((batch,), 0.73)
    return x, sigma, context


class TestTransformer(unittest.TestCase):
    def test_matches_upstream(self):
        reference, model = pair()
        x, sigma, context = inputs()
        with torch.no_grad():
            expected = reference(x, sigma * 1000, context)
            got = model(x, sigma, context)
        self.assertEqual(got.shape, x.shape)
        torch.testing.assert_close(got, expected, rtol=1e-4, atol=1e-5)

    def test_fused_qkv(self):
        reference, model = pair(fused=True)
        x, sigma, context = inputs()
        with torch.no_grad():
            torch.testing.assert_close(model(x, sigma, context), reference(x, sigma * 1000, context),
                                       rtol=1e-4, atol=1e-5)

    def test_negative_pass_skips_blocks(self):
        reference, model = pair(seed=2)
        x, sigma, context = inputs(batch=2)
        with torch.no_grad(), mock.patch.object(settings, "uncond_skip_layers", return_value=frozenset({2})):
            # one chunk per pass, as Forge batches cond (0) and uncond (1)
            got = model(x, sigma, context, transformer_options={"cond_or_uncond": [0, 1]})
            only_uncond = model(x[1:], sigma[1:], context[1:], transformer_options={"cond_or_uncond": [1]})
            only_cond = model(x[:1], sigma[:1], context[:1], transformer_options={"cond_or_uncond": [0]})
            plain = reference(x, sigma * 1000, context)
            skipped = reference(x, sigma * 1000, context, skip_layers=[2])
        torch.testing.assert_close(got[0], plain[0], rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(got[1], skipped[1], rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(only_uncond[0], skipped[1], rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(only_cond[0], plain[0], rtol=1e-4, atol=1e-5)
        self.assertFalse(torch.allclose(plain[1], skipped[1]))

    def test_no_skipping_without_cfg_or_setting(self):
        _, model = pair(seed=3)
        x, sigma, context = inputs(batch=1)
        with torch.no_grad():
            base = model(x, sigma, context)
            with mock.patch.object(settings, "uncond_skip_layers", return_value=frozenset()):
                torch.testing.assert_close(model(x, sigma, context, transformer_options={"cond_or_uncond": [1]}),
                                           base)

    def test_upstream_default_is_block_22(self):
        self.assertEqual(settings.uncond_skip_layers(42), frozenset({22}))
        self.assertEqual(settings.parse("22, 24;x 99 -1", 42), frozenset({22, 24}))
        self.assertEqual(settings.parse("", 42), frozenset())


class TestRealCheckpoints(unittest.TestCase):
    def check(self, name, fused):
        header = json.loads((FIXTURES / f"{name}.shapes.json").read_text())
        config = detect.detect(header)
        self.assertEqual(config, {"image_model": "moss_image", "dim": 3072, "num_heads": 24, "num_layers": 42,
                                  "ffn_dim": 8192, "text_dim": 4096, "freq_dim": 256, "latent_channels": 128,
                                  "fused_qkv": fused})
        checkpoints.clean_transformer_state_dict(header)
        with torch.device("meta"):
            model = MossImageDiT(**config)
        expected = {k: tuple(v.shape) for k, v in model.state_dict().items()}
        weights = {k: tuple(v["shape"]) for k, v in header.items() if not k.endswith(QUANT_SUFFIXES)}
        self.assertEqual(weights, expected)

    def test_bf16(self):
        self.check("moss_image_sft_bf16", fused=True)

    def test_int8_convrot(self):
        self.check("moss_image_sft_int8_convrot_attn", fused=True)

    def test_prefixed_and_other_models(self):
        header = json.loads((FIXTURES / "moss_image_sft_bf16.shapes.json").read_text())
        prefixed = {f"model.diffusion_model.{k}": v for k, v in header.items()}
        self.assertEqual(detect.detect(prefixed, "model.diffusion_model.")["num_layers"], 42)
        text_encoder = json.loads((FIXTURES / "qwen3.5_9b_int8_convrot.shapes.json").read_text())
        self.assertIsNone(detect.detect(text_encoder))

    def test_unfused_export(self):
        header = json.loads((FIXTURES / "moss_image_sft_bf16.shapes.json").read_text())
        for i in range(42):
            header.pop(f"layers.{i}.proj.qkv.weight")
            for p in "qkv":
                header[f"layers.{i}.proj.{p}.weight"] = {"dtype": "BF16", "shape": [3072, 3072]}
        self.assertFalse(detect.detect(header)["fused_qkv"])


class TestComponentContracts(unittest.TestCase):
    @staticmethod
    def shaped(shape):
        return {"dtype": "BF16", "shape": list(shape)}

    def test_flux2_vae_requires_its_real_structure(self):
        vae = {key: self.shaped(shape) for key, shape in checkpoints.VAE_SHAPES.items()}
        checkpoints.check_vae(vae)
        for bad in ({f"x.{i}": self.shaped((1,)) for i in range(32)},
                    {**vae, "decoder.conv_in.weight": self.shaped((512, 16, 3, 3))}):
            with self.assertRaisesRegex(ValueError, "FLUX.2 autoencoder"):
                checkpoints.check_vae(bad)

    def test_unvalidated_multimodal_qwen35_is_not_reused(self):
        released = json.loads((FIXTURES / "qwen3.5_9b_int8_convrot.shapes.json").read_text())
        self.assertEqual(checkpoints.text_encoder_prefix(released), "model.language_model.")
        prompt_enhancer = {k: v for k, v in released.items() if not k.startswith("mtp.")}
        with self.assertRaisesRegex(ValueError, "cannot validate.*MTP head is missing"):
            checkpoints.text_encoder_prefix(prompt_enhancer)


if __name__ == "__main__":
    unittest.main()
