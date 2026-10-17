"""Numerical adapter checks against the installed Forge backend.

Run from Forge's root so its bundled packages are available:
  venv/Scripts/python -m unittest discover -s <extension>/tests -p test_native_adapters.py -v
"""
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from forge_moss.native import lora as moss_lora


class TinyFused(torch.nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.diffusion_model = torch.nn.Module()
        self.diffusion_model.layers = torch.nn.ModuleList([torch.nn.Module()])
        self.diffusion_model.layers[0].proj = torch.nn.Module()
        layer = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False)
        layer.weight = torch.nn.Parameter(weight.clone(), requires_grad=False)
        self.diffusion_model.layers[0].proj.qkv = layer
        self.diffusion_model.layers[0].proj.o = torch.nn.Linear(4, 4, bias=False)


class TestInstalledForgeAdapters(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        forge = Path.cwd()
        if not (forge / "backend" / "patcher" / "lora.py").is_file():
            raise unittest.SkipTest("run this test from the installed Forge root")
        from modules_forge.initialization import initialize_forge

        initialize_forge()
        from backend import operations
        from backend.patcher.base import ModelPatcher
        from modules_forge.packages.comfy.lora import load_lora

        cls.ModelPatcher = ModelPatcher
        cls.QuantizedTensor = operations.QuantizedTensor
        cls.mixed_precision_ops = operations.mixed_precision_ops
        cls.load_lora = staticmethod(load_lora)

    def merged(self, model, adapter, device=torch.device("cpu")):
        key_map = moss_lora.moss_lora_keys(model.state_dict())
        patches, remaining = self.load_lora(adapter, key_map)
        self.assertEqual(remaining, {})
        patcher = self.ModelPatcher(model, torch.device("cpu"), torch.device("cpu"))
        self.assertEqual(len(patcher.add_patches(patches)), 1)
        key = next(iter(patcher.patches))
        return key, patcher.patch_weight_to_device(key, device_to=device, return_weight=True)

    def test_lora_updates_only_q_third_of_fused_qkv(self):
        base = torch.arange(48, dtype=torch.float32).reshape(12, 4) / 100
        model = TinyFused(base)
        down = torch.tensor([[1., 2., 0., -1.], [0.5, -1., 1., 0.]])
        up = torch.tensor([[1., 0.], [0., 1.], [1., 1.], [-1., 0.5]])
        prefix = "transformer.layers.0.proj.q"
        key, got = self.merged(model, {f"{prefix}.lora_A.weight": down,
                                       f"{prefix}.lora_B.weight": up})
        expected = base.clone()
        expected[:4] += up @ down
        self.assertEqual(key, "diffusion_model.layers.0.proj.qkv.weight")
        torch.testing.assert_close(got, expected)

    def test_lokr_updates_only_k_third_of_fused_qkv(self):
        base = torch.zeros(12, 4)
        model = TinyFused(base)
        w1 = torch.tensor([[1., 2.], [3., 4.]])
        w2 = torch.tensor([[0.5, -1.], [2., 0.25]])
        prefix = "transformer.layers.0.proj.k"
        key, got = self.merged(model, {f"{prefix}.lokr_w1": w1, f"{prefix}.lokr_w2": w2})
        expected = base.clone()
        expected[4:8] = torch.kron(w1, w2)
        self.assertEqual(key, "diffusion_model.layers.0.proj.qkv.weight")
        torch.testing.assert_close(got, expected)

    def test_lora_updates_unfused_projection(self):
        model = TinyFused(torch.zeros(12, 4))
        base = model.diffusion_model.layers[0].proj.o.weight.detach().clone()
        down = torch.tensor([[1., 0., -1., 2.]])
        up = torch.tensor([[1.], [2.], [-1.], [0.5]])
        prefix = "transformer.layers.0.proj.o"
        key, got = self.merged(model, {f"{prefix}.lora_A.weight": down,
                                       f"{prefix}.lora_B.weight": up})
        self.assertEqual(key, "diffusion_model.layers.0.proj.o.weight")
        torch.testing.assert_close(got, base + up @ down)

    def test_lora_requantizes_real_int8_convrot_weight(self):
        if not torch.cuda.is_available():
            self.skipTest("ConvRot backend requires CUDA")
        device = torch.device("cuda")
        generator = torch.Generator(device=device).manual_seed(8)
        base = torch.randn(256, 256, device=device, dtype=torch.bfloat16, generator=generator) * 0.1
        quantized = self.QuantizedTensor.from_float(
            base, "TensorWiseINT8Layout", scale="recalculate", per_channel=True,
            convrot=True, convrot_groupsize=256)
        ops = self.mixed_precision_ops(compute_dtype=torch.bfloat16)
        linear = ops.Linear(256, 256, bias=False, device=device)
        linear.quant_format = "int8_tensorwise"
        linear.layout_type = "TensorWiseINT8Layout"
        linear._per_row = True
        linear.weight = torch.nn.Parameter(quantized, requires_grad=False)
        model = torch.nn.Module()
        model.diffusion_model = torch.nn.Module()
        model.diffusion_model.proj = linear

        rank = 2
        down = torch.randn(rank, 256, device=device, dtype=torch.float32, generator=generator) * 0.01
        up = torch.randn(256, rank, device=device, dtype=torch.float32, generator=generator) * 0.01
        original = quantized.dequantize().float()
        key, patched = self.merged(model, {"transformer.proj.lora_A.weight": down,
                                           "transformer.proj.lora_B.weight": up}, device)
        self.assertEqual(key, "diffusion_model.proj.weight")
        self.assertIsInstance(patched, self.QuantizedTensor)
        got = patched.dequantize().float()
        expected = original + up @ down
        self.assertGreater(float((got - original).abs().max()), 0)
        torch.testing.assert_close(got, expected, rtol=0.05, atol=0.005)


if __name__ == "__main__":
    unittest.main()
