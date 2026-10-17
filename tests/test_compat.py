import functools
import sys
import types
import unittest
from unittest import mock

from forge_moss.native import compat


def loader_original(*args, **kwargs):
    branches = ("Krea2Transformer2DModel", "AutoencoderKLFlux2", "Qwen3Model")
    # from backend.nn.llm.llama import Qwen3_8B
    # from backend.nn.krea import SingleStreamDiT
    return branches


loader_original.__module__ = "backend.loader"


def unrelated_decoy(*args, **kwargs):
    """DECOY Krea2Transformer2DModel AutoencoderKLFlux2 Qwen3Model"""


unrelated_decoy.__module__ = "third_party.wrapper"


class TestLoaderSource(unittest.TestCase):
    def test_follows_wrapped_loader(self):
        @functools.wraps(loader_original)
        def wrapper(*args, **kwargs):
            return loader_original(*args, **kwargs)

        wrapper.__module__ = "h3.loader_wrapper"
        source = compat._loader_source(wrapper)
        self.assertIn("Qwen3_8B", source)
        self.assertIn("SingleStreamDiT", source)

    def test_follows_closure_original_but_ignores_decoy(self):
        original = loader_original
        decoy = unrelated_decoy

        def wrapper(*args, **kwargs):
            decoy
            return original(*args, **kwargs)

        wrapper.__module__ = "h3.loader_wrapper"
        source = compat._loader_source(wrapper)
        self.assertIn("Qwen3_8B", source)
        self.assertNotIn("DECOY", source)

    def test_follows_default_original(self):
        def wrapper(*args, _original=loader_original, **kwargs):
            return _original(*args, **kwargs)

        wrapper.__module__ = "h3.loader_wrapper"
        self.assertIn("Qwen3Model", compat._loader_source(wrapper))

    def test_follows_partial_function(self):
        self.assertIn("AutoencoderKLFlux2", compat._loader_source(functools.partial(loader_original)))

    def test_cycles_and_opaque_callables_are_safe(self):
        def first():
            pass

        def second():
            pass

        first.__wrapped__ = second
        second.__wrapped__ = first
        self.assertEqual(compat._loader_source(first), "")
        self.assertEqual(compat._loader_source(len), "")

    def test_check_reports_missing_loader_branches_without_crashing(self):
        def incomplete_loader():
            return "Krea2Transformer2DModel"

        incomplete_loader.__module__ = "backend.loader"
        backend = types.ModuleType("backend")
        backend.loader = types.SimpleNamespace(load_huggingface_component=incomplete_loader)

        class RMSNorm:
            def __init__(self, size, add=False):
                pass

        backend.operations = types.SimpleNamespace(
            ForgeOperations=types.SimpleNamespace(RMSNorm=RMSNorm))
        with mock.patch.object(compat, "REQUIRED", {}), mock.patch.dict(
                sys.modules, {"backend": backend}):
            problems = compat.check()
        self.assertTrue(any("AutoencoderKLFlux2" in p and "Qwen3Model" in p for p in problems))
        self.assertTrue(any("Qwen3_8B" in p and "SingleStreamDiT" in p for p in problems))


if __name__ == "__main__":
    unittest.main()
