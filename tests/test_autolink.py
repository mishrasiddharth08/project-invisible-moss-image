import sys
import types
import unittest
from pathlib import Path
from unittest import mock

from forge_moss.native import autolink


class Fingerprint:
    def __init__(self, identity):
        self.identity = identity


class IndexedFile:
    def __init__(self, path, fp, components, confidence, quant):
        self.path = str(path)
        self.fp = fp
        self.components = components
        self.confidence = confidence
        self.quant = quant


class TestAutolink(unittest.TestCase):
    def modules(self, resolver=True):
        checkpoint = Path("C:/models/moss.safetensors").resolve()
        encoder = Path("C:/models/qwen35.safetensors").resolve()
        vae = Path("C:/models/flux2-vae.safetensors").resolve()
        old_fp = Fingerprint("old")
        stale = IndexedFile(encoder, old_fp, ("qwen3vl_8b_qwen21",), 0.97, "int8_convrot")
        other = IndexedFile("C:/models/other.safetensors", old_fp, ("other",), 1.0, "bf16")
        registry = types.SimpleNamespace(
            prototypes={"checkpoints": [("other", old_fp)], "components": [("other", old_fp)]},
            index=[stale, other],
        )
        compatibility = types.ModuleType("model_autolink.compatibility")
        compatibility.MATRIX = {"other": {"slots": []}}
        integration = types.ModuleType("model_autolink.integration")
        integration.FAMILY_PRESETS = {"other": "other"}
        integration._resolver = types.SimpleNamespace(registry=registry) if resolver else None
        fingerprinting = types.ModuleType("model_autolink.fingerprint")
        fingerprinting.fingerprint = lambda path: Fingerprint("checkpoint" if Path(path) == checkpoint else "encoder")
        fingerprinting.quantization = lambda path, fp: "int8_convrot"
        registry_module = types.ModuleType("model_autolink.registry")
        registry_module.IndexedFile = IndexedFile
        paths_internal = types.SimpleNamespace(models_path="C:/models")
        modules = types.ModuleType("modules")
        modules.paths_internal = paths_internal
        package = types.ModuleType("model_autolink")
        package.__path__ = []
        injected = {
            "modules": modules,
            "model_autolink": package,
            "model_autolink.compatibility": compatibility,
            "model_autolink.fingerprint": fingerprinting,
            "model_autolink.integration": integration,
            "model_autolink.registry": registry_module,
        }
        return injected, registry, compatibility, integration, stale, other, (checkpoint, encoder, vae)

    def test_registers_exact_runtime_policy_and_repairs_only_moss_encoder(self):
        injected, registry, compatibility, integration, stale, other, found = self.modules()
        with mock.patch.dict(sys.modules, injected), mock.patch.object(autolink.assets, "discover", return_value=found):
            self.assertTrue(autolink.register())
            self.assertTrue(autolink.register())

        self.assertEqual(compatibility.MATRIX[autolink.FAMILY]["slots"],
                         [[autolink.ENCODER], ["flux2_vae"]])
        self.assertEqual(integration.FAMILY_PRESETS[autolink.FAMILY], autolink.PRESET)
        self.assertEqual([(label, fp.identity) for label, fp in registry.prototypes["checkpoints"]].count(
            (autolink.FAMILY, "checkpoint")), 1)
        self.assertEqual([(label, fp.identity) for label, fp in registry.prototypes["components"]].count(
            (autolink.ENCODER, "encoder")), 1)
        self.assertEqual(stale.components, (autolink.ENCODER,))
        self.assertEqual(stale.confidence, 1.0)
        self.assertEqual(other.components, ("other",))

    def test_adds_missing_encoder_to_runtime_index(self):
        injected, registry, _, _, stale, _, found = self.modules()
        registry.index.remove(stale)
        with mock.patch.dict(sys.modules, injected), mock.patch.object(autolink.assets, "discover", return_value=found):
            self.assertTrue(autolink.register())
        matches = [item for item in registry.index if item.components == (autolink.ENCODER,)]
        self.assertEqual(len(matches), 1)

    def test_is_optional_and_does_nothing_without_assets_or_resolver(self):
        for resolver, found in ((False, (Path("x"), Path("y"), Path("z"))), (True, None)):
            injected, registry, compatibility, integration, _, _, default_found = self.modules(resolver=resolver)
            before = dict(compatibility.MATRIX)
            with mock.patch.dict(sys.modules, injected), mock.patch.object(
                    autolink.assets, "discover", return_value=found if resolver else default_found):
                self.assertFalse(autolink.register())
            self.assertEqual(compatibility.MATRIX, before)
            self.assertNotIn(autolink.FAMILY, integration.FAMILY_PRESETS)
            self.assertFalse(any(label == autolink.FAMILY for label, _ in registry.prototypes["checkpoints"]))


if __name__ == "__main__":
    unittest.main()
