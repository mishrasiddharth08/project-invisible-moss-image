import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch


class FakeUnet:
    def __init__(self):
        self.options = {'existing_extension': 'preserved'}
        self.model_options = {'transformer_options': self.options}

    def clone(self):
        clone = FakeUnet()
        clone.options = dict(self.options)
        clone.model_options = {'transformer_options': clone.options}
        return clone

    def set_transformer_option(self, key, value):
        self.options[key] = value


class Request:
    def __init__(self):
        self.sd_model = SimpleNamespace(is_moss_image=True,
            forge_objects=SimpleNamespace(unet=FakeUnet()))
        self.sampler_name = 'Euler'
        self.scheduler = 'MOSS'
        self.enable_hr = False
        self.sampler = SimpleNamespace(model_wrap_cfg=SimpleNamespace(step=0, total_steps=50))
        self.extra_generation_params = {}
        self.closed = 0

    def close(self):
        self.closed += 1
        self.sampler = None


def script(mode='Balanced'):
    modules = ModuleType('modules')
    modules.scripts = SimpleNamespace(Script=object, AlwaysVisible=True)
    modules.shared = SimpleNamespace(opts=SimpleNamespace(moss_spectrum=mode))
    path = Path(__file__).resolve().parents[1]/'scripts/moss_spectrum.py'
    spec = importlib.util.spec_from_file_location('moss_spectrum_test', path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {'modules': modules}):
        spec.loader.exec_module(module)
    return module.Script()


class TestSpectrumLifecycle(unittest.TestCase):
    def test_close_on_error_clears_history_without_changing_original_unet(self):
        hook, request = script(), Request()
        original = request.sd_model.forge_objects.unet
        hook.process_before_every_sampling(request)
        self.assertEqual(original.options, {'existing_extension': 'preserved'})
        session = request._moss_spectrum_session
        patched = request.sd_model.forge_objects.unet
        session.states['test'] = dict(context=torch.ones(1), values=[torch.ones(1)])
        request.close()
        self.assertEqual(session.nbytes, 0)
        self.assertEqual(session.progress(), (0, 0))
        self.assertNotIn('moss_spectrum_session', patched.options)
        self.assertEqual(request.closed, 1)
        self.assertIs(request.close.__func__, Request.close)

    def test_existing_close_override_is_preserved(self):
        hook, request = script(), Request()
        calls = []
        previous = lambda: calls.append('previous_extension')
        request.close = previous
        hook.process_before_every_sampling(request)
        request.close()
        self.assertIs(request.close, previous)
        self.assertEqual(calls, ['previous_extension'])

    def test_resident_model_does_not_retain_closed_request(self):
        import gc
        import weakref
        hook, request = script(), Request()
        request.sampler.model_wrap_cfg.p = request
        engine = request.sd_model
        reference = weakref.ref(request)
        hook.process_before_every_sampling(request)
        session = request._moss_spectrum_session
        request.close()
        del request
        gc.collect()
        self.assertIsNone(reference())
        self.assertNotIn('moss_spectrum_session', engine.forge_objects.unet.options)
        self.assertEqual(session.progress(), (0, 0))

    def test_other_models_and_unsupported_modes_are_unchanged(self):
        request = Request()
        request.sd_model.is_moss_image = False
        script().process_before_every_sampling(request)
        self.assertEqual(request.extra_generation_params, {})
        self.assertFalse(hasattr(request, '_moss_spectrum_session'))
        for sampler, hires, mode in [('Euler', False, 'Off'), ('Heun', False, 'Balanced'), ('Euler', True, 'Balanced')]:
            request = Request()
            request.sampler_name, request.enable_hr = sampler, hires
            script(mode).process_before_every_sampling(request)
            self.assertFalse(hasattr(request, '_moss_spectrum_session'))
            self.assertTrue(request.extra_generation_params['MOSS Spectrum'].startswith('Off'))
