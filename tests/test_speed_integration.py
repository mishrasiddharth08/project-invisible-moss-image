import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from forge_moss.native import dmd, lora
from forge_moss.native.spectrum_hidden import HiddenSession
from forge_moss.native.spectrum_policy import SpectrumConfig
from tests.test_transformer import inputs, pair
from tests.test_spectrum_lifecycle import Request, script


class SpeedIntegration(unittest.TestCase):
    def test_forecast_head_matches_real_head(self):
        _, core = pair()
        x, sigma, context = inputs()
        features = []
        hook = core.layers[-1].register_forward_hook(lambda m, a, out: features.append(out[:, :15].clone()))
        with torch.inference_mode():
            expected = core(x, sigma, context)
            actual = core.forecast_head(x, sigma, context.dtype, features[-1])
        hook.remove()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_manual_and_adaptive_real_features_cfg_and_cleanup(self):
        _, core = pair()
        x, _, context = inputs(batch=1)
        for mode in ['manual', 'adaptive']:
            progress = [0, 30]
            session = HiddenSession(lambda: tuple(progress), SpectrumConfig(mode=mode, warmup_steps=5))
            with torch.inference_mode():
                for step in range(30):
                    progress[0] = step
                    before = session.real
                    for branch in [0, 1]:
                        result = session.run(core._forward_full, x, torch.tensor([1-step/30]), context+branch,
                                             transformer_options={'cond_or_uncond':[branch]})
                        self.assertTrue(torch.isfinite(result).all())
                    if step < 5 or step >= 28:
                        self.assertEqual(session.real-before, 2)
            self.assertGreater(session.forecast, 0)
            self.assertEqual(session.real+session.forecast, 60)
            self.assertEqual(len(core.layers[-1]._forward_hooks), 0)
            self.assertLessEqual(session.peak_bytes, session.max_bytes)
            self.assertTrue(all(not s['branch'].forecaster._anchors for s in session.states.values()))
            session.close()
            self.assertEqual(session.nbytes, 0)
            self.assertEqual(session.progress(), (0, 0))

    def test_cache_budget_controls_and_rewind_fail_closed(self):
        _, core = pair()
        x, sigma, context = inputs(batch=1)
        progress = [0, 30]
        session = HiddenSession(lambda: tuple(progress), SpectrumConfig(), max_bytes=1)
        with torch.inference_mode():
            got = session.run(core._forward_full, x, sigma, context, transformer_options={'cond_or_uncond':[0]})
            torch.testing.assert_close(got, core(x, sigma, context), rtol=0, atol=0)
        self.assertEqual(session.nbytes, 0)
        session.max_bytes = 1024**2
        with torch.inference_mode():
            for step in [0, 1, 2, 0]:
                progress[0] = step
                session.run(core._forward_full, x, sigma, context, transformer_options={'cond_or_uncond':[0]})
        state = next(iter(session.states.values()))
        self.assertEqual(len(state['branch'].forecaster._anchors), 1)
        with torch.inference_mode():
            session.run(core._forward_full, x, sigma, context, transformer_options={'cond_or_uncond':[0], 'patches':{'unknown':[]}})
        self.assertEqual(session.nbytes, 0)

    def test_exception_removes_feature_hook(self):
        _, core = pair()
        x, sigma, context = inputs(batch=1)
        session = HiddenSession(lambda: (0, 30), SpectrumConfig())
        with patch.object(core.layers[0], 'forward', side_effect=RuntimeError('synthetic failure')):
            with self.assertRaisesRegex(RuntimeError, 'synthetic failure'):
                session.run(core._forward_full, x, sigma, context, transformer_options={'cond_or_uncond':[0]})
        self.assertFalse(core.layers[-1]._forward_hooks)
        session.close()
        self.assertEqual(session.nbytes, 0)

    def test_actual_dmd_header_all_252_targets_match_fused_map(self):
        header = json.loads((Path(__file__).parent/'fixtures/dmd-latest-header.json').read_text())
        targets = [key.removesuffix('.lora_down.weight') for key in header if key.endswith('.lora_down.weight')]
        self.assertEqual(len(targets), 252)
        for filename in ['moss_image_sft_bf16.shapes.json', 'moss_image_sft_int8_convrot_attn.shapes.json']:
            actual = json.loads((Path(__file__).parent/'fixtures'/filename).read_text())
            state = {'diffusion_model.'+key: SimpleNamespace(shape=value['shape'])
                     for key, value in actual.items() if isinstance(value, dict) and 'shape' in value}
            keys = lora.moss_lora_keys(state)
            self.assertTrue(all(t in keys for t in targets), filename)
            for target in targets:
                mapped = keys[target]
                shape = state[mapped[0] if isinstance(mapped, tuple) else mapped].shape
                out_features = mapped[1][2] if isinstance(mapped, tuple) else shape[0]
                down, up = header[target+'.lora_down.weight']['shape'], header[target+'.lora_up.weight']['shape']
                self.assertEqual(down[1], shape[1], target)
                self.assertEqual(up, [out_features, down[0]], target)

    def test_dmd_preset_gates_and_request_only_settings(self):
        import sys
        from types import ModuleType
        networks = ModuleType('networks')
        networks.list_available_networks = lambda: None
        p = Request()
        p.sd_model.sd_checkpoint_info = SimpleNamespace(filename='MOSS/moss-SFT-int8_convrot.safetensors')
        p.all_prompts = ['a ceramic teapot', 'a brass clock']
        with patch.dict(sys.modules, {'networks': networks}), patch.object(dmd, 'verified_adapter', return_value='DMD'):
            dmd.apply(p, 0.7)
        self.assertEqual((p.steps, p.cfg_scale, p.sampler_name, p.scheduler), (8, 1., 'Euler', 'MOSS'))
        self.assertTrue(all(s.endswith('<lora:DMD:0.7>') for s in p.all_prompts))
        with self.assertRaises(ValueError):
            dmd.apply(p, float('nan'))
        p.sd_model.is_moss_image = False
        with self.assertRaisesRegex(ValueError, 'requires a MOSS'):
            dmd.apply(p, 0.7)

    def test_inline_modes_and_dmd_disable_forecasting(self):
        hook, p = script(), Request()
        hook.process(p, 'Adaptive')
        hook.process_before_every_sampling(p)
        self.assertIsInstance(p._moss_spectrum_session, HiddenSession)
        p.close()
        hook, p = script(), Request()
        with patch('forge_moss.native.dmd.apply') as apply:
            hook.process(p, 'Adaptive', True)
        apply.assert_called_once_with(p, 0.7)
        hook.process_before_every_sampling(p)
        self.assertFalse(hasattr(p, '_moss_spectrum_session'))

    def test_inline_ui_has_no_accordion(self):
        try:
            import gradio as gr
        except ImportError:
            self.skipTest('Gradio is optional in CPU-only CI; installed Forge validates this locally')
        with gr.Blocks() as panel:
            controls = script().ui(False)
        self.assertEqual(len(controls), 11)
        config = panel.get_config_file()
        self.assertFalse(any(c['type'] == 'accordion' for c in config['components']))
        self.assertEqual(controls[0].choices, [('Off', 'Off'), ('Balanced', 'Balanced'),
                                              ('Manual', 'Manual'), ('Adaptive', 'Adaptive')])

    def test_nonfinite_controls_and_missing_dmd_fail_clearly(self):
        for kwargs in [{'tolerance': float('nan')}, {'cache': 'unknown'}, {'tail': 0}]:
            request = Request()
            script().process(request, 'Adaptive', **kwargs)
            with self.assertRaises(ValueError):
                request.init()
        with self.assertRaisesRegex(ValueError, 'missing or hash mismatch'):
            dmd.verified_adapter(SimpleNamespace(available_networks={}))

    def test_rejection_survives_forge_swallowing_script_callback_errors(self):
        request = Request()
        with patch('forge_moss.native.dmd.apply', side_effect=ValueError('unsupported adapter pairing')):
            # Forge may swallow callback exceptions, but the core request init
            # must still fail before another model path can generate an image.
            script().process(request, 'Adaptive', True)
        self.assertIn('unsupported adapter pairing', request.extra_generation_params['MOSS validation error'])
        with self.assertRaisesRegex(ValueError, 'unsupported adapter pairing'):
            request.init([], [], [])


if __name__ == '__main__':
    unittest.main()
