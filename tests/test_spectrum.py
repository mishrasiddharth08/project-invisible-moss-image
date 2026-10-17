import unittest

import torch

from forge_moss.native.spectrum import Config, Session, weights
from tests.test_transformer import inputs, pair


class TestSpectrum(unittest.TestCase):
    def test_disabled_transformer_is_reference_identical(self):
        reference, model = pair()
        x, sigma, context = inputs()
        with torch.inference_mode():
            torch.testing.assert_close(model(x, sigma, context), reference(x, sigma*1000, context),
                                       rtol=1e-4, atol=1e-5)

    def test_fit_is_finite_and_refuses_duplicate_coordinates(self):
        self.assertIsNone(weights([0.1, 0.1], 0.2, 0.01))
        self.assertTrue(torch.isfinite(weights([1., 0.9, 0.8, 0.7], 0.6, 0.01)).all())

    def test_cfg_branches_warmup_tail_and_real_only_history(self):
        progress = [0, 50]
        session = Session(lambda: tuple(progress))
        x = torch.ones(1, 2, 2, 2)
        calls = []
        def real(x, timestep, context, **kwargs):
            calls.append((progress[0], float(context.mean())))
            return torch.ones_like(x)*(context.mean()+timestep[0]*0.1)
        for step in range(50):
            progress[0] = step
            for branch, value in ((0, 1.), (1, 100.)):
                t = torch.tensor([1-step/50])
                context = torch.full((1, 3, 4), value)
                got = session.run(real, x, t, context, transformer_options={'cond_or_uncond': [branch]})
                self.assertLess(float((got-(value+t[0]*0.1)).abs().max()), 1.)
        self.assertGreater(session.forecast, 0)
        self.assertEqual(session.real+session.forecast, 100)
        self.assertTrue(all((step, branch) in calls for step in list(range(8))+list(range(44, 50))
                            for branch in (1., 100.)))
        self.assertLessEqual(session.nbytes, session.config.max_bytes)
        for state in session.states.values():
            self.assertLessEqual(len(state['values']), 4)
        session.clear()
        self.assertEqual(session.nbytes, 0)

    def test_condition_change_restarts_warmup(self):
        progress = [0, 50]
        session = Session(lambda: tuple(progress))
        x = torch.ones(1, 2, 2, 2)
        def real(x, timestep, context, **kwargs):
            return torch.ones_like(x)*context.mean()
        for step in range(12):
            progress[0] = step
            value = 100. if step >= 8 else 1.
            got = session.run(real, x, torch.tensor([1-step/50]), torch.full((1, 2, 2), value),
                              transformer_options={'cond_or_uncond': [0]})
            torch.testing.assert_close(got, torch.full_like(x, value))
        self.assertEqual(session.forecast, 0)

    def test_memory_cap_and_unknown_patches_fall_back(self):
        progress = [0, 50]
        session = Session(lambda: tuple(progress), Config(max_bytes=1))
        x, context = torch.ones(1, 2, 2, 2), torch.ones(1, 2, 2)
        def real(x, *args, **kwargs):
            return x*3
        for step in range(50):
            progress[0] = step
            torch.testing.assert_close(session.run(real, x, torch.tensor([.5]), context,
                transformer_options={'cond_or_uncond': [0]}), x*3)
        self.assertEqual(session.forecast, 0)
        self.assertEqual(session.nbytes, 0)
        roomy = Session(lambda: (10, 50))
        roomy.run(real, x, torch.tensor([.5]), context,
                  transformer_options={'cond_or_uncond': [0], 'patches': {'custom': []}})
        self.assertEqual(roomy.nbytes, 0)

    def test_explosive_forecast_falls_back_to_real(self):
        from unittest.mock import patch
        progress = [0, 50]
        session = Session(lambda: tuple(progress))
        x, context = torch.ones(1, 2, 2, 2), torch.ones(1, 2, 2)
        def real(x, *args, **kwargs):
            return x
        for step in range(9):
            progress[0] = step
            with patch('forge_moss.native.spectrum.weights', return_value=torch.tensor([float('nan')]*4)):
                got = session.run(real, x, torch.tensor([1-step/50]), context,
                                  transformer_options={'cond_or_uncond': [0]})
            torch.testing.assert_close(got, x)
        self.assertEqual(session.forecast, 0)
        self.assertGreater(session.rejected, 0)
