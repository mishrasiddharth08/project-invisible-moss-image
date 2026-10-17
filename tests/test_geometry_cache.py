import unittest

import torch

from forge_moss.native.transformer import rope_table
from tests.test_transformer import inputs, pair


class TestGeometryChanges(unittest.TestCase):
    def test_repeated_and_changed_requests_match_reference(self):
        reference, model = pair(seed=7)
        with torch.inference_mode():
            for height, width, text in [(3, 5, 7), (3, 5, 7), (5, 3, 7), (5, 3, 9), (3, 5, 7)]:
                x, sigma, context = inputs(height=height, width=width, text=text)
                torch.testing.assert_close(model(x, sigma, context), reference(x, sigma * 1000, context),
                                           rtol=1e-4, atol=1e-5)
        self.assertFalse(any('cache' in key for key in model.state_dict()))

    def test_dtype_change_preserves_position_values(self):
        _, model = pair()
        device = torch.device('cpu')
        for dtype in (torch.float32, torch.bfloat16, torch.float32):
            got = model.positional_encoding(3, 5, 7, device, dtype)
            expected = rope_table(3, 5, 7, 16, 10000.0, device).to(dtype)
            torch.testing.assert_close(got, expected)
