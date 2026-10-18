import json
from pathlib import Path
import struct
import tempfile
import unittest

from forge_moss.native import assets, checkpoints

FIXTURES = Path(__file__).parent / 'fixtures'


def write_header(path, state):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(state).encode()
    path.write_bytes(struct.pack('<Q', len(data)) + data)
    return path


class TestAssetDiscovery(unittest.TestCase):
    def test_canonical_renamed_vae_and_encoder_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = json.loads((FIXTURES / 'moss_image_sft_int8_convrot_attn.shapes.json').read_text())
            encoder = json.loads((FIXTURES / 'qwen3.5_9b_int8_convrot.shapes.json').read_text())
            vae = {key: {'shape': shape} for key, shape in checkpoints.VAE_SHAPES.items()}
            vae['encoder.quant_conv.weight'] = vae.pop('quant_conv.weight')
            vae['decoder.post_quant_conv.weight'] = vae.pop('post_quant_conv.weight')
            expected = (
                write_header(root / 'Stable-diffusion/moss_image_sft_int8_convrot.safetensors', model),
                write_header(root / 'text_encoder/qwen3.5_9b_int8_convrot.safetensors', encoder),
                write_header(root / 'VAE/FLUX.2-VAE--Renamed-by-Autolink.safetensors', vae),
            )
            self.assertEqual(assets.discover(root), expected)
            self.assertFalse(assets.valid_encoder(expected[0]))
            self.assertFalse(assets.valid_checkpoint(expected[1]))
            self.assertFalse(assets.valid_vae(expected[1]))

    def test_corrupt_header_and_incomplete_components_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'moss.safetensors'
            path.write_bytes(b'bad')
            self.assertFalse(assets.valid_checkpoint(path))
            self.assertFalse(assets.valid_encoder(path))
            self.assertFalse(assets.valid_vae(path))
            self.assertIsNone(assets.discover(root))


    def test_malformed_encoder_norm_is_rejected_with_clear_error(self):
        source = json.loads((FIXTURES / 'qwen3.5_9b_int8_convrot.shapes.json').read_text())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'qwen3.5-malformed.safetensors'
            for shape in ([], [4096, 1], [1, 4096], [4095]):
                with self.subTest(shape=shape):
                    source['model.language_model.norm.weight']['shape'] = shape
                    with self.assertRaisesRegex(ValueError, 'another size'):
                        checkpoints.text_encoder_prefix(source)
                    write_header(path, source)
                    self.assertFalse(assets.valid_encoder(path))
