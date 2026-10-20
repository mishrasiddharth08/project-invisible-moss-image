"""CPU numerical integration checks; never initialize CUDA."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import sys
import unittest
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
torch.set_num_threads(2)
def prohibited(*args, **kwargs):
    raise AssertionError('CUDA initialization prohibited during CPU checks')
torch.cuda._lazy_init = prohibited
suite = unittest.TestSuite()
for name in ['test_speed_integration', 'test_spectrum', 'test_spectrum_lifecycle', 'test_scheduler_lora']:
    suite.addTests(unittest.defaultTestLoader.loadTestsFromName('tests.'+name))
result = unittest.TextTestRunner(verbosity=2).run(suite)
assert not torch.cuda.is_initialized()
sys.exit(not result.wasSuccessful())
