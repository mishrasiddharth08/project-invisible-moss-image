"""Run only selected pure tests; fail if they attempt to import Torch."""
import importlib.abc,sys,unittest,tempfile
from pathlib import Path
class NoTorch(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname=='torch' or fullname.startswith('torch.'):
   raise ImportError('Training pause: real Torch import prohibited in lightweight validation')
sys.meta_path.insert(0,NoTorch())
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root));sys.argv=sys.argv[:1]
scratch=Path.cwd()/'lightweight-test-tmp';scratch.mkdir(exist_ok=True);tempfile.tempdir=str(scratch)
suite=unittest.TestSuite()
for name in ['test_assets','test_autolink','test_compat']:
 suite.addTests(unittest.defaultTestLoader.loadTestsFromName('tests.'+name))
result=unittest.TextTestRunner(verbosity=2).run(suite)
assert 'torch' not in sys.modules
sys.exit(not result.wasSuccessful())
