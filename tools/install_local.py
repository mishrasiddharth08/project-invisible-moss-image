"""Copy this reviewed extension to a requested Forge folder, verifying each file."""
import argparse
import hashlib
from pathlib import Path
import shutil

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--forge', required=True, type=Path)
args = parser.parse_args()
source = Path(__file__).resolve().parents[1]
target = args.forge / 'extensions' / 'project-invisible-moss-image'
assert (args.forge / 'modules/processing.py').is_file(), 'Not a Forge folder'
count = 0
for path in source.rglob('*'):
    relative = path.relative_to(source)
    if not path.is_file() or any(part in {'.git', '.pytest_cache', '__pycache__'} for part in relative.parts):
        continue
    destination = target / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, destination)
    assert hashlib.sha256(path.read_bytes()).digest() == hashlib.sha256(destination.read_bytes()).digest()
    count += 1
print(f'Installed and verified {count} files: {target}')
