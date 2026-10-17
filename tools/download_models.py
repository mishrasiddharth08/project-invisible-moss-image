"""Explicit setup only: download the matched, pinned Comfy-Org INT8 components."""
import argparse
import hashlib
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REVISION = '6c170adf6af419f9877872ae9c7d397fcb341b99'
HASHES = {
    'moss_image_sft_int8_convrot.safetensors': 'b4d1475bc7eeffd7b809e82ea13a1fba1fb85c4e18365af3813722faa5c0fe4f',
    'qwen3.5_9b_int8_convrot.safetensors': '32e3787472f708fa5f83624acbf5ef23f7fc89c439e05065ea57cfe5a97f025a',
    'flux2-vae.safetensors': '868fe7b343cc8f3a19dbcfcafbc3d5f888802be3f89bd81b65b3621a066ce8f3',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forge", required=True, type=Path)
    args = parser.parse_args()
    repo = "Comfy-Org/MOSS-Image"
    with urllib.request.urlopen(f"https://huggingface.co/api/models/{repo}/revision/{REVISION}?blobs=true") as response:
        info = json.load(response)
    revision = info['sha']
    assert revision == REVISION, 'Unexpected model revision'
    files = {f['rfilename']: f for f in info['siblings']}
    print(f"Source: {repo}@{revision}", flush=True)
    def download(item):
        folder, filename, target = item
        spec = files[f'{folder}/{filename}']
        destination = args.forge / 'models' / target / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            for candidate in destination.parent.rglob('*.safetensors'):
                if candidate.stat().st_size != spec['size']:
                    continue
                with candidate.open('rb') as handle:
                    existing_hash = hashlib.file_digest(handle, 'sha256').hexdigest()
                if existing_hash == HASHES[filename]:
                    print(f'Verified existing (possibly renamed): {candidate}', flush=True)
                    return
        partial = destination.with_suffix('.safetensors.partial')
        if destination.exists():
            partial = destination
        offset = partial.stat().st_size if partial.exists() else 0
        while offset < spec['size']:
            end = min(offset + 256 * 1024 * 1024, spec['size']) - 1
            request = urllib.request.Request(
                f'https://huggingface.co/{repo}/resolve/{revision}/{folder}/{filename}?range={offset}',
                headers={'Range': f'bytes={offset}-{end}'})
            with urllib.request.urlopen(request, timeout=60) as response:
                if response.status != 206 or not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                    raise RuntimeError(f'Server did not honor download range: {filename}')
                with partial.open('ab') as out:
                    while chunk := response.read(1024 * 1024):
                        out.write(chunk)
            offset = partial.stat().st_size
            print(f'{filename}: {offset * 100 / spec["size"]:.1f}%', flush=True)
        with partial.open('rb') as handle:
            digest = hashlib.file_digest(handle, 'sha256').hexdigest()
        if partial.stat().st_size != spec['size'] or digest != HASHES[filename]:
            raise RuntimeError(f'Incomplete or incorrect download: {filename}')
        if partial != destination:
            partial.replace(destination)
        print(f'Verified: {destination}', flush=True)

    items = (
        ("diffusion_models", "moss_image_sft_int8_convrot.safetensors", "Stable-diffusion"),
        ("text_encoders", "qwen3.5_9b_int8_convrot.safetensors", "text_encoder"),
        ("vae", "flux2-vae.safetensors", "VAE"),
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(download, items))
    print("DOWNLOAD_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
