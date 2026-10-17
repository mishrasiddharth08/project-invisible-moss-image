"""Verify full Forge UI startup and optional HTTP generation with installed extensions."""
import argparse
import base64
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

import psutil
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--forge', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--port', type=int, default=7862)
    parser.add_argument('--generate', action='store_true')
    parser.add_argument('--spectrum-benchmark', action='store_true')
    parser.add_argument('--spectrum-cases', default='text,portrait,hands')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    endpoint = f'http://127.0.0.1:{args.port}'
    def request(path, data=None, timeout=10):
        body = json.dumps(data).encode() if data is not None else None
        req = urllib.request.Request(endpoint+path, data=body, headers={'Content-Type': 'application/json'})
        with opener.open(req, timeout=timeout) as response:
            return json.load(response)
    config = json.loads((args.forge/'config.json').read_text(encoding='utf-8'))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from forge_moss.native.assets import discover
    checkpoint, encoder, vae = discover(args.forge/'models')
    config.update(forge_preset='moss', samples_save=True, grid_save=False,
                  sd_model_checkpoint=checkpoint.name, forge_checkpoint_moss=checkpoint.name,
                  forge_additional_modules=[str(encoder), str(vae)],
                  forge_additional_modules_moss=[str(encoder), str(vae)],
                  outdir_samples=str(args.output/'images'), outdir_txt2img_samples=str(args.output/'images'))
    if args.spectrum_benchmark:
        config['persistent_cond_cache'] = False
    config_path = args.output/'isolated-config.json'
    config_path.write_text(json.dumps(config), encoding='utf-8')
    command = [str(args.forge/'venv/Scripts/python.exe'), '-u', 'launch.py', '--api',
        '--port', str(args.port), '--skip-install', '--skip-prepare-environment',
        '--skip-version-check', '--skip-python-version-check', '--skip-torch-cuda-test',
        '--disable-console-progressbars', '--ui-settings-file', str(config_path),
        '--ui-config-file', str(args.output/'ui-config.json')]
    with (args.output/'startup.log').open('wb') as log:
        process = subprocess.Popen(command, cwd=args.forge, stdout=log, stderr=subprocess.STDOUT,
            env=dict(os.environ, PYTHONUTF8='1'), creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        try:
            deadline = time.monotonic()+180
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f'Forge exited early: {process.returncode}; see startup.log')
                try:
                    ui = request('/config')
                    break
                except (OSError, ValueError):
                    if time.monotonic() > deadline:
                        raise TimeoutError('Forge startup timed out; see startup.log')
                    time.sleep(2)
            preset = next(c['props'] for c in ui['components'] if c.get('props', {}).get('label') == 'UI Preset')
            assert any('moss' in choice for choice in preset['choices']), preset['choices']
            while True:
                try:
                    options = request('/sdapi/v1/options')
                    break
                except OSError:
                    if time.monotonic() > deadline:
                        raise TimeoutError('Forge API startup timed out; see startup.log')
                    time.sleep(2)
            # Forge's static API schema omits extension-added hidden preset keys.
            checkpoint = options.get('sd_model_checkpoint')
            modules = options.get('forge_additional_modules')
            assert checkpoint and len(modules or []) == 2, (checkpoint, modules)
            assert preset.get('value') == 'moss', preset.get('value')
            print('FULL_UI_PRESET_PASSED', checkpoint, modules, flush=True)
            report = {'preset': 'moss', 'checkpoint': checkpoint, 'modules': modules}
            if args.generate or args.spectrum_benchmark:
                request('/sdapi/v1/options', {'sd_model_checkpoint': checkpoint,
                    'forge_additional_modules': modules, 'forge_preset': 'moss'})
                started = time.perf_counter()
                request('/sdapi/v1/options', {'moss_spectrum': 'Off'})
                result = request('/sdapi/v1/txt2img', {'prompt': 'A red ceramic mug on a wooden table.',
                    'negative_prompt': '', 'seed': 123456,
                    'width': 1024 if args.spectrum_benchmark else 512,
                    'height': 1024 if args.spectrum_benchmark else 512,
                    'steps': 4, 'cfg_scale': 3, 'sampler_name': 'Euler', 'scheduler': 'MOSS',
                    'batch_size': 1, 'n_iter': 1, 'save_images': True}, timeout=180)
                assert result.get('images'), result
                image = Image.open(io.BytesIO(base64.b64decode(result['images'][0])))
                expected_size = 1024 if args.spectrum_benchmark else 512
                assert image.size == (expected_size, expected_size)
                assert any(a != b for a, b in image.convert('RGB').getextrema())
                info = json.loads(result['info'])
                assert info['all_seeds'] == [123456]
                assert 'moss_image' in info.get('sd_model_name', '') or 'moss_image' in str(info.get('infotexts', ''))
                image.save(args.output/'api-result.png')
                report.update(seconds=time.perf_counter()-started, info=info)
                print('FULL_EXTENSIONS_API_GENERATION_PASSED', flush=True)
            if args.spectrum_benchmark:
                cases = [
                    ('text', 'A realistic studio photograph of a red ceramic mug on a wooden table beside a white card reading "MOSS WORKS". Fine ceramic texture, soft natural window light.'),
                    ('portrait', 'Photorealistic head and shoulders portrait of an adult woman aged 35, dark brown wavy hair, natural skin texture and freckles, softly lit by a window, ivory linen blouse, neutral studio background, 85mm lens, sharp eyes, subtle natural makeup.'),
                    ('hands', 'A photorealistic closeup of two adult human hands carefully holding a small handmade ceramic cup, all fingers clearly visible and anatomically correct, natural skin creases, warm pottery studio window light, detailed editorial photograph.'),
                ]
                rows = []
                import numpy as np
                import re
                for name, prompt in cases:
                    if name not in args.spectrum_cases.split(','):
                        continue
                    # Warm this exact conditioning and attention shape before either timing.
                    request('/sdapi/v1/options', {'moss_spectrum': 'Off'})
                    request('/sdapi/v1/txt2img', {'prompt': prompt, 'negative_prompt': '',
                        'seed': 123456, 'width': 1024, 'height': 1024, 'steps': 4,
                        'cfg_scale': 3, 'sampler_name': 'Euler', 'scheduler': 'MOSS',
                        'batch_size': 1, 'n_iter': 1, 'save_images': False}, timeout=180)
                    baseline = None
                    for mode in ('Off', 'Balanced'):
                        request('/sdapi/v1/options', {'moss_spectrum': mode})
                        started = time.perf_counter()
                        result = request('/sdapi/v1/txt2img', {'prompt': prompt, 'negative_prompt': '',
                            'seed': 123456, 'width': 1024, 'height': 1024, 'steps': 50,
                            'cfg_scale': 3, 'sampler_name': 'Euler', 'scheduler': 'MOSS',
                            'batch_size': 1, 'n_iter': 1, 'save_images': True}, timeout=240)
                        elapsed = time.perf_counter()-started
                        image = Image.open(io.BytesIO(base64.b64decode(result['images'][0]))).convert('RGB')
                        image.save(args.output/f'{name}-{mode.lower()}.png')
                        metadata = json.loads(result['info'])
                        params = metadata.get('extra_generation_params', {})
                        array = np.array(image, dtype=np.float32)
                        row = dict(case=name, mode=mode, seconds=elapsed, params=params)
                        if mode == 'Off':
                            baseline = array
                        else:
                            counts = re.search(r'(\d+) real, (\d+) forecast', str(params.get('MOSS Spectrum calls')))
                            assert counts and int(counts[2]) > 0, params
                            row.update(real_calls=int(counts[1]), forecast_calls=int(counts[2]))
                            row['mean_absolute_pixel_difference'] = float(np.abs(array-baseline).mean())
                        rows.append(row)
                        print('SPECTRUM_RESULT', json.dumps(row), flush=True)
                report['spectrum'] = rows
            (args.output/'result.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        finally:
            try:
                owned = psutil.Process(process.pid)
                children = owned.children(recursive=True)
                for child in children:
                    child.terminate()
                owned.terminate()
                _, alive = psutil.wait_procs(children+[owned], timeout=10)
                for child in alive:
                    child.kill()
            except psutil.NoSuchProcess:
                pass


if __name__ == '__main__':
    main()
