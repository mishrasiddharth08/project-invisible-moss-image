"""Run a fixed-seed request through Forge's real processing and PNG save path."""
import argparse
import json
import os
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--forge', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--size', type=int, default=1024)
    parser.add_argument('--img2img-source', type=Path)
    parser.add_argument('--prompt', default='A studio photograph of a red ceramic mug on a wooden table. A white card reads "MOSS WORKS".')
    parser.add_argument('--request-json', type=Path)
    parser.add_argument('--allow-model-override', action='store_true')
    parser.add_argument('--width', type=int)
    parser.add_argument('--height', type=int)
    args = parser.parse_args()
    width, height = args.width or args.size, args.height or args.size
    negative_prompt = ''
    json_request = {}
    if args.request_json:
        json_request = json.loads(args.request_json.read_text(encoding='utf-8'))
        unknown = set(json_request) - {'prompt', 'negative_prompt', 'style', 'aspect_ratio', 'model'}
        if unknown:
            raise ValueError(f'Unsupported JSON fields: {sorted(unknown)}')
        if json_request.get('model', 'moss-image-sft') not in {'moss-image-sft', 'MOSS-Image-SFT'}:
            if not args.allow_model_override:
                raise ValueError('JSON requests another model; use explicit --allow-model-override for a MOSS comparison')
        args.prompt = json_request['prompt']
        negative_prompt = json_request.get('negative_prompt', '')
        if json_request.get('style') and json_request['style'] not in args.prompt:
            args.prompt += '\nStyle: ' + json_request['style']
        if json_request.get('aspect_ratio'):
            a, b = map(int, json_request['aspect_ratio'].split(':'))
            assert width * b == height * a, 'Dimensions do not match JSON aspect ratio'
    assert width > 0 and height > 0 and width % 16 == height % 16 == 0
    args.output.mkdir(parents=True, exist_ok=True)
    previous_files = set(args.output.rglob('*.png'))
    config = args.output / 'isolated-config.json'
    config.write_text('{}', encoding='utf-8')
    sys.argv = [sys.argv[0], '--disable-extra-extensions', '--skip-version-check',
                '--disable-console-progressbars', '--ui-settings-file', str(config)]
    os.chdir(args.forge)
    sys.path.insert(0, str(args.forge))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from modules_forge.initialization import initialize_forge
    initialize_forge()
    from modules import initialize
    initialize.imports()
    initialize.initialize()
    from forge_moss.native import compat, patches
    problems = compat.check()
    if problems:
        raise RuntimeError(problems)
    patches.apply()
    from modules import processing, sd_models, shared
    from modules_forge.main_entry import refresh_model_loading_parameters
    from forge_moss.native.assets import discover
    found = discover(args.forge/'models')
    assert found, 'Missing compatible MOSS checkpoint, encoder or full FLUX.2 VAE'
    checkpoint_path, encoder_path, vae_path = found
    checkpoint = checkpoint_path.name
    for path in found:
        assert path.is_file(), f'Missing required model: {path}'
    shared.opts.data.update(sd_model_checkpoint=checkpoint, forge_unet_storage_dtype='Automatic',
        forge_additional_modules=[str(encoder_path), str(vae_path)],
        samples_save=True, grid_save=False, samples_format='png')
    sd_models.list_models()
    refresh_model_loading_parameters()
    assert Path(sd_models.model_data.forge_loading_parameters['checkpoint_info'].filename).name == checkpoint
    import torch
    from PIL import Image
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    shared.state.begin(job='MOSS live check')
    try:
        task_class = processing.StableDiffusionProcessingTxt2Img
        task_options = {}
        if args.img2img_source:
            task_class = processing.StableDiffusionProcessingImg2Img
            task_options = dict(init_images=[Image.open(args.img2img_source).convert('RGB')], denoising_strength=0.6)
        request = task_class(
            prompt=args.prompt,
            negative_prompt=negative_prompt, seed=123456, width=width, height=height,
            steps=args.steps, cfg_scale=3, sampler_name='Euler', scheduler='MOSS',
            batch_size=1, n_iter=1, outpath_samples=str(args.output),
            outpath_grids=str(args.output), do_not_save_grid=True, **task_options)
        try:
            result = processing.process_images(request)
        finally:
            request.close()
        assert getattr(shared.sd_model, 'is_moss_image', False), 'Wrong engine loaded'
        files = sorted(set(args.output.rglob('*.png')) - previous_files, key=lambda p: p.stat().st_mtime)
        assert result.images and files, 'Missing output/gallery image'
        image = Image.open(files[-1])
        assert image.size == (width, height)
        assert any(lo != hi for lo, hi in image.convert('RGB').getextrema()), 'Blank output'
        assert '123456' in image.info.get('parameters', ''), 'Missing seed metadata'
        assert image.convert('RGB').tobytes() == result.images[0].convert('RGB').tobytes(), 'Gallery/save mismatch'
        report = dict(seconds=time.perf_counter()-started,
            peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
            peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3,
            checkpoint=checkpoint, seed=123456, steps=args.steps, size=args.size,
            width=width, height=height, json_request=json_request,
            task='img2img' if args.img2img_source else 'txt2img',
            output=str(files[-1]), infotext=result.info, torch=torch.__version__,
            gpu=torch.cuda.get_device_name())
        (args.output/'result.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print('LIVE_CHECK_PASSED '+json.dumps(report), flush=True)
    finally:
        shared.total_tqdm.clear()
        shared.state.end()
        # Drop the engine while CUDA and custom operations are still alive.
        sd_models.unload_model_weights()


if __name__ == '__main__':
    main()
