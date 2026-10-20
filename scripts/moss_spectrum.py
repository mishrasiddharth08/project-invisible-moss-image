"""Inline, request-scoped MOSS acceleration controls; no accordion."""
import time
import math

from modules import scripts, shared

from forge_moss.native.spectrum import KEY, Session


class Script(scripts.Script):
    def title(self):
        return 'MOSS Spectrum'

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        import gradio as gr
        with gr.Group():
            with gr.Row():
                mode = gr.Dropdown(['Off', 'Balanced', 'Manual', 'Adaptive'],
                    value=getattr(shared.opts, 'moss_spectrum', 'Off'), label='MOSS Spectrum (approximate)')
                dmd = gr.Checkbox(False, label='MOSS SFT 8-step DMD preset (requires LoRA)')
                strength = gr.Slider(0.05, 2, value=0.7, step=0.05, label='MOSS DMD strength')
            with gr.Row():
                warmup = gr.Slider(0, 50, value=8, step=1, label='MOSS warmup real steps')
                tail = gr.Slider(1, 20, value=2, step=1, label='MOSS final real steps')
                horizon = gr.Slider(0, 8, value=2, step=1, label='MOSS manual forecast cap')
                guard = gr.Slider(0, 50, value=8, step=1, label='MOSS adaptive detail guard')
                cap = gr.Slider(1, 8, value=6, step=1, label='MOSS adaptive horizon cap')
            with gr.Row():
                tolerance = gr.Slider(0.25, 3, value=1, step=0.05, label='MOSS adaptive error tolerance')
                cache = gr.Radio(['cpu', 'main_device'], value='cpu', label='MOSS feature cache')
                memory = gr.Slider(64, 2048, value=512, step=64, label='MOSS feature cache limit MiB')
        controls = [mode, dmd, strength, warmup, tail, horizon, guard, cap, tolerance, cache, memory]
        self.infotext_fields = list(zip(controls, ['MOSS Spectrum mode', 'MOSS DMD preset',
            'MOSS DMD strength', 'MOSS warmup', 'MOSS tail', 'MOSS manual cap', 'MOSS guard',
            'MOSS adaptive cap', 'MOSS tolerance', 'MOSS cache', 'MOSS cache MiB']))
        return controls

    def process(self, p, mode=None, dmd=False, strength=0.7, warmup=8, tail=2,
                horizon=2, guard=8, cap=6, tolerance=1, cache='cpu', memory=512):
        from forge_moss.native.spectrum_policy import SpectrumConfig
        mode = mode or getattr(shared.opts, 'moss_spectrum', 'Off')
        if mode not in {'Off', 'Balanced', 'Manual', 'Adaptive'}:
            raise ValueError('Unknown MOSS Spectrum mode')
        if dmd:
            from forge_moss.native.dmd import apply
            apply(p, strength)
            # Eight-step distilled inference uses actual evaluations only.
            mode = 'Off'
        if not getattr(p.sd_model, 'is_moss_image', False):
            return
        limits = [(warmup, 0, 50), (tail, 1, 20), (horizon, 0, 8), (guard, 0, 50),
                  (cap, 1, 8), (tolerance, 0.25, 3)]
        if any(not math.isfinite(float(v)) or not low <= float(v) <= high for v, low, high in limits):
            raise ValueError('MOSS Spectrum parameter outside the inline control limits')
        if cache not in {'cpu', 'main_device'}:
            raise ValueError('MOSS feature cache must be cpu or main_device')
        if not math.isfinite(float(memory)) or not 64 <= float(memory) <= 2048:
            raise ValueError('MOSS feature cache must be 64..2048 MiB')
        config = SpectrumConfig(mode='adaptive' if mode == 'Adaptive' else 'manual',
            warmup_steps=int(warmup), tail_actual_steps=int(tail), max_consecutive_forecasts=int(horizon),
            detail_guard_steps=int(guard), post_guard_horizon_cap=int(cap), error_tolerance=float(tolerance),
            cache_device=cache)
        config.validate()
        p._moss_speed = (mode, config, int(memory)*1024**2)
        p.extra_generation_params.update(dict(zip(
            ['MOSS Spectrum mode', 'MOSS DMD preset', 'MOSS DMD strength', 'MOSS warmup', 'MOSS tail',
             'MOSS manual cap', 'MOSS guard', 'MOSS adaptive cap', 'MOSS tolerance', 'MOSS cache', 'MOSS cache MiB'],
            [mode, bool(dmd), float(strength), int(warmup), int(tail), int(horizon), int(guard), int(cap),
             float(tolerance), cache, int(memory)])))
        if mode in {'Manual', 'Adaptive'}:
            p.extra_generation_params['MOSS Spectrum settings'] = (
                f'warmup={warmup};tail={tail};manual_cap={horizon};guard={guard};'
                f'adaptive_cap={cap};tolerance={tolerance};cache={cache};MiB={memory}')

    def _finish(self, p):
        started = getattr(p, '_moss_sampling_started', None)
        if started is not None:
            p.extra_generation_params['MOSS sampling+decode seconds'] = round(time.perf_counter()-started, 4)
            p._moss_sampling_started = None
        session = getattr(p, '_moss_spectrum_session', None)
        if session is not None:
            if not session.forecast:
                p.extra_generation_params['MOSS Spectrum'] = 'Off (no forecast qualified)'
            p.extra_generation_params['MOSS Spectrum calls'] = f'{session.real} real, {session.forecast} forecast, {session.rejected} rejected'
            p.extra_generation_params['MOSS Spectrum cache MiB'] = round(session.peak_bytes/1024**2, 2)
            session.close()
            unet = getattr(p, '_moss_spectrum_unet', None)
            options = getattr(unet, 'model_options', {}).get('transformer_options', {})
            if options.get(KEY) is session:
                options.pop(KEY)
            p._moss_spectrum_unet = None
            p._moss_spectrum_session = None

    def process_before_every_sampling(self, p, *args, **kwargs):
        self._finish(p)
        p.extra_generation_params.pop('MOSS Spectrum calls', None)
        p.extra_generation_params.pop('MOSS Spectrum cache MiB', None)
        if not getattr(p.sd_model, 'is_moss_image', False):
            return
        p._moss_sampling_started = time.perf_counter()
        mode, config, max_bytes = getattr(p, '_moss_speed',
            (getattr(shared.opts, 'moss_spectrum', 'Off'), None, 512*1024**2))
        if mode not in {'Balanced', 'Manual', 'Adaptive'}:
            p.extra_generation_params['MOSS Spectrum'] = 'Off'
            return
        if (p.sampler_name != 'Euler' or str(p.scheduler).lower() != 'moss'
                or getattr(p, 'enable_hr', False)):
            p.extra_generation_params['MOSS Spectrum'] = 'Off (requires single-pass Euler/MOSS)'
            return
        denoiser = p.sampler.model_wrap_cfg
        progress = lambda: (denoiser.step, denoiser.total_steps or 0)
        if mode == 'Balanced':
            session = Session(progress)
        else:
            from forge_moss.native.spectrum_hidden import HiddenSession
            session = HiddenSession(progress, config, max_bytes)
        unet = p.sd_model.forge_objects.unet.clone()
        unet.set_transformer_option(KEY, session)
        p.sd_model.forge_objects.unet = unet
        p._moss_spectrum_unet = unet
        p._moss_spectrum_session = session
        p.extra_generation_params['MOSS Spectrum'] = (
            'Balanced (approximate velocity forecasting)' if mode == 'Balanced' else
            f'{mode} (Rkkss hidden-state forecasting)')
        if not getattr(p, '_moss_spectrum_close', False):
            original_close = p.close
            previous_override = p.__dict__.get('close')
            had_override = 'close' in p.__dict__
            def close(*args, **kwargs):
                try:
                    self._finish(p)
                    return original_close(*args, **kwargs)
                finally:
                    if had_override:
                        p.close = previous_override
                    else:
                        del p.close
                    p._moss_spectrum_close = False
            p.close = close
            p._moss_spectrum_close = True

    def postprocess_batch(self, p, *args, **kwargs):
        self._finish(p)

    def postprocess(self, p, processed, *args):
        self._finish(p)
