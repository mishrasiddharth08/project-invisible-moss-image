"""Invisible, request-scoped native Spectrum hook; controls live in Settings."""
import time

from modules import scripts, shared

from forge_moss.native.spectrum import KEY, Session


class Script(scripts.Script):
    def title(self):
        return 'MOSS Spectrum'

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        return None

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

    def process_before_every_sampling(self, p, **kwargs):
        self._finish(p)
        p.extra_generation_params.pop('MOSS Spectrum calls', None)
        p.extra_generation_params.pop('MOSS Spectrum cache MiB', None)
        if not getattr(p.sd_model, 'is_moss_image', False):
            return
        p._moss_sampling_started = time.perf_counter()
        if getattr(shared.opts, 'moss_spectrum', 'Off') != 'Balanced':
            p.extra_generation_params['MOSS Spectrum'] = 'Off'
            return
        if (p.sampler_name != 'Euler' or str(p.scheduler).lower() != 'moss'
                or getattr(p, 'enable_hr', False)):
            p.extra_generation_params['MOSS Spectrum'] = 'Off (requires single-pass Euler/MOSS)'
            return
        denoiser = p.sampler.model_wrap_cfg
        session = Session(lambda: (denoiser.step, denoiser.total_steps or 0))
        unet = p.sd_model.forge_objects.unet.clone()
        unet.set_transformer_option(KEY, session)
        p.sd_model.forge_objects.unet = unet
        p._moss_spectrum_unet = unet
        p._moss_spectrum_session = session
        p.extra_generation_params['MOSS Spectrum'] = 'Balanced (approximate velocity forecasting)'
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
