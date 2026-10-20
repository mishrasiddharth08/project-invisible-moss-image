"""Forge request adapter for Rkkss' GPL Spectrum hidden-state policy.

The shared output head always runs at the current timestep. Histories contain
real last-block image features only, isolated by CFG branch and conditioning.
"""
import torch

from . import settings
from .spectrum_policy import (SpectrumBranchState, _adaptive_validate_and_update,
                              _choose_actual, _normalize_step_position, _note_decision)


class HiddenSession:
    def __init__(self, progress, config, max_bytes=512*1024**2):
        config.validate()
        self.progress, self.config, self.max_bytes = progress, config, max_bytes
        self.states = {}
        self.real = self.forecast = self.rejected = self.peak_bytes = 0

    def __deepcopy__(self, memo):
        return self

    @property
    def nbytes(self):
        return sum(s['context'].numel()*s['context'].element_size() +
                   sum(a.feature.numel()*a.feature.element_size() for a in s['branch'].forecaster._anchors)
                   for s in self.states.values())

    def close(self):
        self.states.clear()
        self.progress = lambda: (0, 0)

    def run(self, real_forward, x, timestep, context, transformer_options=None, **kwargs):
        opts = transformer_options or {}
        step, total = self.progress()
        chunks = tuple(opts.get('cond_or_uncond') or ())
        core = real_forward.__self__
        cfg = self.config
        supported = (chunks and all(c in (0, 1) for c in chunks) and x.shape[0] % len(chunks) == 0
                     and total > cfg.warmup_steps+cfg.tail_actual_steps and step >= 0
                     and not opts.get('patches') and not opts.get('patches_replace')
                     and kwargs.get('control') is None and opts.get('control') is None
                     and not (1 in chunks and len(core.layers)-1 in settings.uncond_skip_layers(len(core.layers)))
                     and bool((timestep == timestep.reshape(-1)[0]).all()))
        if not supported:
            self.states.clear()
            self.real += 1
            return real_forward(x, timestep, context, transformer_options=opts, **kwargs)
        key = (chunks, tuple(x.shape), tuple(context.shape), x.device, x.dtype, context.dtype,
               settings.uncond_skip_layers(len(core.layers)), total)
        # A layout/precision/skip change invalidates the branch, as does step rewind.
        for old in list(self.states):
            if old[0] == chunks and old != key:
                self.states.pop(old)
        state = self.states.get(key)
        if state and (step <= state['step'] or not torch.equal(context, state['context'])):
            self.states.pop(key)
            state = None
        if state is None:
            context_bytes = context.numel()*context.element_size()
            if len(self.states) >= 2 or self.nbytes+context_bytes > self.max_bytes:
                self.real += 1
                return real_forward(x, timestep, context, transformer_options=opts, **kwargs)
            state = dict(context=context.detach().clone(), branch=SpectrumBranchState(cfg), step=-1)
            self.states[key] = state
        branch = state['branch']
        coord = _normalize_step_position(step, total)
        actual, reason = _choose_actual(branch, step, total, False, cfg)
        if not actual:
            try:
                hidden = branch.forecaster.predict(coord)
                if not bool(torch.isfinite(hidden).all()):
                    raise ValueError('nonfinite hidden forecast')
                result = core.forecast_head(x, timestep, context.dtype, hidden)
                if not bool(torch.isfinite(result).all()):
                    raise ValueError('nonfinite output forecast')
            except (RuntimeError, ValueError):
                self.rejected += 1
                actual, reason = True, 'forecast_failure'
        if actual:
            captured = []
            def capture(module, args, output):
                feature = output[:, :x.shape[-2]*x.shape[-1]]
                size = feature.numel()*feature.element_size()
                # Account for temporary capture as well as stored anchors.
                if self.nbytes+size <= self.max_bytes:
                    captured.append(feature.detach().to('cpu' if cfg.cache_device == 'cpu' else x.device,
                                                        copy=True).contiguous())
            handle = core.layers[-1].register_forward_hook(capture)
            try:
                result = real_forward(x, timestep, context, transformer_options=opts, **kwargs)
            finally:
                handle.remove()
            if captured:
                action_start = cfg.warmup_steps + (cfg.detail_guard_steps if cfg.adaptive else 0)
                _adaptive_validate_and_update(cfg, branch, coord, captured[0],
                    action_start <= step < total-cfg.tail_actual_steps and reason != 'local_repair',
                    reason not in {'local_repair', 'tail'}, branch.consecutive_forecasts)
                branch.forecaster.update(coord, captured[0])
                self.peak_bytes = max(self.peak_bytes, self.nbytes)
            else:
                # Missing capture (e.g. skipped last block) must not use stale anchors.
                branch.reset()
            self.real += 1
        else:
            self.forecast += 1
        _note_decision(branch, actual, reason, cfg)
        state['step'] = step
        if step >= total-1:
            branch.forecaster.release_features()
        return result
