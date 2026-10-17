"""Independent Spectrum-style Chebyshev/ridge forecasting of MOSS velocities.

This is an opt-in approximation, not the upstream feature-forecasting adapter.
Histories contain real forwards only and never cross a request/sampling pass.
"""
from dataclasses import dataclass

import torch

from .settings import uncond_skip_layers

KEY = 'moss_spectrum_session'


@dataclass(frozen=True)
class Config:
    warmup: int = 8
    tail: int = 6
    stride: int = 3
    history: int = 4
    ridge: float = 0.01
    blend: float = 0.75
    max_change: float = 0.20
    max_bytes: int = 64 * 1024 * 1024


def weights(coords, target, ridge):
    """Degree-one Chebyshev fit; the tiny regression stays on the CPU."""
    values = torch.tensor(coords, dtype=torch.float64)
    span = values.max() - values.min()
    if len(coords) < 2 or span <= 1e-12 or not torch.isfinite(values).all():
        return None
    normalized = 2 * (values-values.min()) / span - 1
    design = torch.stack((torch.ones_like(normalized), normalized), dim=1)
    point = torch.tensor([1., 2*(target-float(values.min()))/float(span)-1], dtype=torch.float64)
    try:
        gram = design.T @ design + ridge * torch.eye(2, dtype=torch.float64)
        result = point @ torch.linalg.solve(gram, design.T)
    except RuntimeError:
        return None
    return result if torch.isfinite(result).all() and result.abs().sum() <= 4 else None


class Session:
    def __init__(self, progress, config=None):
        self.progress = progress
        self.config = config or Config()
        self.states = {}
        self.real = self.forecast = self.rejected = 0
        self.peak_bytes = 0

    def __deepcopy__(self, memo):
        # ModelPatcher clones options; this remains the same request-owned handle.
        return self

    def clear(self):
        self.states.clear()

    def close(self):
        self.clear()
        # Do not let a resident ModelPatcher retain a finished request/denoiser.
        self.progress = lambda: (0, 0)

    @property
    def nbytes(self):
        return sum(state['context'].numel()*state['context'].element_size()
                   + sum(value.numel()*value.element_size() for value in state['values'])
                   for state in self.states.values())

    def run(self, real_forward, x, timestep, context, transformer_options=None, **kwargs):
        opts = transformer_options or {}
        chunks = tuple(opts.get('cond_or_uncond') or ())
        step, total = self.progress()
        cfg = self.config
        # Unknown layouts and multi-evaluation samplers always take the real path.
        if (not chunks or any(value not in (0, 1) for value in chunks)
                or x.shape[0] % len(chunks) or total < cfg.warmup+cfg.tail+cfg.stride
                or kwargs.get('control') is not None or opts.get('patches') or opts.get('patches_replace')):
            self.clear()
            self.real += 1
            return real_forward(x, timestep, context, transformer_options=opts, **kwargs)
        if timestep.numel() > 1 and not bool((timestep == timestep.reshape(-1)[0]).all()):
            self.clear()
            self.real += 1
            return real_forward(x, timestep, context, transformer_options=opts, **kwargs)
        key = (chunks, tuple(x.shape), tuple(context.shape), x.device, x.dtype, context.dtype,
               uncond_skip_layers(42))
        if any(old_key[0] == chunks and old_key != key for old_key in self.states):
            self.clear()
        state = self.states.get(key)
        if state is not None and (step <= state['step'] or not torch.equal(context, state['context'])):
            self.clear()
            state = None
        # At most separate positive/negative histories or one concatenated history.
        if state is None and len(self.states) < 2:
            context_bytes = context.numel()*context.element_size()
            if self.nbytes + context_bytes < cfg.max_bytes:
                state = dict(context=context.detach().clone(), coords=[], values=[], step=-1, first=step)
                self.states[key] = state
        coord = float(timestep.reshape(-1)[0]) if state is not None else 0.
        eligible = (state is not None and step-state['first'] >= cfg.warmup
                    and step < total-cfg.tail and (step-cfg.warmup) % cfg.stride == 0
                    and len(state['values']) >= 2)
        prediction = None
        if eligible:
            coefficients = weights(state['coords'], coord, cfg.ridge)
            if coefficients is not None:
                last = state['values'][-1]
                prediction = torch.zeros_like(last, dtype=torch.float32)
                for coefficient, value in zip(coefficients.tolist(), state['values']):
                    prediction.add_(value.float(), alpha=coefficient*cfg.blend)
                prediction.add_(last.float(), alpha=1-cfg.blend)
                baseline = last.float().square().mean().sqrt().clamp_min(1e-8)
                change = (prediction-last.float()).square().mean().sqrt()/baseline
                if not bool(torch.isfinite(prediction).all()) or float(change) > cfg.max_change:
                    prediction = None
            if prediction is None:
                self.rejected += 1
        if prediction is not None:
            self.forecast += 1
            state['step'] = step
            return prediction.to(state['values'][-1].dtype)
        result = real_forward(x, timestep, context, transformer_options=opts, **kwargs)
        self.real += 1
        if state is not None:
            value_bytes = result.numel()*result.element_size()
            while state['values'] and (len(state['values']) >= cfg.history or self.nbytes+value_bytes > cfg.max_bytes):
                state['values'].pop(0)
                state['coords'].pop(0)
            if self.nbytes+value_bytes <= cfg.max_bytes:
                state['values'].append(result.detach().clone())
                state['coords'].append(coord)
                state['step'] = step
                self.peak_bytes = max(self.peak_bytes, self.nbytes)
        return result
