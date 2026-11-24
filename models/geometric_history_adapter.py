"""Geometry-forward LILA adapter; original Leech-LILA: Anatolii Kornienko.
No change to the original router, orientation, step or checkpoint.
"""
from copy import deepcopy
import torch
from torch import nn
from .geometric_cross_triads import CrossTriadBitLinear


def _lex_greater(a, b):
    different = a != b
    first = different.to(torch.int64).argmax(dim=-1, keepdim=True)
    return (a.gather(-1, first) > b.gather(-1, first))


def c3_orientation(z):
    """Odd, C3-invariant orientation on simultaneous ordered triads.

    Only zero returns zero; discrete chart, not a smooth training surrogate.
    Not S3-invariant and not a replacement for the original first-nonzero rule.
    """
    if z.ndim < 1 or z.shape[-1] < 3 or z.shape[-1] % 3:
        raise ValueError('complete triads required')
    if not torch.isfinite(z).all():
        raise ValueError('finite chart coordinates required')
    def canonical(v):
        t = v.reshape(*v.shape[:-1], -1, 3)
        best = v
        for k in (1, 2):
            w = t.roll(-k, dims=-1).reshape_as(v)
            best = torch.where(_lex_greater(w, best), w, best)
        return best
    a, b = canonical(z), canonical(-z)
    return _lex_greater(a, b).to(z.dtype) - _lex_greater(b, a).to(z.dtype)


def adapt_history_experts(source, witnesses, *, mode='c3', scale=1., ternary=True):
    """Frozen copy of the exact original expert/transition layout.

    All source linear layers within experts/transitions must have explicit
    witnesses. Never modify router/orientation or silently project weights.
    This is a fixed-weight inference adapter, not an optimizer migration.
    """
    from .history_orbit_moe import HistoryOrbitMoE
    if type(source) is not HistoryOrbitMoE:
        raise TypeError('expected original HistoryOrbitMoE')
    if source.d_model % 3:
        raise ValueError('source width must be divisible by three')
    layers = {name: layer for name, layer in source.named_modules()
              if isinstance(layer, nn.Linear) and name.split('.')[0] in ('experts', 'transitions')}
    if set(witnesses) != set(layers):
        raise ValueError('witness keys must exactly cover expert and transition linears')
    if not layers:
        raise ValueError('no expert or transition linears')
    replacements = {}
    for name, layer in layers.items():
        c = witnesses[name]
        if not torch.isfinite(c).all() or not torch.isfinite(layer.weight).all() or (layer.bias is not None and not torch.isfinite(layer.bias).all()):
            raise ValueError('nonfinite source or witness: ' + name)
        replacements[name] = CrossTriadBitLinear.from_linear(layer, c, mode=mode, scale=scale, ternary=ternary)
    candidate = deepcopy(source)
    for name, layer in replacements.items():
        parent, key = name.rsplit('.', 1)
        setattr(candidate.get_submodule(parent), key, layer)
    return candidate.eval().requires_grad_(False)
