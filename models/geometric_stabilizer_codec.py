"""Exact recurrent storage weld; original Leech-LILA: Anatolii Kornienko.
DASHI BalancedTritBitFibre / TriadicMaskSignFactorization semantics.
Inference storage only. No quantization of arbitrary real activations.
"""
from dataclasses import dataclass
from functools import lru_cache
import torch
from .geometric_reflection_fibre import GROUP, act
from .geometric_orbit_descent import act_tensor

_ID = GROUP[0]
_NEG = (0, 0, 1)
_BITS = {torch.float16: torch.int16, torch.bfloat16: torch.int16,
         torch.float32: torch.int32, torch.float64: torch.int64}


def same_bits(a, b):
    return (a.shape == b.shape and a.dtype == b.dtype and
            torch.equal(a.contiguous().view(_BITS[a.dtype]),
                        b.contiguous().view(_BITS[b.dtype])))


@lru_cache(None)
def slot_plan(K):
    """World-frame representatives: a signed permutation of six slots.
    K is the actual history stabilizer, not a stale joint-control subgroup.
    """
    if _NEG in K:
        return (), (), ()
    unseen = set(range(6))
    reps, src, signs = [], [-1]*6, [0]*6
    while unseen:
        j = min(unseen)
        r = len(reps)
        reps.append(j)
        for k, b, a in K:
            branch, coord = divmod(j, 3)
            # Original R rolls left; J reverses and swaps paired branches.
            i = 3*(branch ^ b) + ((2-coord if b else coord)-k)%3
            sign = (-1)**a
            if src[i] != -1 and (src[i] != r or signs[i] != sign):
                raise ValueError("inconsistent stabilizer")
            src[i], signs[i] = r, sign
            unseen.discard(i)
    return tuple(reps), tuple(src), tuple(signs)


def _pack(bits):
    n = bits.numel()
    pad = (-n) % 8
    bits = torch.cat((bits.flatten(), bits.new_zeros(pad))).to(torch.int16)
    shifts = torch.arange(8, device=bits.device, dtype=torch.int16)
    return (bits.reshape(-1, 8) << shifts).sum(-1).to(torch.uint8)


def _unpack(data, n):
    shifts = torch.arange(8, device=data.device, dtype=torch.int16)
    return (((data.to(torch.int16)[:, None] >> shifts) & 1)
            .flatten()[:n].bool())


@dataclass
class Row:
    K: tuple
    mode: str
    values: torch.Tensor
    signs: torch.Tensor | None = None


@dataclass
class HistoryPacket:
    shape: tuple
    dtype: torch.dtype
    device: torch.device
    rows: tuple


def _slots(h):
    # [E,2,3T] -> [E,T,6], branch-major slots.
    e, _, w = h.shape
    return h.reshape(e, 2, w//3, 3).transpose(1, 2).reshape(e, w//3, 6)


def _history(z, e, w):
    return z.reshape(e, w//3, 2, 3).transpose(1, 2).reshape(e, 2, w)


def _scatter(v, K, e, w):
    _, src, signs = slot_plan(K)
    if _NEG in K:
        return v.new_zeros(e, 2, w)
    idx = torch.tensor(src, device=v.device, dtype=torch.long)
    neg = torch.tensor(signs, device=v.device) < 0
    z = v.index_select(-1, idx)
    return _history(torch.where(neg, -z, z), e, w)


@torch.no_grad()
def encode_history(h):
    if (h.ndim != 4 or not h.shape[0] or not h.shape[1] or
            h.shape[2] != 2 or h.shape[3] < 3 or h.shape[3]%3 or
            h.dtype not in _BITS or not torch.isfinite(h).all()):
        raise ValueError("finite paired floating history required")
    _, e, _, w = h.shape
    # A single global action across all experts/triads; world coordinates
    # retain the frame directly, rather than storing a redundant coset mark.
    fixed = torch.stack([(act_tensor(g,h,True)==h).flatten(1).all(1)
                         for g in GROUP], 1).tolist()
    rows = []
    for x, mask in zip(h, fixed):
        K = tuple(g for g, yes in zip(GROUP, mask) if yes)
        reps, _, _ = slot_plan(K)
        v = _slots(x)[..., list(reps)].contiguous()
        if not same_bits(_scatter(v,K,e,w), x):
            # Numeric equality can forget IEEE signed zero. Preserve the
            # original bytes rather than silently changing the source step.
            rows.append(Row((_ID,), "raw", x.clone()))
            continue
        if bool(((v==0)|(v==1)|(v==-1)).all()) and not bool(((v==0)&torch.signbit(v)).any()):
            support = v.flatten()!=0
            rows.append(Row(K, "trit", _pack(support),
                            _pack(v.flatten()[support]<0)))
        else:
            rows.append(Row(K, "real", v.clone()))
    return HistoryPacket(tuple(h.shape), h.dtype, h.device, tuple(rows))


@torch.no_grad()
def decode_history(p):
    b,e,_,w = p.shape
    if len(p.rows)!=b:
        raise ValueError("row count mismatch")
    out = []
    for row in p.rows:
        if row.mode=="raw":
            out.append(row.values)
            continue
        reps, _, _ = slot_plan(row.K)
        shape = (e,w//3,len(reps))
        if row.mode=="trit":
            n = e*(w//3)*len(reps)
            support = _unpack(row.values,n)
            signs = _unpack(row.signs,int(support.sum()))
            v = torch.zeros(n,device=p.device,dtype=p.dtype)
            v[support] = torch.where(signs,-1,1).to(p.dtype)
            v = v.reshape(shape)
        elif row.mode=="real":
            v = row.values
        else:
            raise ValueError("unknown packet mode")
        out.append(_scatter(v,row.K,e,w))
    return torch.stack(out)


@torch.no_grad()
def rollout_packed(model, seed, controls, tags):
    """Opt-in inference consumer, terminal logits + reversible final state.
    Same original step and readout; no historical trace consumer promised.
    No optimizer/gradient or physical speed claim.
    """
    from scripts.x369_source_geometry import signed_seed
    if model.training:
        raise ValueError("eval required")
    if (controls.ndim!=2 or controls.shape!=tags.shape or
            controls.shape[0]!=len(seed) or not controls.shape[1] or
            controls.dtype!=torch.long or tags.dtype!=torch.long or
            not ((tags>=0)&(tags<4)).all()):
        raise ValueError("indexed control and literal tag sequences required")
    packet = encode_history(signed_seed(seed,model.experts,model.width))
    for t in range(controls.shape[1]):
        h = decode_history(packet)
        tag = tags[:,t]
        out = model.history.step_controlled(controls[:,t],h,
            lambda c:model.control_embed(tag*len(GROUP)+c),train_sign=False)
        packet = encode_history(out.history)
        if t+1==controls.shape[1]:
            return model.consume(out),packet
        del h,out
