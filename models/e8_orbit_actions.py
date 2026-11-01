"""Small exact Weyl E8 subgroup actions on eight-coordinate blocks.

Every action is a signed coordinate permutation preserving the 240 E8
roots. These actions are much smaller than the full E8 Weyl group.
They are NOT punctured-nine site actions, not necessarily Leech lattice
automorphisms and not semantic opposites.

Torch actions are differentiated by autograd. IEEE binary16/32 byte
actions preserve every original bit exactly, including NaN payloads.
"""
from __future__ import annotations
import torch

# dest coordinate j reads source coordinate permutation[j].
# number of negative signs must be even to preserve E8 half-root parity.
WEYL_ACTIONS={
    "identity":((0,1,2,3,4,5,6,7),()),
    "swap01":((1,0,2,3,4,5,6,7),()),
    "flip01":((0,1,2,3,4,5,6,7),(0,1)),
    "negative8":((0,1,2,3,4,5,6,7),tuple(range(8))),
}


def _action(action:str):
    if action not in WEYL_ACTIONS:
        raise ValueError("unknown E8 root-system action")
    permutation,signs=WEYL_ACTIONS[action]
    if sorted(permutation)!=list(range(8)) or len(signs)%2:
        raise ValueError("not an allowed E8 signed coordinate permutation")
    return permutation,signs


def e8_orbit_transform(x:torch.Tensor,action:str)->torch.Tensor:
    if x.shape[-1]%8:
        raise ValueError("E8 input width must be a multiple of 8")
    order,negative=_action(action)
    t=x.reshape(*x.shape[:-1],-1,8)[...,list(order)].clone()
    if negative:
        multiplier=t.new_ones(8)
        multiplier[list(negative)]=-1.
        t=t*multiplier
    return t.reshape(x.shape)


def e8_orbit_bytes(raw:bytes,itemsize:int,action:str)->bytes:
    if itemsize not in (2,4) or not isinstance(raw,bytes) or (
            len(raw)%(8*itemsize)):
        raise ValueError("E8 byte payload must contain 8 complete IEEE words")
    order,negative=_action(action)
    width=8*itemsize
    output=bytearray(len(raw))
    for begin in range(0,len(raw),width):
        for j,source in enumerate(order):
            old=begin+source*itemsize
            new=begin+j*itemsize
            output[new:new+itemsize]=raw[old:old+itemsize]
            if j in negative:output[new+itemsize-1]^=0x80
    return bytes(output)


def e8_inverse_action(action:str)->str:
    # Deliberately choose an involutive, easy-to-audit action set.
    _action(action)
    return action
