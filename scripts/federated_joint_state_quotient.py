"""Whole-state, consumer-scoped, exact federated MoE quotient (JQ1).

The source state is (one SHARED activation, E independent expert memories).
An admitted *selected-execution* consumer needs the activation and only the
k selected memories, provided the other memories remain retained at their
authoritative locations. The packet never pretends k memories are all E.

Candidate exact representations:
  raw: one x and k histories, no per-expert duplicate x;
  zlib: jointly compress the same state;
  orbit: bytewise XOR corrections against x, IEEE sign(x), previous
         selected histories and optionally receiver-ACKed baseline.
This is true joint cross-expert prediction, not E projected router copies.

IEEE sign-bit action is exact on binary16/32 storage, including -0 and
NaN payloads. Fourier translations and "semantic opposite" are NOT
asserted to preserve raw IEEE bytes. Geometric/analytic transforms need
their own exact correction. This codec carries all correction bytes.

A caller may pass a baseline ONLY with a matching acknowledged revision.
The receiving peer MUST supply that baseline and revision and compare its
hash. SHA256 is integrity checking, NOT authentication or proof of ACK;
bind JQ1 in an authenticated session envelope before remote acceptance.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
import struct
import zlib
from typing import Mapping
from scripts.ieee_sign_orbit_codec import flip_ieee_signs
from scripts.punctured_nine_exact_site_actions import (
    ACTIONS,permute_punctured_nine_bytes)

MAGIC=b"JQ1!"
MAX_HEADER=16384
MAX_UNCOMPRESSED=64*1024*1024


@dataclass(frozen=True)
class JointState:
    activation: bytes
    histories: Mapping[int, bytes]


def _sha(raw:bytes)->str:
    return hashlib.sha256(raw).hexdigest()


def _xor(a:bytes,b:bytes)->bytes:
    if len(a)!=len(b):
        raise ValueError("XOR operands must match in length")
    return bytes(x^y for x,y in zip(a,b))


def _validate(state:JointState, selected:tuple[int,...], count:int,
              itemsize:int, width:int, limit:int)->int:
    if itemsize not in (2,4) or width<1 or count<1 or len(selected)<1:
        raise ValueError("unsupported layout or empty consumer")
    if any(type(e) is not int or not 0<=e<count for e in selected):
        raise ValueError("selected expert ID outside model")
    if len(set(selected))!=len(selected) or tuple(sorted(selected))!=selected:
        raise ValueError("selected expert IDs must be sorted and unique")
    if not isinstance(state.activation,bytes):
        raise ValueError("activation must be exact bytes")
    size=len(state.activation)
    if size==0 or size%(itemsize*width):
        raise ValueError("invalid activation layout")
    if size*(len(selected)+1)>limit:
        raise ValueError("consumer payload exceeds declared memory limit")
    for e in selected:
        if e not in state.histories or not isinstance(state.histories[e],bytes):
            raise ValueError("missing exact selected expert history")
        if len(state.histories[e])!=size:
            raise ValueError("selected history and activation length differ")
    return size


def _base_digest(baseline:JointState, selected:tuple[int,...])->str:
    return _sha(baseline.activation+b"".join(baseline.histories[e] for e in selected))


def _framed(meta:dict, body:bytes)->bytes:
    header=json.dumps(meta,sort_keys=True,separators=(",",":")).encode()
    if len(header)>MAX_HEADER:raise ValueError("joint packet header exceeds limit")
    return MAGIC+struct.pack("<I",len(header))+header+body


def _unzip(body:bytes, expected:int)->bytes:
    obj=zlib.decompressobj()
    try:
        result=obj.decompress(body,expected+1)
    except zlib.error as exc:
        raise ValueError("invalid compressed residual") from exc
    if (len(result)!=expected or not obj.eof or obj.unused_data or
            obj.unconsumed_tail):
        raise ValueError("invalid or unbounded compressed residual")
    return result


def pack_joint_selected(state:JointState, *, selected:tuple[int,...],
                        expert_count:int,itemsize:int=4,block_width:int=24,
                        model_revision:str,history_revision:str,
                        acknowledged_baseline:JointState|None=None,
                        acknowledged_revision:str|None=None,
                        permit_compression:bool=True,
                        use_site_actions:bool=True)->bytes:
    """Exact quotient for *selected execution*, not general full-E replay."""
    if not model_revision or not history_revision:
        raise ValueError("model and source history revisions required")
    if (acknowledged_baseline is None)!=(acknowledged_revision is None):
        raise ValueError("baseline and its explicit ACK revision must travel together")
    selected=tuple(selected)
    size=_validate(state,selected,expert_count,itemsize,block_width,MAX_UNCOMPRESSED)
    if acknowledged_baseline is not None:
        if not isinstance(acknowledged_revision,str) or not acknowledged_revision:
            raise ValueError("nonempty authenticated ACK revision required")
        _validate(acknowledged_baseline,selected,expert_count,itemsize,
                  block_width,MAX_UNCOMPRESSED)
    complete=state.activation+b"".join(state.histories[e] for e in selected)
    meta=dict(version=1,consumer="execute_selected",
              model=model_revision,history=history_revision,
              ack=acknowledged_revision,selected=list(selected),
              expert_count=expert_count,itemsize=itemsize,
              block_width=block_width,component_bytes=size,
              sha256=_sha(complete),codec="raw",
              site_actions=bool(use_site_actions and block_width%8==0))
    if acknowledged_baseline is not None:
        meta["baseline_sha256"]=_base_digest(acknowledged_baseline,selected)
    packets=[_framed(meta,complete)]
    if not permit_compression:
        return packets[0]
    packets.append(_framed(dict(meta,codec="zlib"),zlib.compress(complete,level=3)))
    # Joint ordered predictive quotient, each history can reuse an earlier
    # *reconstructed* history, a shared activation or its signed orbit.
    # For random histories no compression is promised; raw remains available.
    components=[state.activation]+[state.histories[e] for e in selected]
    zero=b"\0"*size
    recipes=[]
    pieces=[]
    previous={}
    for index,value in enumerate(components):
        candidate={"zero":zero}
        if index==0:
            if acknowledged_baseline is not None:
                candidate["base_x"]=acknowledged_baseline.activation
        else:
            e=selected[index-1]
            candidate["x"]=state.activation
            candidate["neg_x"]=flip_ieee_signs(state.activation,itemsize)
            if meta["site_actions"]:
                for action in ACTIONS:
                    if action=="identity":continue
                    mapped=permute_punctured_nine_bytes(
                        state.activation,itemsize,action)
                    candidate["sheet_x:"+action]=mapped
                    candidate["neg_sheet_x:"+action]=flip_ieee_signs(
                        mapped,itemsize)
            if acknowledged_baseline is not None:
                candidate["base_h"]=acknowledged_baseline.histories[e]
            for previous_id,old in previous.items():
                candidate["prev:"+str(previous_id)]=old
                candidate["neg_prev:"+str(previous_id)]=flip_ieee_signs(old,itemsize)
        options=[]
        for name,predicted in candidate.items():
            diff=_xor(value,predicted)
            options.append((0 if not any(diff) else len(zlib.compress(diff,1)),
                            name,diff))
        _,reference,residual=min(options,key=lambda row:(row[0],row[1]))
        nonzero=bool(any(residual))
        recipes.append({"reference":reference,"residual":nonzero})
        if nonzero:pieces.append(residual)
        if index:previous[selected[index-1]]=value
    residual=b"".join(pieces)
    joint_meta=dict(meta,codec="orbit",recipes=recipes)
    packets.append(_framed(joint_meta,zlib.compress(residual,level=3)
                          if residual else b""))
    return min(packets,key=len)


def unpack_joint_selected(packet:bytes, *, expected_model:str,
                          expected_history:str,expected_selected:tuple[int,...]|None=None,
                          acknowledged_baseline:JointState|None=None,
                          expected_ack_revision:str|None=None,
                          max_uncompressed:int=MAX_UNCOMPRESSED)->JointState:
    if not isinstance(packet,bytes) or len(packet)<8 or packet[:4]!=MAGIC:
        raise ValueError("invalid JQ1 packet")
    nheader=struct.unpack("<I",packet[4:8])[0]
    if nheader>MAX_HEADER or len(packet)<8+nheader:
        raise ValueError("invalid joint header framing")
    try:
        m=json.loads(packet[8:8+nheader])
    except (ValueError,UnicodeDecodeError) as exc:
        raise ValueError("invalid joint metadata") from exc
    if not isinstance(m,dict):
        raise ValueError("invalid joint metadata type")
    if (m.get("version")!=1 or m.get("consumer")!="execute_selected" or
        m.get("model")!=expected_model or m.get("history")!=expected_history):
        raise ValueError("wrong model, history, version or consumer")
    selected=m.get("selected")
    ecount=m.get("expert_count")
    itemsize=m.get("itemsize")
    width=m.get("block_width")
    size=m.get("component_bytes")
    if (not isinstance(selected,list) or not isinstance(ecount,int) or
        not isinstance(size,int) or size<1 or
        not isinstance(itemsize,int) or not isinstance(width,int)):
        raise ValueError("invalid expert/size metadata")
    selected=tuple(selected)
    if expected_selected is not None and selected!=tuple(expected_selected):
        raise ValueError("selected expert IDs disagree with consumer")
    dummy=JointState(b"\0"*size,{e:b"\0"*size for e in selected})
    _validate(dummy,selected,ecount,itemsize,width,max_uncompressed)
    n=(len(selected)+1)*size
    ack=m.get("ack")
    if ack is not None:
        if (not isinstance(ack,str) or ack!=expected_ack_revision or
                acknowledged_baseline is None):
            raise ValueError("missing acknowledged receiver baseline")
        _validate(acknowledged_baseline,selected,ecount,itemsize,width,max_uncompressed)
        if _base_digest(acknowledged_baseline,selected)!=m.get("baseline_sha256"):
            raise ValueError("baseline content does not match certified snapshot")
    elif acknowledged_baseline is not None or expected_ack_revision is not None:
        raise ValueError("unexpected receiver baseline")
    body=packet[8+nheader:]
    codec=m.get("codec")
    if codec=="raw":
        plain=body
    elif codec=="zlib":
        plain=_unzip(body,n)
    elif codec=="orbit":
        recipes=m.get("recipes")
        if not isinstance(recipes,list) or len(recipes)!=len(selected)+1:
            raise ValueError("missing orbit recipe")
        expected_residual=sum(size for recipe in recipes
            if isinstance(recipe,dict) and recipe.get("residual") is True)
        if expected_residual>max_uncompressed:
            raise ValueError("orbit correction exceeds memory limit")
        residual=_unzip(body,expected_residual) if expected_residual else b""
        if not expected_residual and body:
            raise ValueError("unexpected data for zero-residual orbit")
        zero=b"\0"*size
        components=[]
        cursor=0
        for index,recipe in enumerate(recipes):
            if not isinstance(recipe,dict) or type(recipe.get("residual")) is not bool:
                raise ValueError("invalid orbit recipe")
            ref=recipe.get("reference")
            candidate={"zero":zero}
            if index==0:
                if ack is not None:
                    candidate["base_x"]=acknowledged_baseline.activation
            else:
                e=selected[index-1]
                x=components[0]
                candidate["x"]=x
                candidate["neg_x"]=flip_ieee_signs(x,itemsize)
                if m.get("site_actions") is True:
                    if width%8:
                        raise ValueError("site action on non-punctured layout")
                    for action in ACTIONS:
                        if action=="identity":continue
                        mapped=permute_punctured_nine_bytes(x,itemsize,action)
                        candidate["sheet_x:"+action]=mapped
                        candidate["neg_sheet_x:"+action]=flip_ieee_signs(
                            mapped,itemsize)
                if ack is not None:
                    candidate["base_h"]=acknowledged_baseline.histories[e]
                for j,old in enumerate(components[1:]):
                    prev_id=selected[j]
                    candidate["prev:"+str(prev_id)]=old
                    candidate["neg_prev:"+str(prev_id)]=flip_ieee_signs(old,itemsize)
            if ref not in candidate:
                raise ValueError("unavailable or forward orbit reference")
            current=residual[cursor:cursor+size] if recipe["residual"] else zero
            cursor+=size if recipe["residual"] else 0
            components.append(_xor(candidate[ref],current))
        if cursor!=len(residual):
            raise ValueError("orbit residual length mismatch")
        plain=b"".join(components)
    else:
        raise ValueError("unknown joint codec")
    if len(plain)!=n or _sha(plain)!=m.get("sha256"):
        raise ValueError("joint packet failed exact reconstruction")
    return JointState(plain[:size],
                      {e:plain[(i+1)*size:(i+2)*size] for i,e in enumerate(selected)})


def merge_selected_into_authoritative_state(
        authoritative:JointState,reopened:JointState, *,
        selected:tuple[int,...],expert_count:int,
        itemsize:int=4,block_width:int=24)->JointState:
    """Exact selected-only state update, preserving *all* inactive histories.

    The caller is responsible for authenticating the execution receipt and
    verifying its state revision. This does not infer updated histories from
    a router quotient and does not implement cross-peer atomic commit.
    """
    if not isinstance(authoritative,JointState) or not isinstance(reopened,JointState):
        raise ValueError("joint state required")
    ids=tuple(selected)
    if (len(authoritative.histories)!=expert_count or
        set(authoritative.histories)!=set(range(expert_count))):
        raise ValueError("full authoritative history map required")
    _validate(reopened,ids,expert_count,itemsize,block_width,MAX_UNCOMPRESSED)
    if any(len(h)!=len(reopened.activation) for h in authoritative.histories.values()):
        raise ValueError("authoritative histories have mismatched lengths")
    result=dict(authoritative.histories)
    for e in ids:
        result[e]=reopened.histories[e]
    return JointState(reopened.activation,result)
