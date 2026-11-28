"""Observe exact ORIGINAL fitted-R4 source activation and all-E histories.

This module does NOT reconstruct source weights. The caller must first use
the existing strict source C2+R4 delta restoration owner. It then invokes
the original LeechGPT forward on a validated native SLR DEVELOPMENT record
while temporarily observing exactly one selected HistoryOrbitMoE.step.

No source code bytes or checkpoint weights are altered. The same captured
input must replay BITWISE through an independently reconstructed MoE
layer before any probe input is emitted. Reserved F5 is never opened.
"""
import hashlib
import json
import math

import torch

from models.history_orbit_moe import HistoryOrbitMoE
from scripts.x369_r4_execution_device import runtime, tensor_bytes, bitwise_equal
from scripts.run_x369_r4_three_arm_source_maxcut import (
    validated_r1_summary, validate_source_docs)
from scripts.audit_x369_slr_spacy_grammar_early_epoch import load_docs


def capture_original_slr_development_probe(restored,moe,module_cfg,dtype,
        *,layer_index,native_slr_jsonl,native_sha,r1_summary,tokenizer,
        document_id,token_position,load_levels,response_kind,
        response_index,nat_scale):
    import sentencepiece as spm
    sha=lambda data:hashlib.sha256(data).hexdigest()
    tokenizer_sha=sha(tokenizer.read_bytes())
    r1=json.loads(r1_summary.read_text(encoding="utf8"))
    validated_r1_summary(r1,jsonl_sha=native_sha,
                         tokenizer_sha=tokenizer_sha)
    sp=spm.SentencePieceProcessor(model_file=str(tokenizer))
    records=load_docs(native_slr_jsonl,sp,tokenizer_sha)
    _,development=validate_source_docs(records,restored.cfg)
    chosen=[rec for rec in development if str(rec["id"])==document_id]
    if len(chosen)!=1:
        raise ValueError("original source development document not unique")
    record=chosen[0]
    token_ids=record["token_ids"]
    if (type(token_position) is not int or
        not 0<=token_position<len(token_ids)):
        raise ValueError("original selected source token position invalid")
    if (not 2<=len(token_ids)<=restored.cfg.block_size
        or any(type(z) is not int or not 0<=z<restored.cfg.vocab_size
               for z in token_ids)):
        raise ValueError("original SLR token IDs invalid")
    device=next(restored.parameters()).device
    execution_runtime=runtime(device)
    if next(moe.parameters()).device!=device:
        raise ValueError("original source and selected layer device mismatch")
    if restored.training or moe.training:
        raise ValueError("original source capture requires eval mode")
    observed=[]
    count=0
    original_step=moe.step
    def observe_step(x,h,train_sign=False):
        nonlocal count
        current=count
        count+=1
        chosen_step=current==token_position
        if chosen_step:
            if (tuple(x.shape)!=(1,moe.d_model) or
                tuple(h.shape)!=(1,moe.num_experts,moe.d_model)):
                raise RuntimeError("original source state tensor shape wrong")
            source_x=x.detach().clone()
            source_history=h.detach().clone()
        result=original_step(x,h,train_sign=train_sign)
        if chosen_step:
            observed.append((source_x,source_history,
                 tuple(v.detach().clone() for v in result)))
        return result
    # Runtime-only method shadowing; source Py files and learned tensors
    # remain byte-exact for R4 delta geometry fingerprint acceptance.
    moe.step=observe_step
    try:
        with torch.no_grad():
            restored(torch.tensor([token_ids],dtype=torch.long,device=device),
                     student_only=True,return_expert_histories=True)
    finally:
        del moe.step
    if len(observed)!=1 or count!=len(token_ids):
        raise RuntimeError("source trace did not cover original full document")
    x,h,step_result=observed[0]
    if not bool(torch.isfinite(x).all() and torch.isfinite(h).all()):
        raise ValueError("captured original x/H not finite")
    independently_restored=HistoryOrbitMoE(**module_cfg).to(device=device,dtype=dtype).eval()
    independently_restored.history_cap=moe.history_cap
    independently_restored.load_state_dict(moe.state_dict(),strict=True)
    with torch.no_grad():
        replay=independently_restored.step(x,h,train_sign=False)
    if (len(replay)!=len(step_result) or
        any(not bitwise_equal(a,b) for a,b in zip(replay,step_result))):
        raise RuntimeError("original full GPT source step and exact MoE replay differ")
    try:
        loads=[float(v) for v in load_levels.split(",")]
    except (TypeError,ValueError,OverflowError) as e:
        raise ValueError("invalid original four-load selection") from e
    if (len(loads)!=4 or any(not math.isfinite(v) for v in loads)
        or not all(u<v for u,v in zip(loads,loads[1:]))
        or 1. not in loads):
        raise ValueError("four increasing loads must include the original load 1")
    if type(nat_scale) is not int or not 1<=nat_scale<=10**9:
        raise ValueError("invalid original Nat response scale")
    if response_kind=="coordinate":
        if type(response_index) is not int or not 0<=response_index<moe.d_model:
            raise ValueError("original response coordinate invalid")
        response={"type":"coordinate","index":response_index,"scale":nat_scale}
    elif response_kind=="l2":
        response={"type":"l2","scale":nat_scale}
    else:
        raise ValueError("unsupported source observable")
    history_hash=sha(tensor_bytes(h))
    activation_hash=sha(tensor_bytes(x))
    return {
       "model":module_cfg,
       "source_execution_runtime":execution_runtime,
       "base_x":[repr(float(v)) for v in x[0].tolist()],
       "history":[[repr(float(v)) for v in row]
                  for row in h[0].tolist()],
       "load_levels":[repr(v) for v in loads],
       "response":response,
       "dtype":str(dtype).split(".")[-1],
       "history_cap":moe.history_cap,
       "history_revision":"native-SLR:"+str(record["id"])+
           ":layer"+str(layer_index)+":token"+str(token_position)+
           ":"+history_hash,
       "source_document_id":record["id"],
       "source_document_sha256":record["source_sha"],
       "source_record_split":"development",
       "source_native_slr_jsonl_sha256":native_sha,
       "source_token_position":token_position,
       "original_observed_load":1,
       "source_history_tensor_sha256":history_hash,
       "source_activation_tensor_sha256":activation_hash,
       "source_step_and_standalone_MoE_bitwise_replay_verified":True,
       "fitted_kink_certified_by_this_export":False,
       "float_evaluation_error_intervals_certified":False,
       "F5_or_reserved_data_read":False
    }
