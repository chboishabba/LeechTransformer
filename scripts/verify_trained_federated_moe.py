"""Run the actual saved LeechGPT weights through the federated/geometry gates.

Usage:
 python -m scripts.verify_trained_federated_moe --checkpoint /path/to/model.pt
 python -m scripts.verify_trained_federated_moe --checkpoint model.pt --config cfg.json --device cuda
No synthetic checkpoint is silently substituted; a non-MoE checkpoint is
reported as dense and cannot certify trained expert specialization.
Only torch.load(weights_only=True) is used. JSON emitted for archival receipts.
"""
import argparse
import __main__
import dataclasses
import hashlib
import json
import math
import pickle
import time
from pathlib import Path

import torch
from torch.serialization import add_safe_globals
from models.model import LeechConfig, LeechGPT
from scripts.federated_activation_transport import pack_activation, unpack_activation

_MISSING = object()


def _load_checkpoint_weights_only(path):
    """Load a tensor checkpoint without enabling arbitrary pickle execution.

    Older local Leech checkpoints serialize their configuration as
    ``__main__.LeechConfig``.  PyTorch's safe loader quite properly rejects
    that unregistered class.  It carries configuration fields only, so retry
    after allowlisting a deliberately empty class with precisely that name;
    all other unsupported globals remain rejected.
    """
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError as error:
        if "__main__.LeechConfig" not in str(error):
            raise

    previous = getattr(__main__, "LeechConfig", _MISSING)
    compatibility_config = type("LeechConfig", (), {})
    compatibility_config.__module__ = "__main__"
    __main__.LeechConfig = compatibility_config
    add_safe_globals([compatibility_config])
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    finally:
        if previous is _MISSING:
            delattr(__main__, "LeechConfig")
        else:
            __main__.LeechConfig = previous


def checkpoint_parts(path):
    obj = _load_checkpoint_weights_only(path)
    if not isinstance(obj,dict): raise ValueError("checkpoint must be a state dict or dict containing one")
    candidates=("model_state_dict","state_dict","model","model_state")
    state=next((obj[key] for key in candidates if key in obj and isinstance(obj[key],dict)),obj)
    if not state or not all(isinstance(k,str) and isinstance(v,torch.Tensor) for k,v in state.items()):
        raise ValueError("checkpoint has no plain tensor state_dict")
    if any(k.startswith("module.") for k in state):
        if not all(k.startswith("module.") for k in state):
            raise ValueError("mixed DDP key prefixes")
        state={k[7:]:v for k,v in state.items()}
    cfg=obj.get("config",obj.get("cfg",{}))
    return state, cfg if isinstance(cfg,dict) else {}

def infer_config(state, saved, config_file):
    extra=json.loads(Path(config_file).read_text()) if config_file else {}
    if not isinstance(extra,dict): raise ValueError("config must be JSON object")
    inp=state["tok_emb.weight"]
    if inp.ndim!=2: raise ValueError("bad embedding tensor")
    vocab,d=inp.shape
    layer_ids={int(k.split(".")[1]) for k in state if k.startswith("blocks.") and k.split(".")[1].isdigit()}
    if not layer_ids or layer_ids != set(range(max(layer_ids)+1)):
        raise ValueError("missing/irregular model layers")
    moe_ids = {int(k.split(".")[4]) for k in state
               if k.startswith("blocks.0.moe.experts.") and k.split(".")[4].isdigit()}
    moe_count=max(moe_ids)+1 if moe_ids else 0
    if moe_count and "moe_top_k" not in saved and "moe_top_k" not in extra:
        raise ValueError("checkpoint does not encode top-k: specify --config with moe_top_k")
    if "n_heads" not in saved and "n_heads" not in extra:
        raise ValueError("checkpoint does not encode attention head count: specify --config with n_heads")
    n_heads = extra.get("n_heads", saved.get("n_heads", 8))
    if d%n_heads!=0 or (d//n_heads)%24!=0:
        # The checkpoint does not store head count; fail rather than guess wrong.
        raise ValueError("n_heads must be supplied through --config for this checkpoint")
    width=(state["blocks.0.moe.experts.0.0.weight"].shape[0] if moe_count else None)
    transition_key="blocks.0.moe.transitions.0.0.weight"
    transition_rank=(int(state[transition_key].shape[0]) if transition_key in state else None)
    cfg=dict(vocab_size=vocab,d_model=d,n_layers=len(layer_ids),n_heads=n_heads,
             block_size=state["pos_emb"].shape[1],
             dropout=0.,lambda_geo=0.,moe_experts=moe_count,
             moe_top_k=saved.get("moe_top_k",2) if moe_count else 2,
             moe_expert_width=width,moe_transition_rank=transition_rank)
    cfg.update({k:v for k,v in saved.items() if k in LeechConfig.__dataclass_fields__})
    cfg.update(extra)
    for key,val in dict(vocab_size=vocab,d_model=d,n_layers=len(layer_ids),
                        block_size=state["pos_emb"].shape[1],moe_experts=moe_count,
                        moe_transition_rank=transition_rank).items():
        if cfg[key]!=val: raise ValueError("configuration contradicts checkpoint: "+key)
    if moe_count and (not 1 <= cfg["moe_top_k"] <= moe_count):
        raise ValueError("explicit --config moe_top_k is required")
    return LeechConfig(**cfg)

def tensor_error(a,b):
    difference=(a.detach().float()-b.detach().float()).abs()
    return {"max_abs":float(difference.max()),"rmse":float(difference.square().mean().sqrt()),
            "all_finite":bool(torch.isfinite(a).all() and torch.isfinite(b).all())}

@torch.no_grad()
def evaluate(model,tokens,gbps=(1,10),rtt_ms=5.,atol=2e-4,rtol=2e-4):
    model.eval()
    device=tokens.device
    start=time.perf_counter()
    full=model(tokens,return_expert_histories=True)
    full_s=time.perf_counter()-start
    kv=None; history=None; generated=[]
    start=time.perf_counter()
    for index in range(tokens.shape[1]):
        row=model(tokens[:,index:index+1],use_cache=True,past_key_values=kv,
                  past_expert_histories=history,return_expert_histories=True)
        generated.append(row[0])
        kv,history=row[3],row[4]
    incremental_s=time.perf_counter()-start
    joined=torch.cat(generated,dim=1)
    errors=tensor_error(joined,full[0])
    equal=bool(torch.allclose(joined,full[0],atol=atol,rtol=rtol))
    history_errors=[]
    if history is not None:
        for a,b in zip(history,full[4]):
            history_errors.append(tensor_error(a,b))
            equal &= bool(torch.allclose(a,b,atol=atol,rtol=rtol))
    # Capture actual learned representation rather than random synthetic codec bytes.
    last=full[1].detach().contiguous().to("cpu",dtype=torch.float32)
    raw=last.numpy().tobytes()
    if len(raw)%96: raise AssertionError("model dimension must be multiple of 24")
    encoded={}
    for compressed in (False,True):
        label="adaptive" if compressed else "raw"
        t0=time.perf_counter()
        pkt=pack_activation(raw,dtype_size=4,block_width=24,
             model_revision="local-checkpoint",history_revision="full-sequence",
             permit_compression=compressed)
        encode_ms=1000*(time.perf_counter()-t0)
        t0=time.perf_counter()
        restored=unpack_activation(pkt,expected_model="local-checkpoint",
                                   expected_history="full-sequence")
        decode_ms=1000*(time.perf_counter()-t0)
        if raw!=restored: raise AssertionError("exact codec byte mismatch")
        encoded[label]={"wire_bytes":len(pkt),"encode_ms":encode_ms,
                        "decode_ms":decode_ms,"sha256":hashlib.sha256(restored).hexdigest()}
    modeled={}
    for speed in gbps:
        rows={}
        for mode,entry in encoded.items():
            rows[mode]={"one_way_serialization_ms":entry["wire_bytes"]*8/(speed*1e9)*1e3,
                        "estimated_remote_latency_ms":rtt_ms+entry["wire_bytes"]*16/(speed*1e9)*1e3+
                             entry["encode_ms"]+entry["decode_ms"]}
        modeled[str(speed)]=rows
    # A generic trained router is NOT presumed inversion-equivariant.
    geometry=[]
    for i,block in enumerate(model.blocks):
        if block.moe is None: continue
        x=torch.randn(2,model.cfg.d_model,device=device)
        h=torch.randn(2,block.moe.num_experts,model.cfg.d_model,device=device)
        positive=block.moe.step(x,h)
        negative=block.moe.step(-x,-h)
        geometry.append({"layer":i,"same_route_rate":float((positive[2]==negative[2]).float().mean()),
            "same_weight_max_abs":tensor_error(positive[4],negative[4])["max_abs"],
            "inversion_output_max_abs":tensor_error(positive[0],-negative[0])["max_abs"],
            "note":"diagnostic only; semantic opposition is not geometric inversion"})
    return {"trained_weights_loaded":True,"moe_enabled":model.cfg.moe_experts>=2,
            "cache_parity_pass":equal,"full_vs_cache_logits":errors,
            "final_history_parity":history_errors,"full_forward_ms":1000*full_s,
            "cached_forward_ms":1000*incremental_s,
            "activation_bytes":len(raw),"wire":encoded,"link_arithmetic":modeled,
            "geometry_diagnostics":geometry,
            "security":"local byte reconstruction only; authenticated remote protocol not executed",
            "semantic":"no Sophontic labels or antipode identification inferred"}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint",required=True)
    p.add_argument("--config",help="JSON with n_heads, moe_top_k or other original saved config")
    p.add_argument("--device",default="cpu")
    p.add_argument("--tokens",type=int,default=8)
    p.add_argument("--batch",type=int,default=2)
    p.add_argument("--seed",type=int,default=41)
    p.add_argument("--rtt-ms",type=float,default=5)
    p.add_argument("--output",default="trained_federated_moe_receipt.json")
    a=p.parse_args()
    torch.manual_seed(a.seed)
    state,saved=checkpoint_parts(a.checkpoint)
    cfg=infer_config(state,saved,a.config)
    model=LeechGPT(cfg)
    model.load_state_dict(state,strict=True)
    model=model.to(a.device).eval()
    if a.tokens<1 or a.batch<1 or a.tokens>cfg.block_size:
        raise ValueError("batch/tokens out of bounds")
    tokens=torch.randint(0,cfg.vocab_size,(a.batch,a.tokens),device=a.device)
    receipt=evaluate(model,tokens,rtt_ms=a.rtt_ms)
    receipt["config"]=dataclasses.asdict(cfg)
    receipt["checkpoint_path"]=str(Path(a.checkpoint).resolve())
    receipt["checkpoint_size_bytes"]=Path(a.checkpoint).stat().st_size
    receipt["cache_parity_thresholds"]={"atol":2e-4,"rtol":2e-4}
    Path(a.output).write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt,indent=2))
    if not receipt["cache_parity_pass"]: raise SystemExit(2)
if __name__=="__main__":main()
