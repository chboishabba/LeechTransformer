"""Capture actual routed joint activation/expert histories from a trained MoE.

A DENSE best_model.pt checkpoint has no learned expert histories and is
deliberately rejected. First train/export a signed MoE candidate.
Use original token IDs from real corpus, then evaluate JQ1 packet savings:
  python -m scripts.capture_joint_moe_state --checkpoint student.pt \
    --token-ids heldout_ids.npy --layer 0 --output capture_layer0.npz
  python -m scripts.benchmark_joint_state_quotient --capture capture_layer0.npz

These traces are from real model execution, not an inference of semantic
opposition or evidence of remote ACK. No history baselines are fabricated.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from models.model import LeechGPT
from scripts.verify_trained_federated_moe import checkpoint_parts,infer_config


@torch.no_grad()
def capture_from_expert(moe, hidden, initial=None):
    if hidden.ndim!=3 or hidden.shape[2]!=moe.d_model:
        raise ValueError("hidden activations must be [B,T,d]")
    b,t,d=hidden.shape
    state=(initial.clone() if initial is not None else
           hidden.new_zeros(b,moe.num_experts,d))
    activations=[];histories=[];routes=[];outputs=[]
    for index in range(t):
        before=state.clone()
        output,state,ids,_,_=moe.step(hidden[:,index],state)
        activations.append(hidden[:,index].detach().cpu())
        histories.append(before.detach().cpu())
        routes.append(ids.detach().cpu())
        outputs.append(output.detach().cpu())
    return {
        "activation":torch.stack(activations,1).reshape(b*t,d).float().numpy(),
        "histories":torch.stack(histories,1).reshape(
            b*t,moe.num_experts,d).float().numpy(),
        "selected":torch.stack(routes,1).reshape(b*t,moe.top_k).sort(-1).values.numpy(),
        "replayed_output":torch.stack(outputs,1).reshape(b*t,d).float().numpy(),
        "replayed_final_history":state.detach().cpu().float().numpy(),
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--checkpoint",type=Path,required=True)
    ap.add_argument("--config",type=Path)
    ap.add_argument("--token-ids",type=Path,required=True)
    ap.add_argument("--layer",type=int,default=0)
    ap.add_argument("--batch",type=int,default=2)
    ap.add_argument("--device",default="cpu")
    ap.add_argument("--output",type=Path,default=Path("joint_moe_capture.npz"))
    args=ap.parse_args()
    state,saved=checkpoint_parts(args.checkpoint)
    cfg=infer_config(state,saved,args.config)
    if cfg.moe_experts<2:
        raise ValueError("checkpoint is dense; no trained MoE histories to quotient")
    if not 0<=args.layer<cfg.n_layers or args.batch<1:
        raise ValueError("invalid layer or batch")
    if args.token_ids.suffix!=".npy":
        raise ValueError("provide one numeric token-ID .npy array")
    tokens=np.load(args.token_ids,allow_pickle=False)
    if (tokens.ndim!=2 or tokens.shape[0]<1 or tokens.shape[1]<1 or
        tokens.shape[1]>cfg.block_size or
        not np.issubdtype(tokens.dtype,np.integer) or
        np.any(tokens<0) or np.any(tokens>=cfg.vocab_size)):
        raise ValueError("token IDs out of range or bad shape")
    model=LeechGPT(cfg)
    model.load_state_dict(state,strict=True)
    model.to(args.device).eval()
    records=[]
    observed=0
    with torch.no_grad():
        for start in range(0,len(tokens),args.batch):
            captured=[]
            module=model.blocks[args.layer].moe
            def hook(mod,inputs):
                captured.append((
                    inputs[0].detach().clone(),
                    inputs[1].detach().clone() if len(inputs)>1 and
                    inputs[1] is not None else None))
            h=module.register_forward_pre_hook(hook)
            try:
                batch=torch.from_numpy(np.asarray(
                    tokens[start:start+args.batch],dtype=np.int64)).to(args.device)
                forward=model(batch,return_expert_histories=True)
            finally:
                h.remove()
            if len(captured)!=1:
                raise RuntimeError("expected exactly one MoE call per layer")
            source,initial=captured[0]
            one=capture_from_expert(module,source,initial)
            # Check the captured sequential execution against the actual
            # layer's final histories before saving the evidence.
            final=forward[4][args.layer].detach().cpu().float().numpy()
            if not np.allclose(one["replayed_final_history"],final,
                               atol=2e-5,rtol=1e-4):
                raise AssertionError("captured replay disagrees with model forward")
            records.append(one)
            observed+=batch.shape[0]
    result={name:np.concatenate([part[name] for part in records],axis=0)
            for name in ("activation","histories","selected")}
    # Source-model-known action chart, never inferred from a compression
    # match. This is the trusted orbit law for selected-only receivers.
    if getattr(cfg,"moe_shared_e8_history",False):
        result["orbit_actions"]=np.asarray(
            model.blocks[args.layer].moe.action_names,dtype="<U24")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output,**result)
    print(json.dumps({
        "capture_file":str(args.output),"source_checkpoint":str(args.checkpoint),
        "samples":int(result["activation"].shape[0]),
        "source_sequences":observed,"layer":args.layer,
        "model_experts":cfg.moe_experts,"top_k":cfg.moe_top_k,
        "route_geometry":cfg.moe_route_geometry,
        "shared_e8_canonical_history":getattr(cfg,"moe_shared_e8_history",False),
        "source":"strictly loaded trained model + replay-verified MoE forward",
        "ack_history":"none fabricated: capture holds time-indexed source histories"
    },indent=2))


if __name__=="__main__":
    main()
