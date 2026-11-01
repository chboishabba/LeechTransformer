"""Distil real dense Leech-LILA FFNs into signed history experts.

The trained dense checkpoint is the teacher, NEVER reinitialised.
Training objective is an exact odd-parity decomposition of its FFNs,
not a claim to improve semantic reasoning without paired evaluation.

Example:
 python -m scripts.train_signed_wave_experts \
  --checkpoint data/best_model.pt --config leech_dense_config.json \
  --token-ids data/heldout_token_ids.npy --layer 0 --epochs 3 \
  --device cpu --output signed_expert_layer0.pt

The .npy file must contain token IDs, N by T, numeric dtype, no pickle.
Specify --synthetic-smoke to test plumbing only; output will be labelled
synthetic, never evidence of real-data learning.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from models.model import LeechGPT
from models.dense_to_signed_experts import DenseToSignedExperts
from models.trialectic_wave_prior import geometry_shape,phase_rotation_residual
from scripts.verify_trained_federated_moe import checkpoint_parts,infer_config


def load_tokens(path, *, n_synthetic, block_size, vocab_size, seed):
    if path is None:
        if n_synthetic <= 0:
            raise ValueError("--token-ids required unless --synthetic-smoke is explicit")
        rng=np.random.default_rng(seed)
        return torch.from_numpy(rng.integers(0,vocab_size,size=(n_synthetic,min(8,block_size)),dtype=np.int64)),True
    arr=np.load(path,allow_pickle=False)
    if arr.ndim != 2 or arr.shape[0]<2 or not 1 <= arr.shape[1] <= block_size:
        raise ValueError("token_ids must be [N>=2,1<=time<=block_size]")
    if not np.issubdtype(arr.dtype,np.integer) or np.any(arr<0) or np.any(arr>=vocab_size):
        raise ValueError("token IDs out of vocabulary or not integer")
    return torch.from_numpy(np.ascontiguousarray(arr,dtype=np.int64)),False


@torch.no_grad()
def layer_inputs(model,token_ids,layer):
    outputs=[]
    hook=model.blocks[layer].ln2.register_forward_hook(
        lambda module,inputs,value: outputs.append(value.detach()))
    try:
        model.eval()
        model(token_ids)
    finally:
        hook.remove()
    if len(outputs)!=1:
        raise RuntimeError("expected exactly one normalized hidden-state hook")
    return outputs[0]


def run_layer(model,tokens,layer,*,epochs,batch,lr,num_experts,top_k,
              expert_width,transition_rank,route_geometry,seed):
    if model.blocks[layer].ffn is None:
        raise ValueError("teacher layer must be a trained dense FFN")
    torch.manual_seed(seed+layer)
    d=model.cfg.d_model
    adapter=DenseToSignedExperts(model.blocks[layer].ffn,d,
        num_experts,top_k,expert_width,route_geometry,transition_rank)
    adapter.to(next(model.parameters()).device)
    optimizer=torch.optim.AdamW(
        list(adapter.student.parameters())+list(adapter.even_student.parameters()),lr=lr)
    n=tokens.shape[0]
    train_n=max(1,int(.8*n))
    if train_n==n: train_n=n-1
    history=[]
    for epoch in range(epochs):
        adapter.train()
        order=torch.randperm(train_n,generator=torch.Generator().manual_seed(seed+epoch))
        losses=[]
        for start in range(0,train_n,batch):
            idx=order[start:start+batch]
            hidden=layer_inputs(model,tokens[idx].to(next(model.parameters()).device),layer)
            optimizer.zero_grad(set_to_none=True)
            record=adapter(hidden,blend=0.)
            loss=record["student_distillation_loss"]
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        adapter.eval()
        with torch.no_grad():
            validation=[]
            for start in range(train_n,n,batch):
                ids=tokens[start:start+batch].to(next(model.parameters()).device)
                x=layer_inputs(model,ids,layer)
                validation.append(float(adapter(x,blend=1.)["student_distillation_loss"]))
        history.append({"epoch":epoch+1,"train_mse":sum(losses)/len(losses),
                        "heldout_parity_mse":sum(validation)/len(validation)})
    adapter.eval()
    with torch.no_grad():
        x=layer_inputs(model,tokens[train_n:train_n+1].to(next(model.parameters()).device),layer)
        h=torch.zeros(1,num_experts,d,device=x.device)
        pos=adapter.student(x,h)
        neg=adapter.student(-x,-h)
        sign_output_max=float((pos[0]+neg[0]).abs().max())
        sign_history_max=float((pos[1]+neg[1]).abs().max())
        ids_identical=bool(torch.equal(pos[2],neg[2]))
        weights_identical=bool(torch.equal(pos[4],neg[4]))
        recon=adapter(x,blend=0.)
        dense_transfer_max=float((recon["output"]-recon["reference"]).abs().max())
        wave3_max=float(phase_rotation_residual(x).abs().max())
        deployed=adapter.predict_only(x,h)
        correction_max=float((recon["reference"]-deployed["output"]).abs().max())
    return adapter,{"layer":layer,"epoch_receipts":history,
         "sign_output_max":sign_output_max,
         "sign_history_max":sign_history_max,
         "route_inversion_identical":ids_identical,
         "weight_inversion_identical":weights_identical,
         "dense_blend_zero_max":dense_transfer_max,
         "deployed_student_residual_max":correction_max,
         "c3_phase_rotation_residual_max":wave3_max,
         "student_parameters":sum(p.numel() for p in adapter.student.parameters()),
         "even_student_parameters":sum(p.numel() for p in adapter.even_student.parameters()),
         "teacher_parameters":sum(p.numel() for p in adapter.teacher.parameters()),
         "face_trained":False,
         "semantic_paired_training":False}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint",required=True)
    p.add_argument("--config",required=True)
    p.add_argument("--token-ids",type=Path)
    p.add_argument("--synthetic-smoke",type=int,default=0,metavar="N")
    p.add_argument("--layer",type=int,default=0,help="-1 trains all 12 layers")
    p.add_argument("--epochs",type=int,default=3)
    p.add_argument("--batch",type=int,default=2)
    p.add_argument("--lr",type=float,default=1e-3)
    p.add_argument("--experts",type=int,default=4)
    p.add_argument("--top-k",type=int,default=2)
    p.add_argument("--expert-width",type=int,default=192)
    p.add_argument("--transition-rank",type=int,default=24)
    p.add_argument("--route-geometry",choices=("c2_invariant","c2c3_wave","punctured9_wave","e8_root_invariant"),default="punctured9_wave")
    p.add_argument("--device",default="cpu")
    p.add_argument("--seed",type=int,default=41)
    p.add_argument("--output",type=Path,default=Path("signed_wave_experts.pt"))
    args=p.parse_args()
    if args.epochs<1 or args.batch<1 or args.top_k<1 or args.top_k>args.experts:
        raise ValueError("invalid training config")
    state,saved=checkpoint_parts(args.checkpoint)
    cfg=infer_config(state,saved,args.config)
    if cfg.moe_experts:
        raise ValueError("teacher checkpoint must be dense; do not overwrite a trained MoE")
    model=LeechGPT(cfg)
    model.load_state_dict(state,strict=True)
    model.to(args.device).eval()
    for parameter in model.parameters():parameter.requires_grad_(False)
    tokens,synthetic=load_tokens(args.token_ids,n_synthetic=args.synthetic_smoke,
        block_size=cfg.block_size,vocab_size=cfg.vocab_size,seed=args.seed)
    layers=list(range(cfg.n_layers)) if args.layer==-1 else [args.layer]
    if any(i<0 or i>=cfg.n_layers for i in layers): raise ValueError("invalid layer index")
    records=[]; weights={}
    for i in layers:
        adapter,record=run_layer(model,tokens,i,epochs=args.epochs,batch=args.batch,lr=args.lr,
            num_experts=args.experts,top_k=args.top_k,expert_width=args.expert_width,
            transition_rank=args.transition_rank,route_geometry=args.route_geometry,seed=args.seed)
        weights[str(i)]={"odd_experts":adapter.student.state_dict(),
                         "even_student":adapter.even_student.state_dict(),
                         "face":adapter.face.state_dict()}
        records.append(record)
    receipt={
        "architecture":geometry_shape(cfg.d_model,cfg.n_heads,cfg.n_layers),
        "source_checkpoint":str(Path(args.checkpoint).resolve()),
        "synthetic_smoke_only":synthetic,
        "training_rows":int(tokens.shape[0]),
        "layers":records,
        "expert_configuration":{"num_experts":args.experts,"top_k":args.top_k,
            "expert_width":args.expert_width,"transition_rank":args.transition_rank,
            "route_geometry":args.route_geometry},
        "certification_limits":"Dense-FFN even+odd distillation, not language improvement, Sophontic paired reasoning, actual Leech automorphism or WAN speed."
    }
    args.output.parent.mkdir(parents=True,exist_ok=True)
    torch.save({"student_state_dicts":weights,"receipt":receipt},args.output)
    args.output.with_suffix(".json").write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt,indent=2))

if __name__=="__main__":
    main()
