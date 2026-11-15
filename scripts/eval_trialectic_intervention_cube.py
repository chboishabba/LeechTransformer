"""Source-labelled 2x2x2 trialectic intervention cube evaluation.

Eight conditions are ordered by binary coordinates (x,y,z), index
4*x + 2*y + z, from 000 to 111. A single FIXED probe token per cube
measures the genuine third-order discrete interaction in log-probability.
Separate independently annotated target tokens evaluate correctness at
each corner. Without a fixed probe target, mixing eight target scores
would not be a valid third-order difference.

This is an experimental response interaction, not proof of Sophontic
semantics, causal independence, E8 actions, or Leech gluing.
Original Leech-LILA by Anatolii Kornienko (DOI 10.5281/zenodo.18784424).
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
from scripts.verify_trained_federated_moe import checkpoint_parts, infer_config

# Third-order finite difference f111 - f110 - f101 - f011
# + f100 + f010 + f001 - f000, for index 4*x+2*y+z.
FACE_COEFFICIENTS=(-1.,1.,1.,-1.,1.,-1.,-1.,1.)


@torch.no_grad()
def cube_evidence(logits: torch.Tensor, target_tokens: torch.Tensor,
                  fixed_probe: torch.Tensor) -> dict:
    if logits.ndim!=3 or logits.shape[1]!=8 or logits.shape[0]<1:
        raise ValueError("logits must be [N,8,vocab]")
    n,_,v=logits.shape
    if (target_tokens.shape!=(n,8) or fixed_probe.shape!=(n,) or
        target_tokens.dtype!=torch.long or fixed_probe.dtype!=torch.long or
        bool((target_tokens<0).any()) or bool((target_tokens>=v).any()) or
        bool((fixed_probe<0).any()) or bool((fixed_probe>=v).any()) or
        not bool(torch.isfinite(logits).all())):
        raise ValueError("invalid independent targets, fixed probe or logits")
    correct=logits.argmax(-1)==target_tokens
    logp=F.log_softmax(logits.float(),dim=-1)
    fixed=logp.gather(
        -1,fixed_probe[:,None,None].expand(n,8,1)).squeeze(-1)
    sign=logp.new_tensor(FACE_COEFFICIENTS)
    face=(fixed*sign[None,:]).sum(-1)
    target_logp=logp.gather(-1,target_tokens[:,:,None]).squeeze(-1)
    return {
        "cubes":n,
        "corner_correct_rate":float(correct.float().mean()),
        "entire_cube_correct_rate":float(correct.all(-1).float().mean()),
        "corner_correct_rates":correct.float().mean(0).tolist(),
        "mean_target_nll_nats":float(-target_logp.mean()),
        "third_order_fixed_probe_mean_nats":float(face.mean()),
        "third_order_fixed_probe_mean_abs_nats":float(face.abs().mean()),
        "third_order_fixed_probe_max_abs_nats":float(face.abs().max()),
        "third_order_nonzero_fraction_above_1e_5":
            float((face.abs()>1e-5).float().mean()),
        "cube_order":"000,001,010,011,100,101,110,111",
        "face_coefficients":list(FACE_COEFFICIENTS),
        "probe_semantics":"same independently supplied probe token across each eight-corner cube",
        "scope":"Third-order model-output interaction and separately scored target correctness. No identified cause, semantic trialectic theorem, or lattice equivalence."}


@torch.no_grad()
def evaluate_model(model: LeechGPT, cube_tokens: np.ndarray, *,
                   batch: int = 2, device: str = "cpu") -> torch.Tensor:
    if batch<1:
        raise ValueError("batch must be positive")
    n,_,t=cube_tokens.shape
    flat=cube_tokens.reshape(n*8,t)
    model.to(device).eval()
    logits=[]
    for start in range(0,len(flat),batch):
        tokens=torch.from_numpy(np.ascontiguousarray(
            flat[start:start+batch],dtype=np.int64)).to(device)
        logits.append(model(tokens)[0][:,-1,:].detach().cpu())
    return torch.cat(logits,0).reshape(n,8,-1)


def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint",required=True,type=Path)
    p.add_argument("--config",type=Path)
    p.add_argument("--interventions",required=True,type=Path,
                   help="NPZ with tokens [N,8,T], targets [N,8], fixed_probe [N]")
    p.add_argument("--batch",type=int,default=2)
    p.add_argument("--device",default="cpu")
    p.add_argument("--output",required=True,type=Path)
    args=p.parse_args()
    weights,meta=checkpoint_parts(args.checkpoint)
    cfg=infer_config(weights,meta,args.config)
    model=LeechGPT(cfg)
    model.load_state_dict(weights,strict=True)
    with np.load(args.interventions,allow_pickle=False) as file:
        if not {"tokens","targets","fixed_probe"}.issubset(file.files):
            raise ValueError("missing independently annotated 2x2x2 cube fields")
        tokens=file["tokens"];targets=file["targets"];probe=file["fixed_probe"]
    if (tokens.ndim!=3 or tokens.shape[1]!=8 or tokens.shape[0]<1 or
        not 1<=tokens.shape[2]<=cfg.block_size or
        targets.shape!=(tokens.shape[0],8) or
        probe.shape!=(tokens.shape[0],)):
        raise ValueError("invalid intervention cube dimensions")
    for array in (tokens,targets,probe):
        if (not np.issubdtype(array.dtype,np.integer) or
            np.any(array<0) or np.any(array>=cfg.vocab_size)):
            raise ValueError("cube integer token IDs out of vocabulary")
    results=cube_evidence(evaluate_model(model,tokens,batch=args.batch,
                         device=args.device),
        torch.as_tensor(targets,dtype=torch.long),
        torch.as_tensor(probe,dtype=torch.long))
    digest=hashlib.sha256()
    with args.interventions.open("rb") as stream:
        for chunk in iter(lambda:stream.read(1<<20),b""):
            digest.update(chunk)
    results.update({
        "checkpoint":str(args.checkpoint),
        "source_interventions_sha256":digest.hexdigest(),
        "route_geometry":cfg.moe_route_geometry,
        "independent_labels_required":True,
        "no_synthetic_fallback":True})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(results,indent=2))
    print(json.dumps(results,indent=2))


if __name__=="__main__":
    main()
