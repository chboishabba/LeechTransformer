"""Replace dense FFN blocks by trained parity students and verify the full model.

Requires signed-wave adapters trained for ALL layers. This is a
candidate model converter: no claim of language parity or geometric
quality unless held-out logits/loss meet the declared acceptance bounds.
No face channel is inferred from absent semantic face annotations.
"""
from __future__ import annotations
import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from models.model import LeechGPT
from scripts.verify_trained_federated_moe import checkpoint_parts,infer_config


def tensor_sha256(path: Path):
    digest=hashlib.sha256()
    with path.open("rb") as file:
        while chunk:=file.read(4*1024*1024):digest.update(chunk)
    return digest.hexdigest()


def convert(dense_checkpoint,config_json,adapters_file, *,
            allow_synthetic=False):
    source,saved=checkpoint_parts(dense_checkpoint)
    cfg=infer_config(source,saved,config_json)
    if cfg.moe_experts:
        raise ValueError("a dense source checkpoint is required")
    bundle=torch.load(adapters_file,map_location="cpu",weights_only=True)
    manifest=bundle["receipt"]
    if manifest.get("synthetic_smoke_only") and not allow_synthetic:
        raise ValueError("synthetic smoke weights are not trained-model evidence")
    candidates=bundle["student_state_dicts"]
    required={str(i) for i in range(cfg.n_layers)}
    if set(candidates)!=required:
        raise ValueError("all layers require individually distilled parity adapters")
    p=manifest["expert_configuration"]
    upgraded=dataclasses.replace(cfg,
        moe_experts=p["num_experts"],moe_top_k=p["top_k"],
        moe_expert_width=p["expert_width"],
        moe_route_geometry=p["route_geometry"],
        moe_transition_rank=p["transition_rank"],
        moe_deployed_parity=True,moe_even_width=max(24,cfg.d_model//2),
        moe_shared_e8_history=bool(p.get("shared_e8_history",False)))
    destination=LeechGPT(upgraded)
    target=destination.state_dict()
    expected_new=set()
    for layer in range(cfg.n_layers):
        prefix=f"blocks.{layer}."
        adapter=candidates[str(layer)]
        for part,owner in (("odd_experts","moe"),("even_student","even_student")):
            for key,value in adapter[part].items():
                name=prefix+owner+"."+key
                if name not in target or target[name].shape!=value.shape:
                    raise ValueError("student key/layout mismatch: "+name)
                target[name]=value
                expected_new.add(name)
    source_keys={k for k in source if ".ffn." not in k}
    if not source_keys.issubset(target):
        raise ValueError("source non-FFN weights missing from target")
    for key in source_keys:
        if target[key].shape!=source[key].shape:
            raise ValueError("source parameter mismatch: "+key)
        target[key]=source[key]
    if set(target)!=(source_keys | expected_new):
        extra=set(target)-(source_keys | expected_new)
        raise ValueError("uninitialized destination parameters: "+str(sorted(extra)[:8]))
    destination.load_state_dict(target,strict=True)
    original=LeechGPT(cfg)
    original.load_state_dict(source,strict=True)
    return original.eval(),destination.eval(),upgraded


@torch.no_grad()
def compare_language_models(dense,student,ids,device="cpu",batch=2):
    dense.to(device).eval()
    student.to(device).eval()
    total=0
    max_delta=0.
    squared_delta=0.
    original_nll=0.
    candidate_nll=0.
    for start in range(0,len(ids),batch):
        tokens=torch.from_numpy(np.ascontiguousarray(ids[start:start+batch],
                                     dtype=np.int64)).to(device)
        old=dense(tokens)[0].float()
        new=student(tokens)[0].float()
        difference=(new-old).abs()
        max_delta=max(max_delta,float(difference.max()))
        squared_delta+=float(difference.square().sum())
        total+=difference.numel()
        if tokens.shape[1]>1:
            targets=tokens[:,1:].reshape(-1)
            old_logits=old[:,:-1].reshape(-1,old.shape[-1])
            new_logits=new[:,:-1].reshape(-1,new.shape[-1])
            original_nll+=float(F.cross_entropy(old_logits,targets,reduction="sum"))
            candidate_nll+=float(F.cross_entropy(new_logits,targets,reduction="sum"))
    num_targets=len(ids)*(ids.shape[1]-1)
    return {"max_logit_abs_delta":max_delta,
        "rmse_logit_delta":(squared_delta/total)**.5,
        "dense_next_token_ce":original_nll/num_targets if num_targets else None,
        "student_next_token_ce":candidate_nll/num_targets if num_targets else None,
        "language_parity_verified":False}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint",required=True,type=Path)
    p.add_argument("--config",required=True,type=Path)
    p.add_argument("--adapters",required=True,type=Path)
    p.add_argument("--heldout-token-ids",required=True,type=Path)
    p.add_argument("--output",type=Path,default=Path("signed_wave_full_model.pt"))
    p.add_argument("--device",default="cpu")
    p.add_argument("--batch",type=int,default=2)
    p.add_argument("--max-logit-delta",type=float,default=None)
    p.add_argument("--allow-synthetic-smoke",action="store_true")
    args=p.parse_args()
    dense,student,config=convert(args.checkpoint,args.config,args.adapters,
                                 allow_synthetic=args.allow_synthetic_smoke)
    ids=np.load(args.heldout_token_ids,allow_pickle=False)
    if ids.ndim!=2 or ids.shape[0]<1 or ids.shape[1]>config.block_size or ids.shape[1]<1:
        raise ValueError("heldout tokens must be [N,1<=T<=block_size]")
    if (not np.issubdtype(ids.dtype,np.integer) or np.any(ids<0)
            or np.any(ids>=config.vocab_size)):
        raise ValueError("invalid heldout token IDs")
    evidence=compare_language_models(dense,student,ids,args.device,args.batch)
    evidence["language_parity_verified"]=(
        args.max_logit_delta is not None and
        evidence["max_logit_abs_delta"]<=args.max_logit_delta)
    evidence.update({
        "source_model_sha256":tensor_sha256(args.checkpoint),
        "adapter_file_sha256":tensor_sha256(args.adapters),
        "candidate_only":not evidence["language_parity_verified"],
        "semantic_face":"not supplied; no semantic opposition/antipode inference",
        "model_architecture":dataclasses.asdict(config),
        "warning":"Held-out loss is empirical; model/consumer action theorems require separate proofs."})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    torch.save({"model_state_dict":student.cpu().state_dict(),
                "config":dataclasses.asdict(config),"verification":evidence},args.output)
    args.output.with_suffix(".json").write_text(json.dumps(evidence,indent=2))
    print(json.dumps(evidence,indent=2))
    if args.max_logit_delta is not None and not evidence["language_parity_verified"]:
        raise SystemExit(2)


if __name__=="__main__":
    main()
