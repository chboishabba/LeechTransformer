"""Audit the actual Leech C2/C3 source action, without retraining.

Checks checkpoint *stored geometry*, original router decisions under sign
reversal / simultaneous cyclic rotation / swap reflection. The reflection
conjugates a 3-cycle but is NOT the central C2 antipodal generator.
All three transformations act on an explicitly ordered hidden-state triad,
not on token IDs, semantic concepts, E8 roots or a D4 root automorphism.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
from models.trialectic_wave_prior import (
    rotate3,trialectic_router_invariants,directed_trialectic_invariants)
from scripts.verify_trained_federated_moe import checkpoint_parts,infer_config
from scripts.train_guarded_moe_continuation import sha256_file
from models.model import LeechGPT

def reflect3(x):
    if x.shape[-1]%3:raise ValueError("reflection requires ordered triples")
    tri=x.reshape(*x.shape[:-1],x.shape[-1]//3,3)
    return tri[..., [0,2,1]].reshape(x.shape)

def group_witness(x):
    """Finite C3 and S3 laws on the precise ordered-triad carrier."""
    return {"c3_cube":bool(torch.equal(rotate3(rotate3(rotate3(x))),x)),
        "reflection_square":bool(torch.equal(reflect3(reflect3(x)),x)),
        "srs_inverse":bool(torch.equal(
            reflect3(rotate3(reflect3(x))),rotate3(rotate3(x)))),
        "antipodal_commutes_c3":bool(torch.equal(rotate3(-x),-rotate3(x))),
        "antipodal_commutes_reflection":bool(torch.equal(reflect3(-x),-reflect3(x)))}

@torch.no_grad()
def inspect_router(moe,x,h,tolerance):
    if x.shape!=(h.shape[0],moe.d_model) or h.shape!=(len(x),moe.num_experts,moe.d_model):
        raise ValueError("actual original x/history shapes needed")
    # These actions only describe hidden triad geometry. They are not
    # automatically compatible with the source embedding or full model.
    def routed(a,b):
        ids,weights=moe.select_source_experts(a,b)
        return ids,weights
    base=routed(x,h)
    variants={"antipodal":(-x,-h)}
    if moe.d_model%3==0:
        variants["cyclic"]=(rotate3(x),rotate3(h))
        variants["reflect"]=(reflect3(x),reflect3(h))
    report={}
    for label,(a,b) in variants.items():
        ids,weights=routed(a,b)
        report[label]={"expert_ids_identical":bool(torch.equal(ids,base[0])),
            "max_topk_weight_abs_delta":float((weights-base[1]).abs().max()),
            "weights_within_tolerance":bool(torch.allclose(
                weights,base[1],rtol=0,atol=tolerance))}
    return report

@torch.no_grad()
def feature_contract(x,h,atol):
    if x.shape!=h.shape or x.shape[-1]%3:
        raise ValueError("triadic feature probe requires equal, divisible-by-three widths")
    def feature(a,b):
        return torch.cat((trialectic_router_invariants(a),
                          directed_trialectic_invariants(a,b)),dim=-1)
    ref=feature(x,h)
    return {name:{"max_abs":float((feature(a,b)-ref).abs().max()),
        "within_tolerance":bool(torch.allclose(feature(a,b),ref,rtol=0,atol=atol))}
        for name,a,b in (
            ("antipodal",-x,-h),
            ("cyclic",rotate3(x),rotate3(h)),
            ("reflect",reflect3(x),reflect3(h)))}

def audit(args):
    if args.output.exists():raise FileExistsError("refuse overwrite original receipt")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("native GPU unavailable")
    state,metadata=checkpoint_parts(args.candidate)
    cfg=infer_config(state,metadata,None)
    model=LeechGPT(cfg).eval()
    model.load_state_dict(state,strict=True)
    if not cfg.moe_experts:raise ValueError("source checkpoint has no MoE")
    if cfg.d_model%3:raise ValueError("ordered triad width not supported")
    model.to(args.device).eval()
    # Diagnostic inputs are reproducible synthetic HIDDEN states because the
    # C3 action is not a defined token-ID transformation. This is not an
    # empirical semantic equivariance test or a real sequence history.
    generator=torch.Generator(device=args.device)
    generator.manual_seed(args.seed)
    x=torch.randn(args.rows,cfg.d_model,device=args.device,generator=generator)
    h=torch.randn(args.rows,cfg.moe_experts,cfg.d_model,
                  device=args.device,generator=generator)
    results=[inspect_router(block.moe,x,h,args.atol) for block in model.blocks]
    group=group_witness(x)
    feature=feature_contract(x,h[:,0,:],args.atol)
    report={"schema":1,"kind":"original_checkpoint_c2c3_triads_router_audit",
        "source_checkpoint_sha256":sha256_file(args.candidate),
        "source_moe_route_geometry":cfg.moe_route_geometry,
        "num_experts":cfg.moe_experts,"layers":cfg.n_layers,
        "hidden_width":cfg.d_model,"sample_rows":args.rows,"seed":args.seed,
        "input_kind":"synthetic_hidden_states_not_token_semantics",
        "actions":{"central_c2":"antipodal -I",
                   "c3":"simultaneous ordered-triad rotation",
                   "reflection":"independent triad reflection, not central -I"},
        "exact_finite_group_laws":group,
        "c2c3_feature_residuals":feature,
        "original_trained_router_per_layer":results,
        "trained_checkpoint_used_c2c3_wave":cfg.moe_route_geometry=="c2c3_wave",
        "whole_model_equivariance_proved":False,
        "semantic_c3_intervention_proved":False,
        "lattice_E8_or_D4_triality_proved":False}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print(json.dumps(report,indent=2,sort_keys=True,allow_nan=False))
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate",type=Path,required=True)
    p.add_argument("--device",default="cuda")
    p.add_argument("--rows",type=int,default=8)
    p.add_argument("--seed",type=int,default=41)
    p.add_argument("--atol",type=float,default=2e-5)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    if args.rows<1 or args.atol<0:raise ValueError("invalid probe size/tolerance")
    audit(args)
if __name__=="__main__":main()
