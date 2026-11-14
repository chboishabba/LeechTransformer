"""Original-checkpoint matched-pair C2 parity and C3 phase spectral diagnostic.

Collects real teacher LayerNorm pre-FFN hidden activations from original
train-derived development token rows. Runs original trained student's odd
HistoryOrbitMoE and even_student separately on (x,0), (-x,0), with zero
initial recurrent expert histories. This isolates the layer-local *source*
approximation and C3 phase-energy error, not full student recurrence.

The C3 rotation acts on arbitrary consecutive three-coordinate blocks of
hidden state and is a TESTED representational geometry only; these blocks
have not been proved to encode semantic triads, modular inputs or Leech
lattice vectors. No heldout data, weight mutation or optimizer.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import numpy as np
import torch
from models.trialectic_wave_prior import wave3,rotate3
from models.dense_to_signed_experts import canonical_even_input
from scripts.reproduce_c2_quality_and_layers import load_trained
from scripts.train_guarded_moe_continuation import sha256_file

def stats(x):
    x=x.detach().double()
    return float(x.square().mean())

@torch.no_grad()
def local_probe(block,x):
    if x.ndim!=2 or x.shape[-1]%3:
        raise ValueError("expected teacher-normalised source hidden triads")
    if block.even_student is None or block.moe is None or block.ffn is None:
        raise ValueError("original guarded parity block required")
    moe=block.moe
    even=block.even_student
    def parts(z):
        o=moe(z.unsqueeze(1),train_sign=False)[0].squeeze(1)
        e=even(canonical_even_input(z))
        return o,e
    odd,ev=parts(x)
    odd_neg,ev_neg=parts(-x)
    y=block.ffn(x)
    y_neg=block.ffn(-x)
    expected_even=(y+y_neg)*.5
    expected_odd=(y-y_neg)*.5
    residual=odd+ev-y
    phase=wave3(residual).reshape(-1,x.shape[-1]//3,3)
    dc=phase[...,0]
    rot=phase[...,1:]
    denom=dc.square().sum()+rot.square().sum()
    phase_fraction=float(rot.square().sum()/denom) if float(denom)>0 else 0.
    p3_odd,p3_ev=parts(rotate3(x))
    # This is an empirical intertwiner defect of source channels under
    # an arbitrary selected hidden-triad action, NOT a C3 architecture law.
    ox=float((p3_odd-rotate3(odd)).square().mean().sqrt())
    ex=float((p3_ev-rotate3(ev)).square().mean().sqrt())
    cross=(odd.double()*ev.double()).sum(dim=-1)
    cross_neg=(odd_neg.double()*ev_neg.double()).sum(dim=-1)
    paired=(cross+cross_neg)*.5
    parity_odd=float((odd_neg+odd).abs().max())
    parity_even=float((ev_neg-ev).abs().max())
    return {"hidden_rows":int(x.shape[0]),
        "teacher_odd_mse":stats(odd-expected_odd),
        "teacher_even_mse":stats(ev-expected_even),
        "teacher_total_mse":stats(residual),
        "source_output_even_odd_cross_mean":float(cross.mean()),
        "cross_mean_at_negated_hidden":float(cross_neg.mean()),
        "paired_cross_mean":float(paired.mean()),
        "paired_cross_abs_max":float(paired.abs().max()),
        "source_odd_parity_max_abs":parity_odd,
        "source_even_parity_max_abs":parity_even,
        "c3_wave_total_residual_mse":stats(residual),
        "c3_wave_rotating_residual_energy_fraction":phase_fraction,
        "c3_odd_output_equivariance_rmse":ox,
        "c3_even_output_equivariance_rmse":ex,
        "even_output_pair_sum_rms":float((ev+ev_neg).square().mean().sqrt()),
        "triad_action_is_semantic":False}

@torch.no_grad()
def collect_layernorm(dense,ids,layers,device):
    captured={}
    handles=[]
    for index in layers:
        def hook(module,inputs,out,index=index):
            captured[index]=out.detach().reshape(-1,out.shape[-1]).cpu()
        handles.append(dense.blocks[index].ln2.register_forward_hook(hook))
    try:
        dense.to(device).eval()
        dense(ids.to(device))
    finally:
        for handle in handles:handle.remove()
        dense.to("cpu")
        if device.startswith("cuda"):torch.cuda.empty_cache()
    if set(captured)!=set(layers):
        raise AssertionError("original dense block LayerNorm capture incomplete")
    return captured

def run(a):
    if a.output.exists():raise FileExistsError("do not overwrite source proof receipt")
    receipt=json.loads(a.original_receipt.read_text())
    if receipt.get("kind")!="native_gpu_cumulative_student_quality_curriculum":
        raise ValueError("wrong original source curriculum receipt")
    if sha256_file(a.train_ids)!=receipt.get("train_ids_sha256"):
        raise ValueError("original fit-derived train data SHA mismatch")
    if sha256_file(a.teacher)!=receipt.get("source_teacher_sha256"):
        raise ValueError("original dense source SHA mismatch")
    source=np.load(a.train_ids,allow_pickle=False)
    indices=receipt.get("train_development_indices")
    if (not isinstance(indices,list) or not indices
            or any(type(i)!=int or i<0 or i>=len(source) for i in indices)):
        raise ValueError("no original development split")
    ids=torch.as_tensor(source[indices[:a.rows]],dtype=torch.long)
    if ids.ndim!=2 or ids.shape[0]<1:
        raise ValueError("no original development rows")
    dense,student=load_trained(a.candidate,a.teacher,a.config)
    layers=(0,1)
    norm=collect_layernorm(dense,ids,layers,a.device)
    del dense
    student.to(a.device).eval()
    out={}
    for layer in layers:
        out[str(layer)]=local_probe(student.blocks[layer],norm[layer].to(a.device))
    result={"schema":1,"kind":"actual_c2_parity_c3_fourier_grokking_diagnostic",
        "source_candidate_sha256":sha256_file(a.candidate),
        "source_teacher_sha256":sha256_file(a.teacher),
        "source_train_sha256":sha256_file(a.train_ids),
        "selected_train_development_indices":indices[:a.rows],
        "input_kind":"actual_dense_teacher_pre_ffn_normalized_activations",
        "source_history":"zero_initial_history_local_MoE_not_real_recurrent_rollout",
        "trained_route_geometry":student.cfg.moe_route_geometry,
        "paired_input_action":"(x,zero_history) and (-x,zero_history)",
        "hidden_c3_action":"cyclic rotation of consecutive coordinates in triples",
        "layers":out,
        "independent_heldout_used":False,"weights_changed":False,
        "formal_semantic_face_proven":False,
        "source_c3_equivariance_proven":False,
        "quality_promotion":False}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print(json.dumps(result,indent=2,sort_keys=True,allow_nan=False))
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate",type=Path,required=True)
    p.add_argument("--teacher",type=Path,required=True)
    p.add_argument("--config",type=Path)
    p.add_argument("--train-ids",type=Path,required=True)
    p.add_argument("--original-receipt",type=Path,required=True)
    p.add_argument("--device",default="cuda")
    p.add_argument("--rows",type=int,default=2)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.rows<1:raise ValueError("positive original dev rows required")
    if a.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("native GPU unavailable")
    run(a)
if __name__=="__main__":main()
