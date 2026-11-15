"""X369-F3 source C3 and C3² relative-phase controls.

Evaluates actual Leech routing feature functions on explicit inputs. These
are mathematical controls, NOT evidence that a trained LM learned a C3
task action, phase composition or a causally useful circuit.
"""
from __future__ import annotations
import argparse,json
import torch
from models.trialectic_wave_prior import (
    rotate3,trialectic_router_invariants,directed_trialectic_invariants)
from models.punctured_nine_wave_prior import (
    punctured_blocks_to_wave_blocks,pair_wave_router_features,translate_wave8)

def c3_relational_controls(x,h):
    if x.shape!=h.shape or x.shape[-1]%24:
        raise ValueError("matching original 24D activation/history blocks required")
    def features(a,b):
        return torch.cat((trialectic_router_invariants(a),
                          directed_trialectic_invariants(a,b)),-1)
    base=features(x,h)
    diagonal=features(rotate3(x),rotate3(h))
    relative=features(rotate3(x),h)
    return {
        "diagonal_c3_max_feature_defect":float((base-diagonal).abs().max()),
        "relative_c3_max_feature_difference":float((base-relative).abs().max()),
        "diagonal_c2_max_feature_defect":float((base-features(-x,-h)).abs().max()),
        "relative_rotation_input_norm_defect":
            float((x.norm(dim=-1)-rotate3(x).norm(dim=-1)).abs().max()),
        "task_action_or_learned_c3_proved":False}

def c3_square_controls(x,h):
    if x.shape!=h.shape or x.shape[-1]%24:
        raise ValueError("matching original 24D activation/history blocks required")
    wx=punctured_blocks_to_wave_blocks(x).reshape(*x.shape[:-1],-1,8)
    wh=punctured_blocks_to_wave_blocks(h).reshape(*h.shape[:-1],-1,8)
    def features(a,b):
        return pair_wave_router_features(a.reshape(x.shape),b.reshape(h.shape),
                                         compact=True)
    baseline=features(wx,wh)
    defects=[];relative=[];norms=[]
    for a in range(3):
        for b in range(3):
            tx=translate_wave8(wx,(a,b))
            th=translate_wave8(wh,(a,b))
            defects.append(float((baseline-features(tx,th)).abs().max()))
            relative.append(float((baseline-features(tx,wh)).abs().max()))
            norms.append(float((wx.square().sum(-1)-tx.square().sum(-1)).abs().max()))
    return {"nine_group_elements_evaluated":9,
        "diagonal_c3_square_max_feature_defect":max(defects),
        "one_sided_c3_square_max_feature_difference":max(relative),
        "wave_fourier_pair_norm_defect":max(norms),
        "naive_punctured_8_norm_claimed_preserved":False,
        "learned_c3_square_mechanism_proved":False}

def report(dtype=torch.float64):
    x=torch.arange(1,49,dtype=dtype).reshape(2,24)/48.
    h=torch.flip(x,[-1]).clone()
    c3=c3_relational_controls(x,h)
    square=c3_square_controls(x,h)
    if c3["diagonal_c3_max_feature_defect"]>1e-11 or (
            c3["relative_c3_max_feature_difference"]<=1e-5):
        raise AssertionError("C3 source action/negative control failed")
    if square["diagonal_c3_square_max_feature_defect"]>1e-11 or (
            square["one_sided_c3_square_max_feature_difference"]<=1e-5):
        raise AssertionError("C3² source action/negative control failed")
    return {"schema":1,
        "kind":"original_source_c3_c3_square_relative_phase_negative_controls",
        "source_c3":c3,"source_c3_square":square,
        "original_model_checkpoint_loaded":False,
        "causal_ce_measured":False,
        "note":"Source mathematical feature tests; not a trained neural C3 circuit."}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device",default="cpu",
                   help="Native launcher compatibility; finite controls use CPU")
    p.add_argument("--output",type=str)
    a=p.parse_args()
    result=report()
    rendered=json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n"
    if a.output:
        from pathlib import Path
        path=Path(a.output)
        if path.exists():raise FileExistsError("refuse overwrite exact source receipt")
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(rendered)
    print(rendered,end="")
if __name__=="__main__":main()
