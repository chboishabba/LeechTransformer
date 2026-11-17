#!/usr/bin/env python3
"""Source-NATIVE R4 topk/ResponseKink architecture fixture on CPU.

Runs the actual HistoryOrbitMoE model and its original selected-execution
consumer, not a duplicate scorer or substitute GELU kernel. The weights
are explicit synthetic source-legal parameters, not a trained checkpoint.
No GPU, Agda, CI or physical performance inference.
"""
import argparse
import json
import math
from pathlib import Path
import sys
import subprocess
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from models.history_orbit_moe import HistoryOrbitMoE


def test_original_source_topk_replacement_kink(emit_probe_dir=None):
    m=HistoryOrbitMoE(d_model=6,num_experts=4,top_k=2,
                      route_geometry="c2_c3_residual",
                      router_selection="topk").double().eval()
    slopes=[0,4,9,11]
    biases=[8,3,-11,-16]
    with torch.no_grad():
        m.router.weight.zero_()
        m.router.bias.zero_()
        m.c3_residual_weight.zero_()
        # Original R4 residual features consist of unary 6 +
        # directed 6; channel 0 selects the first self inner product.
        m.c3_residual_weight[0,6]=1
        m.expert_bias.copy_(torch.tensor(biases,dtype=torch.double))
        m.orientation.weight.zero_()
        m.orientation.weight[0,6]=1
        m.orientation.bias.zero_()
        for e in range(4):
            m.experts[e][0].weight.zero_()
            m.experts[e][0].bias.zero_()
            m.experts[e][2].weight.zero_()
            m.experts[e][2].bias.zero_()
            m.experts[e][2].bias[0]=1 if e==1 else 0
            m.transitions[e].weight.zero_()
            m.transitions[e].bias.zero_()
    h=torch.zeros(1,4,6,dtype=torch.double)
    for e,slope in enumerate(slopes):
        h[0,e,0]=1
        h[0,e,2]=slope
    expected_ids=[[0,1],[0,1],[1,0],[3,2]]
    expected_logits=[[8,3,-11,-16],[8,7,-2,-5],
                     [8,11,7,6],[8,15,16,17]]
    selected_obs=[]
    forced_obs=[]
    nat=[]
    fixed=None
    with torch.no_grad():
        for t in range(4):
            x=torch.zeros(1,6,dtype=torch.double)
            x[0,0]=t
            # Backward-compatible production caller must retain 2-tuple
            # and the opt-in observer must not change the original route.
            old_ids,old_weights=m.select_source_experts(x,h)
            ids,weights,logits=m.select_source_experts(x,h,return_logits=True)
            assert torch.equal(ids,old_ids)
            assert torch.equal(weights,old_weights)
            assert ids[0].tolist()==expected_ids[t]
            assert logits[0].tolist()==expected_logits[t]
            assert (weights>0).all()
            y,hnew,ids2,signs,w2=m.step(x,h)
            assert torch.equal(ids,ids2)
            assert torch.equal(weights,w2)
            assert torch.equal(signs,torch.ones_like(signs))
            selected_h=h.gather(1,ids.unsqueeze(-1).expand(-1,-1,6))
            replay,updated,rsigns=m.execute_selected(x,ids,selected_h,weights)
            assert torch.allclose(replay,y,atol=1e-12,rtol=1e-12)
            assert torch.equal(signs,rsigns)
            for slot,e in enumerate(ids[0]):
                assert torch.allclose(updated[0,slot],hnew[0,int(e)],
                                      atol=1e-12,rtol=1e-12)
            if fixed is None:fixed=ids.clone()
            fw=torch.softmax(logits.gather(1,fixed),dim=-1)
            fh=h.gather(1,fixed.unsqueeze(-1).expand(-1,-1,6))
            fy,_,_=m.execute_selected(x,fixed,fh,fw)
            v=float(y[0,0])
            selected_obs.append(v)
            forced_obs.append(float(fy[0,0]))
            nat.append(math.floor(10*v))
    assert nat==[0,2,9,0]
    assert selected_obs[0]<selected_obs[1]<selected_obs[2]
    assert selected_obs[3]<selected_obs[2]
    assert all(a<b for a,b in zip(forced_obs,forced_obs[1:]))
    # The original forced selection removes the overload downturn.
    assert forced_obs[-1]>forced_obs[-2]
    # SAME CLOCK/source input, DIFFERENT original retained histories:
    # prefix t=0 vs t=3, then probe t=2 at step index 1 on both paths.
    # This tests an actual recurrent source interaction, distinct from
    # resetting history independently for each of four loads.
    from scripts.probe_x369_r4_history_path_dependence import path_evaluation
    base_x=torch.zeros(6,dtype=torch.double)
    base_x[0]=1
    paths=path_evaluation(m,base_x,h[0],(0.,1.,2.,3.),0,10)
    assert paths["probe_step_index"]==1
    assert paths["full_history_l2_distance_before_probe"]>0
    assert paths["same_load_and_clock_history_dependent_response_observed"]
    assert paths["probe_nat_divergence_observed"] is True
    assert [p["probe_source_ids"] for p in paths["paths"]]==[[0,2],[1,0]]
    assert [p["probe_source_logits"] for p in paths["paths"]]==[
        [8.,3.,7.,6.],[8.,11.,-11.,-16.]]
    # No selected/top-k-cutoff tie on either second-step path.
    for record in paths["paths"]:
        logits=record["probe_source_logits"]
        ids=record["probe_source_ids"]
        assert logits[ids[0]]>logits[ids[1]]
        assert logits[ids[1]]>max(logits[e] for e in range(4)
                                   if e not in ids)
    assert [p["probe_nat_response"] for p in paths["paths"]]==[0,9]
    assert paths["general_hysteresis_certified"] is False
    print("X369_R4_NATIVE_SAME_CLOCK_HISTORY_PATH_PASS",
          "Nat",[p["probe_nat_response"] for p in paths["paths"]])

    if emit_probe_dir is not None:
        # Persist the SAME synthetic native source state to demonstrate
        # checkpoint byte hashing, load/strict-key reload and full native
        # four-stage producer. Never label these bytes trained weights.
        out=Path(emit_probe_dir)
        out.mkdir(parents=True,exist_ok=True)
        ckpt=out/"synthetic_history_orbit_moe_state_dict.pt"
        inp=out/"synthetic_original_load_input.json"
        torch.save(m.state_dict(),ckpt)
        request={
            "model":{
                "d_model":6,"num_experts":4,"top_k":2,
                "route_geometry":"c2_c3_residual",
                "router_selection":"topk"},
            "base_x":["1","0","0","0","0","0"],
            "history":[[str(int(v)) for v in row] for row in h[0].tolist()],
            "load_levels":["0","1","2","3"],
            "response":{"type":"coordinate","index":0,"scale":10},
            "dtype":"float64",
            "history_cap":None,  # source original nonpersistent setting
            "history_revision":"SYNTHETIC-four-load-fixed-reset-histories"
        }
        with inp.open("w",encoding="utf8") as stream:
            json.dump(request,stream,indent=2,sort_keys=True)
            stream.write("\n")
        # Exercise the SAME original source producer immediately:
        # parse numeric-string JSON, strict checkpoint restore and
        # selected-only replay. The output remains SYNTHETIC.
        from scripts.probe_x369_r4_kink_checkpoint import (
            source_tensor, load_original, original_probe, sha256)
        with inp.open(encoding="utf8") as stream:
            loaded_request=json.load(stream)
        parsed_x=source_tensor(loaded_request["base_x"],(6,),torch.float64)
        parsed_h=source_tensor(loaded_request["history"],(4,6),torch.float64)
        rebuilt=HistoryOrbitMoE(
            d_model=6,num_experts=4,top_k=2,
            route_geometry="c2_c3_residual",
            router_selection="topk").double().eval()
        rebuilt.history_cap=loaded_request["history_cap"]
        load_original(ckpt,rebuilt,"")
        reproduced=original_probe(
            rebuilt,parsed_x,parsed_h,[0.,1.,2.,3.],0,10)
        assert reproduced["nat_kink_observed"] is True
        assert reproduced["topk_replacement_isolated_from_fixed_route"] is True
        assert reproduced["native_vs_fixed_route_nat_kink_difference"] is True
        assert reproduced["fixed_route_nat_kink_observed"] is False
        assert reproduced["fitted_topk_causal_mechanism_certified"] is False
        assert [t["source_nat_response"] for t in reproduced["stages"]]==nat
        assert all(t["source_selected_history_replay"]
                   for t in reproduced["stages"])
        assert len(sha256(ckpt))==64
        # Adversarial JSON parsing and original strict-load safety.
        for invalid in (["NaN"]*6,["Infinity"]*6,
                        [True]*6,["0"]*5):
            try:source_tensor(invalid,(6,),torch.float64)
            except ValueError:pass
            else:raise AssertionError("invalid source JSON tensor accepted")
        try:source_tensor(loaded_request["history"],(6,),torch.float64)
        except ValueError:pass
        else:raise AssertionError("wrong original history shape accepted")
        # Genuine CLI end-to-end acceptance in the ORIGINAL source
        # process, not only direct calls to the functions.
        source_root=Path(__file__).resolve().parents[1]
        reset_out=out/"synthetic_probe_result.json"
        history_out=out/"synthetic_history_path_result.json"
        subprocess.run([
            sys.executable,str(source_root/"scripts/probe_x369_r4_kink_checkpoint.py"),
            "--checkpoint",str(ckpt.resolve()),
            "--input",str(inp.resolve()),
            "--output",str(reset_out.resolve())],
            cwd=source_root,check=True)
        subprocess.run([
            sys.executable,str(source_root/"scripts/probe_x369_r4_history_path_dependence.py"),
            "--checkpoint",str(ckpt.resolve()),
            "--input",str(inp.resolve()),
            "--output",str(history_out.resolve())],
            cwd=source_root,check=True)
        with reset_out.open(encoding="utf8") as stream:
            cli_reset=json.load(stream)
        with history_out.open(encoding="utf8") as stream:
            cli_history=json.load(stream)
        assert [t["source_nat_response"] for t in cli_reset["stages"]]==[0,2,9,0]
        assert cli_reset["native_vs_fixed_route_nat_kink_difference"] is True
        assert [p["probe_nat_response"] for p in cli_history["paths"]]==[0,9]
        assert all(p["probe_strict_topk_gap"]>0 and
                   p["prefix_strict_topk_gap"]>0
                   for p in cli_history["paths"])
        assert cli_reset["original_checkpoint_sha256"]==sha256(ckpt)
        assert cli_history["original_checkpoint_sha256"]==sha256(ckpt)
        assert cli_reset["certified_float_error_intervals"] is False
        assert cli_history["general_hysteresis_certified"] is False
        print("X369_R4_NATIVE_CLI_END_TO_END_PASS",
              "same checkpoint",cli_reset["original_checkpoint_sha256"],
              "reset_nat",[t["source_nat_response"]
                           for t in cli_reset["stages"]],
              "same_clock_nat",[p["probe_nat_response"]
                                for p in cli_history["paths"]])
        print("X369_R4_EMITTED_SYNTHETIC_SOURCE_FIXTURE",ckpt,inp)
        print("RUN_SOURCE_CHECKPOINT_PROBE",
              "python scripts/probe_x369_r4_kink_checkpoint.py",
              "--checkpoint",ckpt,"--input",inp,
              "--output",out/"synthetic_probe_result.json")
    print("X369_R4_NATIVE_SOURCE_KINK_PASS",
          "Nat",nat,"native",selected_obs,"fixed",forced_obs)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--emit-probe-dir",type=Path)
    args=parser.parse_args()
    test_original_source_topk_replacement_kink(args.emit_probe_dir)
