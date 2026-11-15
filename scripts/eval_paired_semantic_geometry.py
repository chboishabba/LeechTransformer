"""Typed Sophontic/DASHI paired semantic and geometric-action test.

An opposite argument is NOT automatically a negated latent activation.
The caller supplies distinct correct token targets for each item. We
measure *joint correctness* and flip calibration, then separately fit
identity/C2/C3 actions on hidden states without promoting fit to cause.
No moral interpretation of geometric signs.
"""
from __future__ import annotations
import argparse
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from models.model import LeechGPT
from models.trialectic_wave_prior import rotate3
from scripts.verify_trained_federated_moe import checkpoint_parts,infer_config

RESPONSE_KINDS={
 "denyPremise","challengeEvidence","challengeRule","disputeCausation",
 "disputeCharacterisation","addContext","alternativeExplanation",
 "acceptWithQualification","suspendJudgement"}


def paired_evidence(logits_a,logits_b,hidden_a,hidden_b,
                    target_a,target_b,response_kinds):
    if (logits_a.shape != logits_b.shape or logits_a.ndim !=2
        or hidden_a.shape != hidden_b.shape or hidden_a.ndim !=2
        or logits_a.shape[0]!=hidden_a.shape[0]):
        raise ValueError("paired tensors must share batch, [B,vocab] and [B,d]")
    n=logits_a.shape[0]
    if target_a.shape!=(n,) or target_b.shape!=(n,) or len(response_kinds)!=n:
        raise ValueError("invalid target/kind lengths")
    if any(str(kind) not in RESPONSE_KINDS for kind in response_kinds):
        raise ValueError("unknown typed response kind")
    pred_a=logits_a.argmax(-1).cpu()
    pred_b=logits_b.argmax(-1).cpu()
    expected_change=(target_a!=target_b).cpu()
    observed_change=(pred_a!=pred_b)
    both_correct=(pred_a==target_a.cpu())&(pred_b==target_b.cpu())
    action_errors=torch.stack((
        (hidden_b-hidden_a).square().mean(-1),
        (hidden_b+hidden_a).square().mean(-1),
        (hidden_b-rotate3(hidden_a)).square().mean(-1)),dim=-1)
    selected=action_errors.argmin(-1).cpu()
    actions=("identity","C2_inversion","C3_cyclic")
    by_kind=defaultdict(lambda:{"n":0,"joint_correct":0,"actual_changed":0,
                                "predicted_changed":0,"spurious_flip":0})
    for i,kind in enumerate(response_kinds):
        row=by_kind[str(kind)]
        row["n"]+=1;row["joint_correct"]+=int(both_correct[i])
        row["actual_changed"]+=int(expected_change[i])
        row["predicted_changed"]+=int(observed_change[i])
        row["spurious_flip"]+=int(observed_change[i] and not expected_change[i])
    return {
        "pairs":n,
        "joint_correct_rate":float(both_correct.float().mean()),
        "target_change_rate":float(expected_change.float().mean()),
        "predicted_flip_rate":float(observed_change.float().mean()),
        "flip_decision_accuracy":float((expected_change==observed_change).float().mean()),
        "action_fit_candidates":actions,
        "best_action_counts":{name:int((selected==j).sum()) for j,name in enumerate(actions)},
        "action_mse":{name:float(action_errors[:,j].mean()) for j,name in enumerate(actions)},
        "by_response_kind":dict(by_kind),
        "nonpromotion":"Fit is not semantic opposition, mechanism, ethical valence, or proof of equivariance."
    }


@torch.no_grad()
def run_model(model,arr,device,batch):
    logits=[];hidden=[]
    for start in range(0,len(arr),batch):
        tokens=torch.from_numpy(np.ascontiguousarray(arr[start:start+batch],
                                     dtype=np.int64)).to(device)
        lg,hd,_,_=model(tokens)
        logits.append(lg[:,-1].detach().cpu())
        hidden.append(hd[:,-1].detach().cpu())
    return torch.cat(logits,0),torch.cat(hidden,0)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint",required=True,type=Path)
    p.add_argument("--config",type=Path)
    p.add_argument("--pairs",required=True,type=Path,help="npz: tokens_a,b targets_a,b kinds")
    p.add_argument("--device",default="cpu")
    p.add_argument("--batch",type=int,default=2)
    p.add_argument("--output",type=Path,default=Path("paired_geometry_receipt.json"))
    a=p.parse_args()
    state,saved=checkpoint_parts(a.checkpoint)
    cfg=infer_config(state,saved,a.config)
    model=LeechGPT(cfg)
    model.load_state_dict(state,strict=True)
    model.to(a.device).eval()
    with np.load(a.pairs,allow_pickle=False) as packet:
        required={"tokens_a","tokens_b","targets_a","targets_b","kinds"}
        if not required.issubset(set(packet.files)):
            raise ValueError("missing paired data fields")
        tokens_a,tokens_b=packet["tokens_a"],packet["tokens_b"]
        targets_a,targets_b=packet["targets_a"],packet["targets_b"]
        kinds=packet["kinds"]
    if tokens_a.shape!=tokens_b.shape or tokens_a.ndim!=2 or len(tokens_a)<1:
        raise ValueError("tokens_a/b must share [N,T] shape")
    if tokens_a.shape[1]>cfg.block_size or tokens_a.shape[1]<1:
        raise ValueError("invalid sequence length")
    for value in (tokens_a,tokens_b,targets_a,targets_b):
        if (not np.issubdtype(value.dtype,np.integer)
            or np.any(value<0) or np.any(value>=cfg.vocab_size)):
            raise ValueError("out of vocabulary or invalid numeric token ids")
    la,ha=run_model(model,tokens_a,a.device,a.batch)
    lb,hb=run_model(model,tokens_b,a.device,a.batch)
    receipt=paired_evidence(la,lb,ha,hb,
        torch.from_numpy(np.asarray(targets_a,dtype=np.int64)),
        torch.from_numpy(np.asarray(targets_b,dtype=np.int64)),kinds)
    receipt["checkpoint"]=str(a.checkpoint)
    receipt["moe_route_geometry"]=cfg.moe_route_geometry
    receipt["semantic_targets_provided"]=True
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt,indent=2))


if __name__=="__main__":
    main()
