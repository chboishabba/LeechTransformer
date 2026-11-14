"""Real source full-student recurrent-history trace on actual token sequence.

Unlike the teacher-forced local MoE(x) diagnostic, this consumes source
token IDs step by step with actual KV + expert-history continuations.
Tracks per-layer magnitude, boundary margins and finite failures; never
normalises, clips, modifies source weights or declares stability by fiat.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import numpy as np
import torch
from scripts.reproduce_c2_quality_and_layers import load_trained
from scripts.screen_c2_history_even_repair import interventions
from scripts.train_guarded_moe_continuation import sha256_file

@torch.no_grad()
def trace(model,token_ids,*,device,greedy=False,max_steps=32):
    if token_ids.ndim!=1 or not len(token_ids) or max_steps<1:
        raise ValueError("source token row and positive token budget required")
    if torch.any(token_ids<0) or torch.any(token_ids>=model.cfg.vocab_size):
        raise ValueError("source token outside original vocabulary")
    model.to(device).eval()
    lengths=min(max_steps,len(token_ids),model.cfg.block_size)
    cache=None;history=None;records=[]
    route_events=[[] for _ in model.blocks]
    hooks=[]
    for layer,block in enumerate(model.blocks):
        def record(module,inputs,out,layer=layer):
            # Router applies per-expert to [batch,experts,features].
            route_events[layer].append(out.detach().clone())
        hooks.append(block.moe.router.register_forward_hook(record))
    try:
        current=token_ids[0:1].to(device).reshape(1,1)
        for t in range(lengths):
            for events in route_events:events.clear()
            out=model(current,past_key_values=cache,past_expert_histories=history,
                use_cache=True,return_expert_histories=True,student_only=True)
            logits=out[0][:,-1,:]
            cache,history=out[3],out[4]
            if not torch.isfinite(logits).all():
                raise AssertionError(f"nonfinite actual student logit at token {t}")
            layer_stats=[]
            for i,h in enumerate(history):
                if h is None or not torch.isfinite(h).all():
                    raise AssertionError(f"nonfinite actual recurrent expert history layer {i} token {t}")
                # The source selector scores router(features) + expert_bias.
                # A raw Linear-hook margin would misrepresent the physical
                # top-k boundary whenever expert_bias is nonzero.
                if len(route_events[i])!=1:
                    raise AssertionError("expected one actual source router selection per layer/token")
                block=model.blocks[i]
                score=route_events[i][0].reshape(-1)+block.moe.expert_bias
                topk=block.moe.top_k
                if score.numel() < topk+1:
                    margin=None
                else:
                    sorted_score=torch.sort(score,descending=True).values
                    margin=float(sorted_score[topk-1]-sorted_score[topk])
                layer_stats.append({"layer":i,
                   "history_rms":float(h.double().square().mean().sqrt()),
                   "history_max_abs":float(h.abs().max()),
                   "router_topk_boundary_margin":margin,
                   "router_calls":len(route_events[i])})
            next_greedy=int(logits.argmax(dim=-1))
            records.append({"step":t+1,"input_token":int(current.item()),
                "next_greedy_token":next_greedy,
                "layers":layer_stats})
            next_token=next_greedy if greedy else (
                int(token_ids[t+1]) if t+1<lengths else next_greedy)
            current=torch.as_tensor([[next_token]],device=device,dtype=torch.long)
    finally:
        for h in hooks:h.remove()
    return {"kind":"actual_c2_recurrent_student_kv_history_trace",
        "steps":len(records),"token_policy":"greedy" if greedy else "original_teacher_forced_tokens",
        "no_weights_changed":True,"source_actual_cache_and_history":True,
        "records":records}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate",type=Path,required=True)
    p.add_argument("--teacher",type=Path,required=True)
    p.add_argument("--config",type=Path)
    p.add_argument("--token-ids",type=Path,required=True)
    p.add_argument("--steps",type=int,default=32)
    p.add_argument("--greedy",action="store_true")
    p.add_argument("--late-cap",type=float)
    p.add_argument("--late-scale",type=float,default=1.)
    p.add_argument("--layer0-even-scale",type=float,default=1.)
    p.add_argument("--layer1-even-scale",type=float,default=1.)
    p.add_argument("--device",default="cuda")
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError("refuse overwrite source history")
    if a.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("native GPU requested but unavailable")
    _,model=load_trained(a.candidate,a.teacher,a.config)
    rows=np.load(a.token_ids,allow_pickle=False)
    if rows.ndim!=2 or not len(rows):
        raise ValueError("expected original nonempty source 2D token array")
    row=torch.as_tensor(np.asarray(rows[0]),dtype=torch.long)
    even_scales={}
    if a.layer0_even_scale!=1.:even_scales[0]=a.layer0_even_scale
    if a.layer1_even_scale!=1.:even_scales[1]=a.layer1_even_scale
    with interventions(model,late_cap=a.late_cap,
                       late_scale=a.late_scale,
                       even_scales=even_scales):
        result=trace(model,row,device=a.device,greedy=a.greedy,max_steps=a.steps)
    result["source_reversible_intervention"]={
        "late_history_cap":a.late_cap,
        "late_history_scale":a.late_scale,
        "even_scales":even_scales}
    if a.late_cap is not None and a.late_scale==1.:
        for step in result["records"]:
            for layer_data in step["layers"]:
                if layer_data["layer"] in (10,11):
                    if layer_data["history_max_abs"]>a.late_cap+1e-4:
                        raise AssertionError("measured capped state exceeds maximum coordinate bound")
    result.update({"source_checkpoint_sha256":sha256_file(a.candidate),
                   "source_tokens_sha256":sha256_file(a.token_ids)})
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k!="records"},indent=2))
if __name__=="__main__":main()
