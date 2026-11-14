"""Q3 same-source teacher-free runtime: prefill and actual KV+history generation.

Requires already-exported independent pure C2 checkpoint. Compare its logits,
strict student-only checkpoint logits, and complete next-token generation
trace. Reports GPU allocated memory and *observed*, not theoretical, tokens/s.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from models.model import LeechGPT
from scripts.verify_trained_federated_moe import checkpoint_parts,infer_config,_load_checkpoint_weights_only
from scripts.train_guarded_moe_continuation import sha256_file

def load_source(path,expect_guarded):
    state,info=checkpoint_parts(path)
    cfg=infer_config(state,info,None)
    if (cfg.moe_guarded_ffn!=expect_guarded or cfg.moe_experts<2 or
            not cfg.moe_deployed_parity):
        raise ValueError("model guard/parity architecture does not match expected")
    model=LeechGPT(cfg).eval()
    model.load_state_dict(state,strict=True)
    if not expect_guarded and any(block.ffn is not None for block in model.blocks):
        raise AssertionError("teacher-free pure C2 unexpectedly contains dense FFN")
    return model

@torch.no_grad()
def run_model(model,rows,*,device,batch,new_tokens,guarded):
    model.to(device).eval()
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        resident=torch.cuda.memory_allocated(device)
    else:
        resident=None
    start=time.perf_counter()
    n=0
    ce_sum=0.
    with torch.no_grad():
        for left in range(0,len(rows),batch):
            ids=torch.as_tensor(np.asarray(rows[left:left+batch]),device=device,dtype=torch.long)
            logits=model(ids,student_only=guarded)[0]
            if not bool(torch.isfinite(logits).all()):
                raise AssertionError("nonfinite original model logits")
            loss=torch.nn.functional.cross_entropy(
                logits[:,:-1,:].transpose(1,2),ids[:,1:],reduction="sum")
            ce_sum+=float(loss)
            n+=ids.shape[0]*(ids.shape[1]-1)
        if str(device).startswith("cuda"):torch.cuda.synchronize(device)
        prefill_s=time.perf_counter()-start
        # Genuine greedy autoregressive extension on exactly one real source
        # token. No synthetic input or teacher logits used for the next token.
        token=torch.as_tensor([[int(rows[0,0])]],device=device,dtype=torch.long)
        cache=None;history=None;decoded=[]
        logit_hash=hashlib.sha256()
        history_hash=hashlib.sha256()
        step_start=time.perf_counter()
        for t in range(new_tokens):
            out=model(token,past_key_values=cache,past_expert_histories=history,
                      use_cache=True,return_expert_histories=True,
                      student_only=guarded)
            logits=out[0][:,-1,:]
            if not bool(torch.isfinite(logits).all()):
                raise AssertionError("nonfinite next-token logits")
            token=logits.argmax(dim=-1,keepdim=True)
            cache,history=out[3],out[4]
            logit_hash.update(logits.detach().float().cpu().contiguous().numpy().astype("<f4").tobytes())
            if history is None or any(h is None for h in history):
                raise AssertionError("missing source expert history during autoregressive generation")
            for layer_history in history:
                history_hash.update(layer_history.detach().float().cpu().contiguous().numpy().astype("<f4").tobytes())
            decoded.append(int(token[0,0]))
        if str(device).startswith("cuda"):torch.cuda.synchronize(device)
        generation_s=time.perf_counter()-step_start
    return {"ce_nats":ce_sum/n,"target_tokens":n,
        "prefill_seconds":prefill_s,"prefill_target_tokens_per_second":n/prefill_s,
        "generated_token_ids":decoded,"generated_tokens":len(decoded),
        "per_step_logits_fp32_sha256":logit_hash.hexdigest(),
        "per_step_expert_histories_fp32_sha256":history_hash.hexdigest(),
        "generation_seconds":generation_s,
        "greedy_generated_tokens_per_second":len(decoded)/generation_s,
        "memory_loaded_allocated_bytes":resident,
        "memory_peak_allocated_bytes":(
            torch.cuda.max_memory_allocated(device) if str(device).startswith("cuda") else None)}

def run(a):
    if a.output.exists():raise FileExistsError("refuse overwrite inference receipt")
    if a.batch<1 or a.ce_tolerance<0:raise ValueError("invalid native measurement parameters")
    if a.device not in ("cuda","cuda:0") or not torch.cuda.is_available():
        raise ValueError("native HIP/CUDA only")
    saved=_load_checkpoint_weights_only(a.pure)
    export_receipt=saved.get("teacher_stripped_bidi_receipt")
    if not isinstance(export_receipt,dict):
        raise ValueError("pure checkpoint missing original teacher-stripped source receipt")
    if (export_receipt.get("source_candidate_sha256")!=sha256_file(a.guarded) or
        export_receipt.get("original_tokens_sha256")!=sha256_file(a.token_ids)):
        raise ValueError("pure checkpoint does not descend from this guarded/token source")
    rows=np.load(a.token_ids,allow_pickle=False)
    if (rows.ndim!=2 or not len(rows) or rows.shape[1]<2 or
            not np.issubdtype(rows.dtype,np.integer)):
        raise ValueError("invalid real source token rows")
    results=[]
    for path,guarded,label in ((a.guarded,True,"guarded_student_only"),
                               (a.pure,False,"teacher_stripped_student")):
        model=load_source(path,guarded).to(a.device).eval()
        if a.generate<1 or a.generate>=model.cfg.block_size:
            raise ValueError("generation budget must be below model block_size")
        if np.any(rows<0) or np.any(rows>=model.cfg.vocab_size):
            raise ValueError("source token IDs outside vocabulary")
        report=run_model(model,rows,batch=a.batch,device=a.device,
                         new_tokens=a.generate,guarded=guarded)
        report.update({"model":label,"source_checkpoint_sha256":sha256_file(path),
                       "original_checkpoint_file_bytes":path.stat().st_size})
        results.append(report)
        del model
        torch.cuda.empty_cache()
    guarded_result,pure_result=results
    matching=(guarded_result["generated_token_ids"]==pure_result["generated_token_ids"] and
              abs(guarded_result["ce_nats"]-pure_result["ce_nats"])<=a.ce_tolerance and
              guarded_result["per_step_logits_fp32_sha256"]==pure_result["per_step_logits_fp32_sha256"] and
              guarded_result["per_step_expert_histories_fp32_sha256"]==pure_result["per_step_expert_histories_fp32_sha256"])
    report={"schema":1,"kind":"native_c2_teacher_free_inference_bidi",
        "token_ids_sha256":sha256_file(a.token_ids),
        "results":results,"same_greedy_autoregressive_tokens":matching,
        "source_student_ce_delta":abs(guarded_result["ce_nats"]-pure_result["ce_nats"]),
        "generation_logit_and_history_trace_sha_match":matching,
        "native_gpu_name":torch.cuda.get_device_name(0),
        "hip":torch.version.hip,"full_source_generation_trace_consistent":matching,
        "memory_measured_with_one_model_resident_at_a_time":True,
        "performance_warning":"Prefill target-token/s and greedy generated-token/s are distinct metrics.",
        "release_admitted":False}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n")
    if not matching:
        raise AssertionError("original guarded vs teacher-stripped generation mismatch")
    print(json.dumps(report,indent=2))
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--guarded",required=True,type=Path)
    p.add_argument("--pure",required=True,type=Path)
    p.add_argument("--token-ids",required=True,type=Path)
    p.add_argument("--batch",type=int,default=1)
    p.add_argument("--generate",type=int,default=8)
    p.add_argument("--ce-tolerance",type=float,default=1e-5)
    p.add_argument("--device",default="cuda")
    p.add_argument("--output",required=True,type=Path)
    run(p.parse_args())
if __name__=="__main__":main()
