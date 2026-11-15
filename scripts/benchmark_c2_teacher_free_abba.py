"""Order-balanced native RX580 source C2 inference comparison.

Reuses exact guarded-vs-pure model code from benchmark_c2_teacher_free.
ABBA alternation is used so first-run compilation/caches/thermal effects do
not become a false claim that removal of unused FFN compute caused speedup.
Hashes entire generated logits/history (outside aggregate logic). This
measures tested source path, not an end-to-end deployment/general speed law.
"""
from __future__ import annotations
import argparse
import json
import math
import statistics
from pathlib import Path

import numpy as np
import torch
from scripts.benchmark_c2_teacher_free import load_source,run_model
from scripts.verify_trained_federated_moe import _load_checkpoint_weights_only
from scripts.train_guarded_moe_continuation import sha256_file

def summarize(samples):
    if len(samples)<4:
        raise ValueError("require at least one ABBA cycle")
    seconds=[float(item["generation_seconds"]) for item in samples]
    rates=[float(item["greedy_generated_tokens_per_second"]) for item in samples]
    return {"repetitions":len(samples),
            "median_generation_seconds":statistics.median(seconds),
            "median_generated_tokens_per_second":statistics.median(rates),
            "min_generated_tokens_per_second":min(rates),
            "max_generated_tokens_per_second":max(rates),
            "all_forward_trace_identical":(
                len({(s["per_step_logits_fp32_sha256"],
                      s["per_step_expert_histories_fp32_sha256"],
                      tuple(s["generated_token_ids"])) for s in samples})==1),
            "median_loaded_model_allocator_bytes":statistics.median(
                [s["memory_loaded_allocated_bytes"] for s in samples]),
            "median_peak_allocator_bytes":statistics.median(
                [s["memory_peak_allocated_bytes"] for s in samples])}

def audit(a):
    if a.output.exists():raise FileExistsError("refuse overwrite prior ABBA receipt")
    if a.device not in ("cuda","cuda:0") or not torch.cuda.is_available():
        raise RuntimeError("native GPU required; no CPU performance substitution")
    if a.cycles<2 or a.batch<1 or a.generate<1:
        raise ValueError("at least two ABBA cycles and positive batch/generation needed")
    pure_object=_load_checkpoint_weights_only(a.pure)
    if not isinstance(pure_object,dict):
        raise ValueError("expected complete pure checkpoint")
    source=pure_object.get("teacher_stripped_bidi_receipt",{})
    if (source.get("source_candidate_sha256")!=sha256_file(a.guarded) or
        source.get("original_tokens_sha256")!=sha256_file(a.token_ids)):
        raise ValueError("wrong source candidate / test token pairing")
    rows=np.load(a.token_ids,allow_pickle=False)
    if (rows.ndim!=2 or len(rows)<1 or rows.shape[1]<2 or
        not np.issubdtype(rows.dtype,np.integer)):
        raise ValueError("invalid source token rows")
    first=load_source(a.guarded,True)
    second=load_source(a.pure,False)
    if (first.cfg.block_size!=second.cfg.block_size or
        first.cfg.vocab_size!=second.cfg.vocab_size):
        raise ValueError("checkpoint architecture mismatch")
    if a.generate>=first.cfg.block_size or np.any(rows<0) or np.any(rows>=first.cfg.vocab_size):
        raise ValueError("invalid generation/context bounds")
    # Load both separately on the GPU, but leave them resident to avoid CPU
    # host->device weight transfers inside warmup and timed loops. Compared
    # memory numbers are model-local peaks: measurements with both resident
    # cannot attribute absolute isolated allocator residency. Report honestly.
    first=first.to(a.device).eval()
    second=second.to(a.device).eval()
    models={"guarded":(first,True),"pure":(second,False)}
    # Warm both independently, and synchronize each warmed-up run.
    for label in ("guarded","pure"):
        m,g=models[label]
        run_model(m,rows[:1],device=a.device,batch=1,new_tokens=a.generate,guarded=g)
    torch.cuda.synchronize(a.device)
    results={"guarded":[],"pure":[]}
    orders=[]
    for cycle in range(a.cycles):
        order=("guarded","pure","pure","guarded") if cycle%2==0 else ("pure","guarded","guarded","pure")
        orders.extend(order)
        for label in order:
            m,g=models[label]
            rec=run_model(m,rows,device=a.device,batch=a.batch,
                          new_tokens=a.generate,guarded=g)
            results[label].append(rec)
    ref=results["guarded"][0]
    source_equal=all(
        (v["generated_token_ids"]==ref["generated_token_ids"] and
         v["per_step_logits_fp32_sha256"]==ref["per_step_logits_fp32_sha256"] and
         v["per_step_expert_histories_fp32_sha256"]==
           ref["per_step_expert_histories_fp32_sha256"] and
         abs(v["ce_nats"]-ref["ce_nats"])<1e-5)
        for group in results.values() for v in group)
    if not source_equal:
        raise AssertionError("ABBA generation/teacher-free history trace drift")
    guarded=summarize(results["guarded"])
    pure=summarize(results["pure"])
    ratio=(pure["median_generated_tokens_per_second"]/
           guarded["median_generated_tokens_per_second"])
    report={"schema":1,"kind":"native_c2_guarded_pure_abba_warmed",
        "guarded_sha256":sha256_file(a.guarded),
        "pure_sha256":sha256_file(a.pure),
        "original_tokens_sha256":sha256_file(a.token_ids),
        "cycles":a.cycles,"iterations_per_model":len(results["pure"]),
        "execution_order":orders,"generated_tokens_per_iteration":a.generate,
        "guarded":guarded,"pure":pure,
        "warmed_median_pure_over_guarded_tps_ratio":ratio,
        "tested_full_logits_and_history_state_trace_equal":source_equal,
        "gpu":torch.cuda.get_device_name(0),"hip":torch.version.hip,
        "both_models_gpu_resident_during_measurement":True,
        "allocator_memory_stats_not_independent_model_residency":True,
        "clock_thermal_energy_controls_not_measured":True,
        "causal_teacher_removal_speedup_established":False,
        "release_admitted":False}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print(json.dumps(report,indent=2,sort_keys=True,allow_nan=False))
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--guarded",type=Path,required=True)
    p.add_argument("--pure",type=Path,required=True)
    p.add_argument("--token-ids",type=Path,required=True)
    p.add_argument("--generate",type=int,default=16)
    p.add_argument("--batch",type=int,default=1)
    p.add_argument("--cycles",type=int,default=3)
    p.add_argument("--device",default="cuda")
    p.add_argument("--output",type=Path,required=True)
    audit(p.parse_args())
if __name__=="__main__":main()
