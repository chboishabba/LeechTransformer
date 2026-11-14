"""Q3: physically remove frozen teacher FFNs and validate student-only BIDI.

Original source candidate retains dense weights for guarded fallback. Export a
non-guarded C2 model with NO dense FFN tensors, strict-load independent model,
and check actual logits and next-token CE. Quality promotion is independent.
"""
from __future__ import annotations
import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from models.model import LeechGPT
from scripts.verify_trained_federated_moe import checkpoint_parts,infer_config,_load_checkpoint_weights_only
from scripts.train_guarded_moe_continuation import sha256_file
from scripts.reproduce_c2_quality_and_layers import numeric_ce

def strip_state(state,config):
    if not config.moe_guarded_ffn or config.moe_experts<2 or not config.moe_deployed_parity:
        raise ValueError("complete guarded C2+even original checkpoint required")
    if not any(".quality_gate" in k for k in state):
        raise ValueError("no original guarded gates found")
    # Only original *frozen dense fallback* and quality gates are removable.
    stripped={k:v for k,v in state.items()
              if ".ffn." not in k and ".quality_gate" not in k}
    for i in range(config.n_layers):
        if not any(k.startswith(f"blocks.{i}.moe.") for k in stripped):
            raise ValueError("missing source MoE layer "+str(i))
        if not any(k.startswith(f"blocks.{i}.even_student.") for k in stripped):
            raise ValueError("missing original even channel "+str(i))
    new_cfg=dataclasses.replace(config,moe_guarded_ffn=False)
    # Instantiate an independent source student, not a runtime mock.
    pure=LeechGPT(new_cfg).eval()
    pure.load_state_dict(stripped,strict=True)
    return pure,new_cfg,stripped

@torch.no_grad()
def compare_bidi(guarded,pure,rows,batch,device):
    guarded.eval().to(device)
    pure.eval().to(device)
    maxerr=0.;maxrel=0.;count=0
    for start in range(0,len(rows),batch):
        ids=torch.as_tensor(np.asarray(rows[start:start+batch]),
                            device=device,dtype=torch.long)
        g=guarded(ids,student_only=True)[0]
        p=pure(ids)[0]
        if g.shape!=p.shape or not bool(torch.isfinite(g).all() and torch.isfinite(p).all()):
            raise AssertionError("pure student forward shape/finite mismatch")
        delta=(g-p).abs()
        maxerr=max(maxerr,float(delta.max()))
        maxrel=max(maxrel,float((delta/(1+g.abs())).max()))
        count+=ids.shape[0]
    return {"rows":count,"max_logits_abs":maxerr,"max_logits_scaled_abs":maxrel}

def export(a):
    if a.output.exists() or a.output.with_suffix(".json").exists():
        raise FileExistsError("refuse overwrite standalone C2 export")
    if a.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("native GPU not available")
    whole=_load_checkpoint_weights_only(a.candidate)
    source_receipt=whole.get("quality_curriculum_receipt")
    if isinstance(source_receipt,dict) and source_receipt.get("kind")==(
        "native_gpu_cumulative_student_quality_curriculum"):
        heldout_hash=source_receipt.get("heldout_ids_sha256")
        teacher_reference=source_receipt.get("teacher_heldout_ce")
    else:
        source_receipt=whole.get("warmstart_repair_receipt")
        if not isinstance(source_receipt,dict) or source_receipt.get("kind")!=(
            "native_c2_bounded_early_even_warmstart"):
            raise ValueError("source curriculum or traced warm-start receipt required")
        heldout_hash=source_receipt.get("source_heldout_sha256")
        teacher_reference=source_receipt.get("source_teacher_heldout_ce")
    if sha256_file(a.token_ids)!=heldout_hash:
        raise ValueError("export token file is not source independent heldout")
    if not isinstance(teacher_reference,(int,float)) or teacher_reference<=0:
        raise ValueError("missing original heldout teacher CE")
    state,info=checkpoint_parts(a.candidate)
    cfg=infer_config(state,info,None)
    source=LeechGPT(cfg).eval()
    source.load_state_dict(state,strict=True)
    pure,new_cfg,removed=strip_state(state,cfg)
    rows=np.load(a.token_ids,allow_pickle=False)
    if (rows.ndim!=2 or not len(rows) or rows.shape[1]<2 or
        not np.issubdtype(rows.dtype,np.integer) or
        np.any(rows<0) or np.any(rows>=cfg.vocab_size)):
        raise ValueError("invalid original heldout token rows")
    start=time.perf_counter()
    state_file_bytes=a.candidate.stat().st_size
    old_tensors=sum(v.numel()*v.element_size() for v in state.values())
    new_tensors=sum(v.numel()*v.element_size() for v in removed.values())
    parity=compare_bidi(source,pure,rows,batch=a.batch,device=a.device)
    if parity["max_logits_abs"]>a.atol:
        raise AssertionError("teacher-free model logits do not reproduce guarded student-only")
    source_scores=numeric_ce(source,rows,batch=a.batch,device=a.device,mode="student")
    pure_scores=numeric_ce(pure,rows,batch=a.batch,device=a.device,mode="teacher")
    if abs(source_scores["ce_nats"]-pure_scores["ce_nats"])>a.ce_tolerance:
        raise AssertionError("source and stripped CE differ")
    # Original heldout teacher score is read from the immutable candidate
    # receipt, never accepted as a user-supplied promotion threshold.
    quality_pass=(pure_scores["ce_nats"]<=teacher_reference+a.quality_budget)
    report={
        "schema":1,"kind":"original_c2_teacher_stripped_checkpoint_bidi",
        "source_candidate_sha256":sha256_file(a.candidate),
        "original_tokens_sha256":sha256_file(a.token_ids),
        "rows":len(rows),"source_student_ce":source_scores["ce_nats"],
        "teacher_free_pure_ce":pure_scores["ce_nats"],
        "teacher_ce_reference":teacher_reference,
        "original_curriculum_heldout_sha256":heldout_hash,
        "source_training_receipt_kind":source_receipt["kind"],
        "source_full_state_tensor_bytes":old_tensors,
        "teacher_stripped_tensor_bytes":new_tensors,
        "tensor_bytes_removed":old_tensors-new_tensors,
        "tensor_storage_reduction_fraction":1-new_tensors/old_tensors,
        "source_checkpoint_file_bytes":state_file_bytes,
        "pure_forward_does_not_have_dense_ffn":True,
        "source_student_only_output_parity":parity,
        "quality_pass":quality_pass,
        "model_export_parity_pass":True,
        "release_admitted":False,
        "release_note":"Standalone BIDI and quality threshold do not establish multi-seed or deployment acceptance.",
        "elapsed_validation_seconds":time.perf_counter()-start,
        "original_checkpoint_kept_unchanged":True}
    # Explicitly never mutate original candidate. Save new model path only.
    a.output.parent.mkdir(parents=True,exist_ok=True)
    torch.save({"model_state_dict":{k:v.cpu() for k,v in removed.items()},
        "config":dataclasses.asdict(new_cfg),
        "teacher_stripped_bidi_receipt":report},a.output)
    report["export_sha256"]=sha256_file(a.output)
    report["export_file_bytes"]=a.output.stat().st_size
    a.output.with_suffix(".json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps(report,indent=2,sort_keys=True))
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate",required=True,type=Path)
    p.add_argument("--token-ids",required=True,type=Path)
    p.add_argument("--quality-budget",type=float,default=.02)
    p.add_argument("--batch",type=int,default=1)
    p.add_argument("--atol",type=float,default=1e-5)
    p.add_argument("--ce-tolerance",type=float,default=1e-5)
    p.add_argument("--device",default="cuda")
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.quality_budget<0 or a.batch<1 or a.atol<0 or a.ce_tolerance<0:
        raise ValueError("invalid validation thresholds")
    export(a)
if __name__=="__main__":main()
