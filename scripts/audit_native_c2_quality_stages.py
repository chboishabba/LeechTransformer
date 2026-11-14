"""Summarise real C2 progressive quality curriculum receipts and rank next cuts.

No model or GPU execution. Do not infer improvements or source equivalence from
a JSON summary. Uses only the exact observed training receipt.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path

def analyse(record):
    if record.get("kind")!="native_gpu_cumulative_student_quality_curriculum":
        raise ValueError("not an original native C2 curriculum receipt")
    stages=record.get("stages")
    if not isinstance(stages,list) or not stages:
        raise ValueError("missing actual stage receipts")
    teacher=record.get("teacher_heldout_ce")
    student=record.get("full_student_only_heldout_ce")
    if not all(type(x) in (int,float) and math.isfinite(x) for x in (teacher,student)):
        raise ValueError("missing finite final source quality scores")
    rows=[]
    prev=None
    all_ids=set()
    for i,s in enumerate(stages):
        active=s.get("active_layers")
        if (not isinstance(active,(list,tuple)) or
            not active or len(active)!=len(set(active)) or
            not all(type(x)==int and x>=0 for x in active)):
            raise ValueError("invalid source active-layer receipt")
        current=set(active)
        if not all_ids.issubset(current) or (i and all_ids==current):
            raise ValueError("noncumulative original stage receipt")
        newly_added=sorted(current-all_ids)
        best=s.get("best_development_ce")
        epochs=s.get("epoch_scores")
        if (not type(best) in (int,float) or not math.isfinite(best)
            or not isinstance(epochs,list) or not epochs):
            raise ValueError("missing stage trajectory")
        observed=[e.get("focused_dev_ce") for e in epochs]
        if any(type(x) not in (int,float) or not math.isfinite(x) for x in observed):
            raise ValueError("invalid stage development loss")
        if abs(min(observed)-best)>1e-5:
            raise ValueError("stage best CE does not match epoch trajectory")
        rows.append({
            "stage":s.get("stage",i+1),
            "active_layers":sorted(current),
            "newly_added_layers":newly_added,
            "best_dev_ce":float(best),
            "change_vs_previous_stage_dev_ce":None if prev is None else float(best-prev),
            "best_epoch":s.get("best_epoch"),
            "first_epoch_dev_ce":float(observed[0]),
            "last_epoch_dev_ce":float(observed[-1]),
            "best_epoch_vs_final_epoch_ce_improvement":float(observed[-1]-best),
            "epochs_recorded":len(observed),
        })
        prev=float(best);all_ids=current
    worsening=[r for r in rows if r["change_vs_previous_stage_dev_ce"] is not None]
    worsening.sort(key=lambda x:x["change_vs_previous_stage_dev_ce"],reverse=True)
    scan=record.get("initial_single_layer_development_scan")
    ranked=[]
    if isinstance(scan,list):
        ranked=sorted([
            {"layer":s.get("layer"),
             "initial_single_layer_dev_ce":s.get("initial_single_layer_development_ce")}
            for s in scan
        ],key=lambda x:x["initial_single_layer_dev_ce"],reverse=True)
    return {
        "kind":"native_c2_quality_stage_audit",
        "source_teacher_sha256":record.get("source_teacher_sha256"),
        "initial_student_sha256":record.get("initial_student_sha256"),
        "heldout_ids_sha256":record.get("heldout_ids_sha256"),
        "fit_rows":record.get("fit_rows"),
        "development_rows":record.get("development_rows"),
        "heldout_rows":record.get("heldout_rows"),
        "native_gpu_elapsed_seconds":record.get("elapsed_seconds"),
        "teacher_heldout_ce":teacher,
        "student_heldout_ce":student,
        "student_minus_teacher_heldout_ce":student-teacher,
        "quality_promotion":record.get("full_student_quality_pass") is True,
        "stage_trajectories":rows,
        "largest_stage_dev_regressions":worsening[:4],
        "weakest_initial_isolated_layers":ranked[:4],
        "teacher_free_checkpoint_exported":False,
        "claim":"Source-observed dev trajectory, not a causal attribution of errors to specific layers."}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--receipt",required=True,type=Path)
    p.add_argument("--output",type=Path)
    a=p.parse_args()
    report=analyse(json.loads(a.receipt.read_text()))
    out=json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n"
    if a.output:
        a.output.parent.mkdir(parents=True,exist_ok=True)
        if a.output.exists():
            raise FileExistsError("refuse overwrite audit")
        a.output.write_text(out)
    print(out,end="")
if __name__=="__main__":
    main()
