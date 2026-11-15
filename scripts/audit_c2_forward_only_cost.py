"""Reconcile real C2 curriculum wall time with source training operations.

Observed CE is an exact cross-entropy statistic of logits+targets; model
training trajectories cannot be deduced from objective definition alone.
Forward-only probes may avoid reverse AD but add forward work. This auditor
never fabricates kernel timing or infers a speedup from the 80.1min wall time.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path

def audit(receipt, wall_minutes, fwd_ms=None, bwd_ms=None, probes=None):
    if receipt.get("kind")!="native_gpu_cumulative_student_quality_curriculum":
        raise ValueError("not an observed native C2 curriculum receipt")
    if not math.isfinite(wall_minutes) or wall_minutes<=0:raise ValueError("invalid measured wall time")
    teacher=receipt.get("teacher_heldout_ce")
    student=receipt.get("full_student_only_heldout_ce")
    if not all(type(v) in (int,float) and math.isfinite(v) for v in (teacher,student)):
        raise ValueError("missing heldout teacher/student CE")
    stages=receipt.get("stages")
    if not isinstance(stages,list) or not stages:raise ValueError("missing stage data")
    rows=[];total_steps=0;total_loop_seconds=0
    for i,stage in enumerate(stages,1):
        perf=stage.get("performance",{})
        epochs=stage.get("epoch_scores")
        if not isinstance(epochs,list) or not epochs:raise ValueError("missing epoch receipt")
        n=perf.get("steps")
        t=perf.get("loop_elapsed_seconds")
        if n is None or t is None or n<=0 or t<=0:
            raise ValueError("missing actual loop timing for stage "+str(i))
        total_steps+=n;total_loop_seconds+=t
        rows.append({"stage":i,"active_layers":stage.get("active_layers"),
                     "optimizer_steps":n,"loop_seconds":t,
                     "best_development_ce":stage.get("best_development_ce"),
                     "dev_epoch_trajectory":[r.get("focused_dev_ce") for r in epochs]})
    wall_s=wall_minutes*60
    result={
        "kind":"native_c2_forward_only_wall_cost_audit",
        "observed_wall_seconds":wall_s,
        "observed_wall_minutes":wall_minutes,
        "receipt_reported_elapsed_seconds":receipt.get("elapsed_seconds"),
        "reported_per_stage_loop_seconds_sum":total_loop_seconds,
        "report_to_user_wall_difference_seconds":wall_s-total_loop_seconds,
        "total_recorded_optimizer_steps":total_steps,
        "observed_whole_run_seconds_per_optimizer_step_including_other_work":wall_s/total_steps,
        "reported_stage_loop_seconds_per_optimizer_step":total_loop_seconds/total_steps,
        "stages":rows,
        "teacher_heldout_ce":teacher,
        "student_heldout_ce":student,
        "heldout_gap_nats":student-teacher,
        "teacher_perplexity":math.exp(teacher),
        "student_perplexity":math.exp(student),
        "student_to_teacher_perplexity_ratio":math.exp(student-teacher),
        "observed_model_math_determined_from_ce_alone":False,
        "backward_pass_fraction_measured":False,
        "per_forward_runtime_measured":fwd_ms is not None,
        "gradient_free_speedup_claimed":False,
        "timing_warning":"Stage loop includes interepoch evaluation; wall minus sum also includes initialization, stage selection, final heldout and saving. No per-kernel breakdown."
    }
    if fwd_ms is not None or bwd_ms is not None or probes is not None:
        if (fwd_ms is None or bwd_ms is None or probes is None or
            fwd_ms<=0 or bwd_ms<0 or probes<1 or
            not all(math.isfinite(v) for v in (fwd_ms,bwd_ms))):
            raise ValueError("supply all measured forward_ms, backward_ms, probes")
        # Only for same-graph, same-batch per-step microbench, without caching.
        # Current curriculum actually runs teacher, hybrid, student forwards.
        result["hypothetical_same_graph_forward_only_extra_compute_ms"]=probes*fwd_ms
        result["measured_backward_compute_ms"]=bwd_ms
        result["probes_forward_equivalent_break_even"]=bwd_ms/fwd_ms
        result["probe_compute_less_than_backward"]=probes*fwd_ms<bwd_ms
        result["probe_cost_is_end_to_end_prediction"]=False
        result["probe_warning"]="Must account for frozen teacher, hybrid, student, cached prefixes, update assembly, probes per stage, token count, memory, variance and target quality; not a measured replacement."
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--receipt",type=Path,required=True)
    p.add_argument("--wall-minutes",type=float,default=80.1)
    p.add_argument("--measured-forward-ms",type=float)
    p.add_argument("--measured-backward-ms",type=float)
    p.add_argument("--probes",type=int)
    p.add_argument("--output",type=Path)
    a=p.parse_args()
    r=audit(json.loads(a.receipt.read_text()),a.wall_minutes,
            a.measured_forward_ms,a.measured_backward_ms,a.probes)
    out=json.dumps(r,indent=2,sort_keys=True,allow_nan=False)+"\n"
    if a.output:
        a.output.parent.mkdir(parents=True,exist_ok=True)
        if a.output.exists():raise FileExistsError("refuse overwrite of prior cost audit")
        a.output.write_text(out)
    print(out,end="")
if __name__=="__main__":main()
